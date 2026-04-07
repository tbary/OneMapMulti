"""
This is the core mapping module, which contains the OneMap class.
"""
from transforms3d.derivations.angle_axes import point

from .nav_goals.frontier import Frontier, detect_frontiers, get_frontier_midpoint
from .nav_goals.clustering import Cluster, cluster_high_similarity_regions
from .nav_goals.navigation_goals import NavGoal
from .projection import Projection

from config import MappingConf
from planning import Planning
from onemap_utils import monochannel_to_inferno_rgb, log_map_rerun, log_dino_embeddings_tsne


from skimage.measure import label
from scipy.ndimage import distance_transform_edt

# enum
from enum import Enum

# NumPy
import numpy as np
import scipy.optimize

# typing
from typing import Tuple, List, Optional, Union, Set, Dict

# rerun
import rerun as rr

# torch
import torch
from torch.nn.functional import normalize

# cv2
import cv2

def rotate_frame(points):
    return [[y, x] for (x, y) in points]

def rotate_pcl(
        pointcloud: torch.Tensor,
        tf_camera_to_episodic: torch.Tensor,
) -> torch.Tensor:
    # TODO We might be interested in a complete 3d rotation if the camera is not perfectly horizontal
    rotation_matrix = tf_camera_to_episodic[:3, :3]

    yaw = torch.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # print(yaw)
    r = torch.tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]], dtype=torch.float32).to("cuda")
    pointcloud[:, :2] = (r @ pointcloud[:, :2].T).T
    return pointcloud

def print_memory_stats(label):
    print(f"\n--- Memory Stats for {label} ---")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")

class FusionType(Enum):
    EMA = "EMA"
    SPATIAL = "Spatial"

class OneMap:
    feature_map: torch.Tensor  # map where first dimension is x direction, second dimension is y, and last direction is
    # feature_dim
    obstacle_map: torch.Tensor  # map where first dimension is x direction, second dimension is y, and last direction is
    # obstacle likelihood
    navigable_map: np.ndarray  # binary traversability map where first dimension is x direction, second dimension is y
    # navigable likelihood
    fully_explored_map: np.ndarray  # binary explored map where first dimension is x direction, second dimension is y
    checked_map: np.ndarray  # binary checked map where first dimension is x direction, second dimension is y,
    # can be reset
    confidence_map: torch.Tensor
    checked_conf_map: torch.Tensor
    updated_mask: torch.Tensor  # tracks which cells have been updated, for lazy similarity computation
    blacklisted_nav_goals: List[np.ndarray]
    nav_goals: List[NavGoal]
    last_nav_goal: Union[NavGoal, None]

    def __init__(self,
                 n_agents: int,
                 feature_dim: int,
                 config: MappingConf,
                 fusion_type: FusionType = FusionType.EMA,
                 map_device: str = "cuda",
                 ) -> None:
        """
        :param feature_dim: The dimension of the feature space
        :param n_cells: The number of cells in the x and y direction respectively
        :param size: The size of the map in meters
        :param fusion_type: The type of fusion to use, must be one of FusionType
        """
        assert isinstance(fusion_type, FusionType), "Invalid fusion_type. It should be one of FusionType."

        self.config = config

        self.fusion_type = fusion_type
        self.map_device = map_device

        self.cell_size = self.config.size / self.config.n_points
        self.feature_dim = feature_dim
        self.feature_map = torch.zeros((self.config.n_points, self.config.n_points, feature_dim), dtype=torch.float32).to(self.map_device)

        self.obstacle_map = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.float32)
        col_kernel_size = self.config.n_points / self.config.size * self.config.agent_radius
        col_kernel_size = int(col_kernel_size) + (int(col_kernel_size) % 2 == 0)
        self.navigable_map = np.ones((self.config.n_points, self.config.n_points), dtype=bool)
        self.occluded_map = np.zeros((self.config.n_points, self.config.n_points), dtype=bool)
        self.navigable_kernel = np.ones((col_kernel_size, col_kernel_size), np.uint8)

        self.blacklisted_nav_goals = []
        self.frontier_depth = int(config.frontier_depth / self.cell_size)

        self.fully_explored_map = np.zeros((self.config.n_points, self.config.n_points), dtype=bool)
        self.checked_map = np.zeros((self.config.n_points, self.config.n_points), dtype=bool)

        self.confidence_map = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.float32).to(self.map_device)
        self.checked_conf_map = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.float32).to(self.map_device)

        self.updated_mask = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.bool).to(self.map_device)

        self.agents_poses = np.zeros((n_agents, 3))

        self.similarity_map = None
        self.initializing = True

        print(f"ValueMap initialized. The map contains {self.config.n_points ** 2} cells, each storing {feature_dim} features. The resulting size is {self.feature_map.element_size() * self.feature_map.nelement() / 1024**2} Mb.")

    def reset(self):
        # Reset value map
        self.feature_map = torch.zeros((self.config.n_points, self.config.n_points, self.feature_dim), dtype=torch.float32).to(self.map_device)

        # Reset obstacle map
        self.obstacle_map = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.float32).to(self.map_device)

        # Reset navigable map
        self.navigable_map = np.ones((self.config.n_points, self.config.n_points), dtype=bool)
        self.occluded_map = np.zeros((self.config.n_points, self.config.n_points), dtype=bool)

        # Reset fully explored map
        self.fully_explored_map = np.zeros((self.config.n_points, self.config.n_points), dtype=bool)

        # Reset checked map
        self.checked_map = np.zeros((self.config.n_points, self.config.n_points), dtype=bool)

        # Reset confidence map
        self.confidence_map = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.float32).to(self.map_device)

        # Reset checked confidence map
        self.checked_conf_map = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.float32).to(self.map_device)

        # Reset updated mask
        self.updated_mask = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.bool).to(self.map_device)

        # Reset previous sims
        self.similarity_map = None
        self.initializing = True

        self.agents_poses = np.zeros_like(self.agents_poses)

        self.blacklisted_nav_goals = []

    def reset_updated_mask(self):
        self.updated_mask = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.bool).to(self.map_device)

    def reset_checked_map(self):
        self.checked_map = np.zeros((self.config.n_points, self.config.n_points), dtype=bool)
        self.checked_conf_map = torch.zeros((self.config.n_points, self.config.n_points), dtype=torch.float32)

    def update(self,
               projection: Projection,
               values: torch.Tensor,
               depth: np.ndarray,
               tf_camera_to_episodic: np.ndarray,
               artificial_obstacles: Optional[Set[Tuple[float]]] = []
               ) -> None:
        """
        Updates the map with values by projecting them into the map from depth
        :param values: torch tensor of values. Either a 3D array of shape (feature_dim, hf, wf)
                        or a 1D array of shape (feature_dim)
        :param depth:  numpy array of depth values of shape (h, w)
        :param tf_camera_to_episodic: 4x4 numpy array representing the transformation from camera to episodic
        """
        assert values.shape[0] == self.feature_dim, "Feature dimension of image does not correspond to expected dimension."
        assert len(values.shape) == 3, "Provided Value observation of unsupported format."

        values = values.permute(1, 2, 0)  # feature_dim last for convenience
        projected_submap = projection.project_dense(values, torch.Tensor(depth).to("cuda"), torch.tensor(tf_camera_to_episodic))

        self._fuse_maps(*projected_submap, artificial_obstacles)

    def update_agent_pose(self, new_pose, agent_id):
        self.agents_poses[agent_id] = np.array(new_pose)

    def _fuse_maps(self,
                  confidences_mapped: torch.Tensor,
                  values_mapped: torch.Tensor,
                  obstacle_mapped: torch.Tensor,
                  obstcl_confidence_mapped: torch.Tensor,
                  artificial_obstacles: Optional[Set[Tuple[float]]] = []
                  ) -> None:
        """
        Fuses the mapped values into the value map using the confidence estimates and tracked confidences
        This function takes in sparse tensors of confidences and values, and fuses them into the map, only updating
        the cells that have been updated.
        :param confidences_mapped: torch: sparse COO tensor of confidences
        :param values_mapped: torchL sparse COO tensor of values
        :return:
        """
        if self.fusion_type == FusionType.EMA:
            indices = tuple(confidences_mapped.indices())
            indices_obstacle = tuple(obstacle_mapped.indices())
            confs_new = confidences_mapped.values().data.squeeze()
            confs_old = self.confidence_map[indices]

            confs_old_obs = self.confidence_map[indices_obstacle]

            confidence_denominator = confs_new + confs_old
            weight_1 = torch.nan_to_num(confs_old / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.updated_mask[indices] = True

            self.feature_map[indices] = self.feature_map[indices] * weight_1.unsqueeze(-1) + \
                                                       values_mapped.values().data * weight_2.unsqueeze(-1)

            self.confidence_map[indices] = confidence_denominator

            # we also need to update the checked confidence
            confs_old_checked = self.checked_conf_map[indices]
            confidence_denominator_checked = confs_new + confs_old_checked
            self.checked_conf_map[indices] = confidence_denominator_checked

            # Obstacle Map update
            confs_new = obstcl_confidence_mapped.values().data.squeeze()
            confidence_denominator = confs_new + confs_old_obs
            weight_1 = torch.nan_to_num(confs_old_obs / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.obstacle_map[indices_obstacle] = self.obstacle_map[indices_obstacle] * weight_1 + \
                                                                          obstacle_mapped.values().data.squeeze() * weight_2

            self.occluded_map = (self.obstacle_map > self.config.obstacle_map_threshold).cpu().numpy()
            if len(artificial_obstacles) != 0:
                for obs in artificial_obstacles:
                    self.occluded_map[obs] = True

            self.navigable_map = 1 - cv2.dilate((self.occluded_map).astype(np.uint8), self.navigable_kernel, iterations=1).astype(bool)

            self.fully_explored_map = (1.0 / (self.confidence_map.cpu().numpy() + 1e-8) < self.config.fully_explored_threshold)
            self.checked_map = (1.0 / (self.checked_conf_map.cpu().numpy() + 1e-8) < self.config.checked_map_threshold)

    def set_similarity_map(self, similarity_map: torch.Tensor|None, mask: np.ndarray|None = None) -> None:
        if mask is not None:
            self.similarity_map[mask] = similarity_map
        else:
            self.similarity_map = similarity_map

    def set_frontiers_exploration_score(self, frontiers:List[Frontier]):
        discovery_map = self.navigable_map.copy()
        discovery_map[self.confidence_map.cpu().numpy()==0] = False
        discovery_map[self.fully_explored_map] = False
        
        connected_components = label(discovery_map)
        connected_components_sizes = np.bincount(connected_components.flatten())

        distances, indices = distance_transform_edt(
            connected_components == 0,        # background mask
            return_indices=True
        )

        for frontier in frontiers:
            for fc in frontier.points:
                distance_to_nearest_comp = distances[tuple(fc)]
                if distance_to_nearest_comp <= 1:
                    nearest_comp_label = connected_components[tuple(indices[...,fc[0], fc[1]])]
                    frontier.frontier_explore_score = connected_components_sizes[nearest_comp_label]
                    break
        if self.config.log_rerun:
            log_map_rerun(discovery_map, path="map/discovery")

    def compare_diversity(self, feature_map:np.ndarray, explored_mask:np.ndarray, nav_goal_coords:List[np.ndarray], max_components:int =5, display: bool=False):
        import time
        t1 = time.time()
        # from sklearn.mixture import GaussianMixture
        # from sklearn.neighbors import KernelDensity
        # from sklearn.decomposition import PCA

        # explored_map_features = feature_map[explored_mask & (feature_map.sum(axis=-1) != 0.0)]
        # nav_goal_features = [feature_map[ngc[:, 0], ngc[:, 1]] for ngc in nav_goal_coords]

        # # print(nav_goal_coords.shape, nav_goal_features.shape, explored_map_features.shape)
        # if display:
        #     log_dino_embeddings_tsne([explored_map_features, *nav_goal_features], perplexity=min(len(explored_map_features)-5,30))

        # # pca = PCA(n_components=np.min((50, *explored_map_features.shape)), svd_solver='randomized', random_state=42)
        # # emf_reduced = pca.fit_transform(explored_map_features)
        
        # nav_goal_scores = np.empty(len(nav_goal_features))
        # k = int(np.ceil(0.001*len(explored_map_features)))
        # for i, ngf in enumerate(nav_goal_features):
        #     cosine_similarities = np.matmul(explored_map_features, ngf.T)
        #     point_proximity_scores = np.mean(np.sort(cosine_similarities, axis=0)[-k:], axis=0)

        #     threshold = np.quantile(point_proximity_scores, 0.5)
        #     filt_point_proximity_scores = point_proximity_scores[point_proximity_scores <= threshold]
        #     nav_goal_scores[i] = np.mean(filt_point_proximity_scores)

        # candidate = np.argmin(nav_goal_scores)
        # pts = nav_goal_coords[candidate]
        # rr.log("map/candidate", rr.Points2D(rotate_frame(pts), colors=[[255,255,0]]*pts.shape[0], radii=[1]*pts.shape[0]))
        # print(time.time() - t1)
        # raise ValueError

    def compute_frontiers_and_POIs(self, allow_retry=True):
        """
        Computes the frontiers (at the border from fully explored to confidence > 0),
        and points of interest (high similarity regions within the fully explored, but not checked map)
        :return:
        """
        nav_goals: List[NavGoal] = []

        if self.similarity_map is None:
            return
        
        # Compute the frontiers
        frontiers, unexplored_map, largest_contour = detect_frontiers(
            self.navigable_map.astype(np.uint8),
            self.fully_explored_map.astype(np.uint8),
            self.confidence_map > 0,
            int(1.0 * ((self.config.n_points / self.config.size) ** 2))
        )

        # moreover we compute points of interest. These are high similarity regions within the fully explored,
        # but not checked map
        # For that we make use of the cluster_high_similarity_regions function, and project the points to the
        # navigable map
        adjusted_score = self.similarity_map + 1.0  # only positive scores
        map_def = self.similarity_map
        normalized_map = (map_def - map_def.min()) / (map_def.max() - map_def.min() + 1e-7)
        # TODO This will give us wrong cluster scores, we will need to adjust this to match the frontier scores!
        clusters = cluster_high_similarity_regions(normalized_map, (self.confidence_map > 0.0).cpu().numpy())
        for cluster in clusters:
            cluster.compute_score(adjusted_score)
            if len(self.blacklisted_nav_goals) == 0 or not np.any(
                    np.all(cluster.get_descr_point() == self.blacklisted_nav_goals, axis=1)):
                if ((largest_contour is None or cv2.pointPolygonTest(largest_contour, cluster.center.astype(float),
                                                                        measureDist=True) > -15.0) or
                    self.fully_explored_map[cluster.center[0], cluster.center[1]]) and \
                        (not self.checked_map[cluster.center[0], cluster.center[1]]):
                    nav_goals.append(cluster)
        
        if self.config.log_rerun:
            log_map_rerun(self.confidence_map.cpu().numpy(), path="map/confidence")
            log_map_rerun(unexplored_map, path="map/unexplored")

        frontiers = [f[..., ::-1].squeeze() for f in frontiers]  # need to flip coords for some reason
        adjusted_score_frontier = adjusted_score.copy()

        # set the score of the fully explored map to 0 for the frontiers

        for frontier_points in frontiers:
            frontier_mp = get_frontier_midpoint(frontier_points).astype(np.uint32)
            score, *_ = Planning.compute_reachable_area_score(
                frontier_mp,
                (self.confidence_map > 0).cpu().numpy(),
                adjusted_score_frontier,
                self.frontier_depth)
            frontier_mp = np.round(frontier_mp)
            if len(self.blacklisted_nav_goals) == 0 or not np.any(np.all(frontier_mp == self.blacklisted_nav_goals, axis=1)):
                frontier = Frontier(frontier_midpoint=frontier_mp, points=frontier_points, frontier_score=score, frontier_explore_score=0)
                nav_goals.append(frontier)

        if self.config.log_rerun:
            if len(nav_goals) > 0:
                pts = np.array([nav_goal.get_descr_point() for nav_goal in nav_goals])
                scores = np.array([nav_goal.get_score() for nav_goal in nav_goals])
                rr.log("map/frontiers_and_POIs",
                        rr.Points2D(rotate_frame(pts), colors=np.flip(monochannel_to_inferno_rgb(scores), axis=-1),
                                    radii=[1] * pts.shape[0]))               

        self.set_frontiers_exploration_score([frontier for frontier in nav_goals if type(frontier)==Frontier])

        if len(nav_goals) == 0:
            if not self.initializing and allow_retry:
                self.reset_checked_map()
                return self.compute_frontiers_and_POIs(allow_retry=False)
            return []

        self.initializing = False
        return nav_goals

    def split_frontiers_and_POIs(self, obj_detected: np.ndarray, nav_goals: List[NavGoal])->Dict[int, List[NavGoal]]:
        agent_coords = self.agents_poses[..., :-1]
        objective_coords = np.array([nav_goal.get_descr_point() for nav_goal in nav_goals])

        n_agents = agent_coords.shape[0]
        n_obj = objective_coords.shape[0]
        if n_obj == 0:
            return {a_id:[] for a_id in range(n_agents)}

        # --- Step 0: filter active agents ---
        active_mask = ~obj_detected
        active_agent_ids = np.where(active_mask)[0]

        n_active = len(active_agent_ids)
        if n_active == 0:
            return {}

        active_coords = agent_coords[active_mask]

        # --- Step 1: compute distance matrix (only active agents) ---
        dists = np.linalg.norm(
            active_coords[:, None, :] - objective_coords[None, :, :],
            axis=2
        )
        
        # --- Step 2: balanced assignment ---
        k = n_obj // n_active

        base_obj_count = k * n_active
        base_indices = np.arange(n_obj)[:base_obj_count]

        expanded_dists = np.repeat(dists[:, base_indices], k, axis=0)

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(expanded_dists)

        # --- Step 3: build full assignment dict (ALL agents) ---
        assignments = {i: [] for i in range(n_agents)}
        assigned_mask = np.zeros(n_obj, dtype=bool)

        for r_idx, c_idx in zip(row_ind, col_ind):
            local_agent = r_idx // k              # index in active agents
            agent_id = active_agent_ids[local_agent]  # map back to original ID

            obj_id = base_indices[c_idx]

            assignments[agent_id].append(nav_goals[obj_id])
            assigned_mask[obj_id] = True

        # --- Step 4: assign remaining objectives ---
        remaining_objs = np.where(~assigned_mask)[0]

        for obj_id in remaining_objs:
            # only consider active agents
            closest_local = np.argmin(dists[:, obj_id])
            agent_id = active_agent_ids[closest_local]

            assignments[agent_id].append(nav_goals[obj_id])

        assignments = {k: sorted(v, key=lambda x: x.get_score(), reverse=True) for k, v in assignments.items()}

        return assignments

if __name__ == "__main__":
    pass
