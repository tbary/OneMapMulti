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
from onemap_utils import monochannel_to_inferno_rgb, log_map_rerun


from skimage.measure import label
from scipy.ndimage import distance_transform_edt

# enum
from enum import Enum

# NumPy
import numpy as np
import scipy.optimize

# typing
from typing import Tuple, List, Optional, Union, Set, Dict, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .navigator import Navigator

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
            weight_1 = torch.nan_to_num(confs_old / confidence_denominator).unsqueeze(-1)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator).unsqueeze(-1)

            self.updated_mask[indices] = True

            self.feature_map[indices] = self.feature_map[indices] * weight_1 + values_mapped.values().data * weight_2

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
    
    def _attach_discovery_to_frontier(self, frontiers:List[Frontier]):
        discovery_map = self.navigable_map.copy()
        discovery_map[self.confidence_map.cpu().numpy()==0] = False
        discovery_map[self.fully_explored_map] = False
        
        connected_components = label(discovery_map)

        distances, indices = distance_transform_edt(
            connected_components == 0,        # background mask
            return_indices=True
        )

        for frontier in frontiers:
            for fc in frontier.points:
                distance_to_nearest_comp = distances[tuple(fc)]
                if distance_to_nearest_comp <= 1:
                    nearest_comp_label = connected_components[tuple(indices[...,fc[0], fc[1]])]
                    frontier.discovery_zone.extend(list(np.argwhere(connected_components == nearest_comp_label)))
            frontier.discovery_zone = np.unique(frontier.discovery_zone, axis=0)
                
        if self.config.log_rerun:
            import matplotlib.pyplot as plt
            frontier_colors = [(int(r*255), int(g*255), int(b*255)) for r,g,b in plt.get_cmap('tab20').colors] * (1 + len(frontiers)//20)
            disc_zones = []
            colors = []
            for f, frontier in enumerate(frontiers):
                disc_zones.extend(frontier.discovery_zone)
                colors.extend([frontier_colors[f]]*len(frontier.discovery_zone))
            rr.log("map/zones", rr.Points2D(rotate_frame(disc_zones), colors=colors, radii=[0.5]*len(disc_zones)))
            log_map_rerun(discovery_map, path="map/discovery")

    def set_exploration_scores(self, nav_goals:List[NavGoal], mode:Literal["size", "dist", "diversity", "certainty", "greedy"]="greedy", fallback:Literal["certainty", "greedy"]="greedy"):
        frontiers = [ng for ng in nav_goals if isinstance(ng, Frontier)]
        clusters = [ng for ng in nav_goals if isinstance(ng, Cluster)]

        self._attach_discovery_to_frontier(frontiers)

        if mode == "size":
            for frontier in frontiers:
                frontier.frontier_explore_score = len(frontier.discovery_zone)
            for cluster in clusters:
                cluster.cluster_explore_score = 0
        
        elif mode == "dist":
            raise NotImplementedError("dist is not implemented yet...")
            for cluster in clusters:
                cluster.cluster_explore_score = 0
        
        # if np.random.randint(0,25) == 10:
        #     import uuid
        #     import pickle
        #     name = uuid.uuid4().hex
            
        #     with open(f"results_multi_one/frontiers_sim/{name}-feature_map.pkl", "wb") as f:
        #         pickle.dump(self.feature_map.cpu().numpy(),f)
        #     with open(f"results_multi_one/frontiers_sim/{name}-fully_explored_map.pkl", "wb") as f:
        #         pickle.dump(self.fully_explored_map,f)
        #     with open(f"results_multi_one/frontiers_sim/{name}-frontiers.pkl", "wb") as f:
        #         pickle.dump([frontier.discovery_zone for frontier in frontiers],f)


        # if np.random.randint(0,100) == 50:
        #     arrays = []
        #     feature_map_array = self.feature_map.cpu().numpy()
        #     explored_map_features = feature_map_array[self.fully_explored_map & (feature_map_array.sum(axis=-1) != 0.0)]
        #     for frontier in frontiers:
        #         points = frontier.discovery_zone
        #         if not len(points):
        #             frontier.frontier_explore_score = 0
        #         else:
        #             frontier_features = feature_map_array[points[:, 0], points[:, 1]]
        #             dissim = 1-np.matmul(explored_map_features, frontier_features.T)

        #             dissim_per_frontier_point = np.min(dissim, axis=0)
        #             arrays.append(dissim_per_frontier_point)

        #     import uuid
        #     # Generate random filename
        #     filename = f"results_multi_one/{uuid.uuid4().hex}.npy"

        #     # Save to file
        #     np.save(filename, np.array(arrays, dtype=object), allow_pickle=True)           

        elif mode == "diversity":
            fully_explored_map_cuda = torch.as_tensor(self.fully_explored_map, device=self.map_device)

            valid_mask = fully_explored_map_cuda & (self.feature_map.sum(dim=-1) != 0.0)
            explored_map_features = self.feature_map[valid_mask]  # shape: (N, D)

            explored_norm = explored_map_features / (explored_map_features.norm(dim=-1, keepdim=True) + 1e-8)

            for frontier in frontiers:
                points = frontier.discovery_zone
                if not len(points):
                    frontier.frontier_explore_score = 0
                    continue

                frontier_features = self.feature_map[points[:, 0], points[:, 1]]  # shape: (M, D)

                frontier_norm = frontier_features / (frontier_features.norm(dim=-1, keepdim=True) + 1e-8)

                sim = explored_norm @ frontier_norm.T  # shape: (N, M)
                dissim = 1 - sim.float()

                dissim_per_point, _ = torch.min(dissim, dim=0)  # shape: (M,)
                k = min(75, dissim_per_point.shape[0])
                score = torch.topk(dissim_per_point, k).values.mean().item()

                frontier.frontier_explore_score = score

            for cluster in clusters:
                cluster.cluster_explore_score = 0
               
        if mode not in ["certainty", "greedy"] and (not len(frontiers) or np.all([frontier.frontier_explore_score == 0 for frontier in frontiers])):
            mode = fallback
            if self.config.log_rerun:
                rr.log("path_updates", rr.TextLog(f"Fallback activated ({mode}). {len(frontiers)} frontiers."))

        if mode == "certainty":
            for frontier in frontiers:
                points = frontier.discovery_zone
                if not len(points):
                    frontier.frontier_explore_score = 0
                else:
                    frontier.frontier_explore_score = 1 / (np.median(self.confidence_map[points[:, 0], points[:, 1]]) + 1e-7)
            for cluster in clusters:
                cluster.cluster_explore_score = 1 / (np.median(self.confidence_map[cluster.points[:, 0], cluster.points[:, 1]]) + 1e-7)
        
        elif mode == "greedy":
            for frontier in frontiers:
                frontier.frontier_explore_score = frontier.frontier_score
            for cluster in clusters:
                cluster.cluster_explore_score = cluster.cluster_score

    def _cluster_reachable(self, cluster:Cluster, largest_contour, req_dist=-15):
        in_fully_explored = self.fully_explored_map[tuple(cluster.center)]
        close_to_reach = largest_contour is None or cv2.pointPolygonTest(largest_contour, cluster.center.astype(float), measureDist=True) > req_dist
        return close_to_reach or in_fully_explored

    def compute_frontiers_and_POIs(self, mode, fallback_mode, allow_retry=True):
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
            if len(self.blacklisted_nav_goals) == 0 or not np.any(np.all(cluster.get_descr_point() == self.blacklisted_nav_goals, axis=1)):
                if self._cluster_reachable(cluster, largest_contour) and not self.checked_map[tuple(cluster.center)]:
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
                frontier = Frontier(frontier_midpoint=frontier_mp, points=frontier_points, frontier_score=score, frontier_explore_score=0, discovery_zone=[])
                nav_goals.append(frontier)

        self.set_exploration_scores(nav_goals, mode=mode, fallback=fallback_mode)

        if self.config.log_rerun:
            if len(nav_goals) > 0:
                pts = np.array([nav_goal.get_descr_point() for nav_goal in nav_goals])
                scores = np.array([nav_goal.get_score() for nav_goal in nav_goals])
                rr.log(
                    "map/frontiers_and_POIs",
                    rr.Points2D(rotate_frame(pts), colors=np.flip(monochannel_to_inferno_rgb(scores), axis=-1), radii=[1] * pts.shape[0])
                )               

        if len(nav_goals) == 0:
            if not self.initializing and allow_retry:
                self.reset_checked_map()
                return self.compute_frontiers_and_POIs(mode, fallback_mode, allow_retry=False)
            return []

        self.initializing = False
        return sorted(nav_goals, key=lambda x: x.get_score(), reverse=True)

    def _find_closest_agent(self, start_pos:List[np.ndarray], mappers: List["Navigator"], nav_goals: List[NavGoal], prev_roles:List[str]) -> Tuple[int, list[np.ndarray], NavGoal]:
        # If no agent sees the object, agent closest to NavGoal of max similarity becomes exploiter
        paths = [None for _ in range(len(mappers))]
        distances = np.ones(len(mappers))*np.inf
        
        while np.all(distances == np.inf) and len(nav_goals) > 0:
            agent_best_nav_goals = np.array([list(mapper._get_current_nav_goal(pose, nav_goals)) for pose, mapper in zip(start_pos, mappers)])
            best_nav_goal_id = np.min(agent_best_nav_goals[:,0])
            best_nav_goal = nav_goals[best_nav_goal_id]
            min_goal_dist = 2 if isinstance(best_nav_goal, Frontier) else 4

            for m, mapper in enumerate(mappers):
                if agent_best_nav_goals[m, 0] == best_nav_goal_id:
                    paths[m] = Planning.compute_to_goal(
                        start_pos[m], 
                        self.navigable_map & (self.confidence_map > 0).cpu().numpy(),
                        (self.confidence_map > 0).cpu().numpy(),
                        best_nav_goal.get_descr_point(),
                        mapper.obstcl_kernel_size, 
                        min_goal_dist
                    )
                    distances[m] = np.linalg.norm(np.array(paths[m])[1:]-np.array(paths[m])[:-1], axis=1).sum() if paths[m] is not None else np.inf

            # remove the nav goal. It is either unreachable by all agents, or will be assigned to the closest agent.
            nav_goals.pop(best_nav_goal_id)
        
        if np.all(distances == np.inf):
            return -1, None, None
        
        # keep consistency in case of equality
        candidate_idx = np.argwhere(distances==np.min(distances)).flatten()
        if len(candidate_idx) > 1 and np.any(prev_roles[candidate_idx] == "exploiter"):
            best_idx =  np.argwhere(prev_roles[candidate_idx]== "exploiter").flatten()[0]
        else: 
            best_idx = candidate_idx[0]

        return best_idx, paths[best_idx], best_nav_goal 

    def assign_roles(self, start_pos:List[np.ndarray], mappers: List["Navigator"], nav_goals: List[NavGoal], unavailable_agents: List[bool]) -> np.ndarray:
        prev_roles = np.array([mapper.role for mapper in mappers])
        for mapper in mappers:
            if mapper.object_detected:
                mapper.role = "exploiter"
            else:
                mapper.role = "explorer"

        # One or more agents sees the object, other ones should explore
        roles = np.array([mapper.role for mapper in mappers], dtype='<U10')
        if np.any(roles == "exploiter"):
            return roles == "exploiter"
        
        # If no agent sees the object, agent closest to NavGoal of max similarity becomes exploiter
        closest_agent_idx, path, best_nav_goal = self._find_closest_agent(start_pos, mappers, nav_goals, prev_roles)

        if closest_agent_idx == -1:
            if self.config.log_rerun:
                rr.log("path_updates", rr.TextLog(f"Resetting checked map as no path found."))
            self.reset_checked_map()
            return roles == "exploiter"

        closest_agent = mappers[closest_agent_idx]
        closest_agent.role = "exploiter"
        roles[closest_agent_idx] = "exploiter"

        # if agent does not have a target, current path becomes new target.
        if not unavailable_agents[closest_agent_idx]:
            closest_agent.path = path
            closest_agent._free_unattainable_goal_agent(start_pos[closest_agent_idx], best_nav_goal)

            if self.config.log_rerun:
                rr.log("path_updates", rr.TextLog(f"Agent {closest_agent.agent_id} (exploiter - from assign_roles): computed path of length {len(closest_agent.path)}"))
                rr.log(
                    f"map/agent_{closest_agent.agent_id}/frontiers_dispatch",
                    rr.Points2D(rotate_frame([best_nav_goal.get_descr_point()]), colors=closest_agent.agent_color,radii=[1])
                )

        return roles == "exploiter"

    def split_frontiers_and_POIs(self, unavailable_agents: np.ndarray, nav_goals: List[NavGoal])->Dict[int, List[NavGoal]]:
        agent_coords = self.agents_poses[..., :-1]
        objective_coords = np.array([nav_goal.get_descr_point() for nav_goal in nav_goals])
        n_agents = agent_coords.shape[0]
        n_obj = objective_coords.shape[0]
        if n_obj == 0:
            return {a_id:[] for a_id in range(n_agents)}

        # --- Step 0: filter active agents ---
        active_mask = ~unavailable_agents
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

        assignments = {k: sorted(v, key=lambda x: x.get_explore_score(), reverse=True) for k, v in assignments.items()}
        return assignments

if __name__ == "__main__":
    pass
