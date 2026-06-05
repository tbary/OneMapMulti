"""
This module contains the Navigator class, which is responsible for the main functionality. It updates Onemap and uses it
for navigation and exploration.
"""
from .feature_map import OneMap
from .nav_goals.frontier import Frontier, detect_frontiers, get_frontier_midpoint
from .nav_goals.clustering import Cluster, cluster_high_similarity_regions
from .nav_goals.navigation_goals import NavGoal
from .projection import Projection

from planning import Planning
from vision_models.base_model import BaseModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from config import Conf
from mobile_sam import sam_model_registry, SamPredictor

# numpy
import numpy as np
import matplotlib.pyplot as plt

# typing
from typing import List, Optional, Set, Any, Union, Tuple

# torch
import torch

# cv2
import cv2

def closest_point_within_threshold(nav_goals: List[NavGoal], target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        nav_goals (List[NavGoal]): An array of potential nav points, where each point is retrieved by nav_goal.get_descr_point
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    points_array = np.array([nav_goal.get_descr_point() for nav_goal in nav_goals])
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1

class HistoricData:
    def __init__(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None):
        self.position = position
        self.frontier_pt = frontier_pt
        self.other = other

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.frontier_pt}_{self.other}"
        return hash(string_repr)


class CyclicChecker:
    history: Set[HistoricData] = set()

    def check_cyclic(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> bool:
        state_action = HistoricData(position, frontier_pt, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> None:
        state_action = HistoricData(position, frontier_pt, other)
        self.history.add(state_action)


class Navigator:
    query_text: List[str]  # the query texts as a list. The first element will be used for planning and frontier score
    # computation
    query_text_features: torch.Tensor
    last_nav_goal: Union[NavGoal, None]

    def __init__(self,
                 model: BaseModel,
                 detector: YOLOWorldDetector,
                 one_map: OneMap,
                 projection: Projection,
                 config: Conf,
                 agent_id: int
                 ) -> None:

        self.cyclic_checker = CyclicChecker()
        self.config = config
        self.agent_id = agent_id
        self.agent_color = [(int(r*255), int(g*255), int(b*255)) for r,g,b in plt.get_cmap('tab10').colors[agent_id:agent_id+1]]
        self.role: str["exploiter", "explorer", "idle"] = "idle"

        # Models
        self.model = model
        self.detector = detector
        self.sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt").to(device="cuda").eval()
        self.sam_predictor = SamPredictor(self.sam)

        self.one_map = one_map
        self.projection = projection

        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to(self.one_map.map_device)

        # Frontier and POIs
        self.artificial_obstacles = set()

        self.last_nav_goal = None
        self.last_pose = None

        self.first_obs = True
        self.object_detected = False
        self.chosen_detection = None
        self.path = None
        self.initializing = True
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0

        self.max_detect_distance = int(config.planner.max_detect_distance / self.one_map.cell_size)
        self.obstcl_kernel_size = int(config.planner.obstcl_kernel_size / self.one_map.cell_size)
        self.min_goal_dist = int(config.planner.min_goal_dist / self.one_map.cell_size)

        # For the closed-vocabulary object detector, not needed for OneMap
        self.class_map = {
            "chair":"chair",
            "tv_monitor":"tv",
            "tv":"tv",
            "plant":"potted plant",
            "potted plant":"potted plant",
            "sofa":"couch",
            "couch":"couch",
            "bed":"bed",
            "toilet":"toilet"
        }

    def reset(self):
        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to(self.one_map.map_device)
        self.object_detected = False
        self.chosen_detection = None
        self.last_nav_goal = None
        self.last_pose = None
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0
        self.path = None
        self.initializing = True
        self.one_map.reset()
        self.first_obs = True
        self.cyclic_checker = CyclicChecker()
        self.artificial_obstacles = set()
        self.role = "idle"

    def set_query(self, txt: List[str]) -> None:
        """
        Sets the query text
        :param txt: List of strings
        :return:
        """
        for t in txt:
            if t in self.class_map:
                txt[txt.index(t)] = self.class_map[t]
        if txt != self.query_text:
            # print(f"Setting query of agent {self.agent_id} to {txt}")
            self.query_text = txt
            self.query_text_features = self.model.get_text_features(["a " + self.query_text[0]]).to(self.one_map.map_device)

            self.one_map.reset_checked_map()
            self.detector.set_classes(self.query_text)
            self.object_detected = False
            self.update_map(reset=True)

    def compute_best_path_to_object(self, start:np.ndarray) -> bool:
        if np.linalg.norm(start - self.chosen_detection) < self.max_detect_distance:
            self.path = [start] * 5
            # We are close to the object, we don't need to move
            return True

        self.path = Planning.compute_to_goal(
            start, 
            self.one_map.navigable_map,
            (self.one_map.confidence_map > 0).cpu().numpy(),
            self.chosen_detection,
            self.obstcl_kernel_size, 
            self.min_goal_dist
        )

        if self.path and len(self.path) > 0:
            return True
        
        else:
            return False

    def __check_previous_frontier(self, nav_goals: List[NavGoal]):
        if self.last_nav_goal is None or not len(nav_goals):
            return None, None
        
        last_point = self.last_nav_goal.get_descr_point()

        # Exact match
        current_index = None
        for i, goal in enumerate(nav_goals):
            if np.array_equal(last_point, goal.get_descr_point()):
                current_index = i
                break

        # Nearby match
        if current_index is None:
            closest_index = closest_point_within_threshold(
                nav_goals,
                last_point,
                0.5 / self.one_map.cell_size
            )
            if closest_index == -1:
                return None, None
            
            current_index = closest_index

        # Check if goal still worth pursuing
        if self.role == "explorer":
            if nav_goals[current_index].get_explore_score() > 0:
                return current_index, nav_goals[current_index]

        else:
            if nav_goals[current_index].get_score() + 0.01 > self.last_nav_goal.get_score():
                return current_index, nav_goals[current_index]

        return None, None

    def try_previous_frontier(self, start:np.ndarray, nav_goals:List[NavGoal])->bool:
        if self.object_detected:
            return False
        
        nav_id, goal = self.__check_previous_frontier(nav_goals)

        if nav_id is None:
            return False

        min_goal_dist = 2 if isinstance(goal, Frontier) else 4
        self.path = Planning.compute_to_goal(
            start, 
            self.one_map.navigable_map & (self.one_map.confidence_map > 0).cpu().numpy(),
            (self.one_map.confidence_map > 0).cpu().numpy(),
            goal.get_descr_point(),
            self.obstcl_kernel_size, 
            min_goal_dist
        )

        if self.path is None:
            return False
        
        self.free_unattainable_goal_agent(start, goal)

        nav_goals.pop(nav_id)
        return True

    def _get_current_nav_goal(self, start, assigned_nav_goals:List[NavGoal])->Tuple[int, Union[Frontier, Cluster]]:
        second_idx = 0 if len(assigned_nav_goals) == 1 else 1
        top_two_vals = tuple((assigned_nav_goals[0].get_score(), assigned_nav_goals[second_idx].get_score()))
               
        # Select the current best nav_goal, and check for cyclic
        for nav_id, goal in enumerate(assigned_nav_goals):
            # Check that the current goal is not in the previous goals
            if not self.cyclic_checker.check_cyclic(start, goal.get_descr_point(), top_two_vals):
                self.cyclic_checker.add_state_action(start, goal.get_descr_point(), top_two_vals)
                return nav_id, goal
            
    def compute_best_path_to_frontier(self, start: np.ndarray, assigned_nav_goals:List[NavGoal]):
        self.path = None
        while self.path is None and len(assigned_nav_goals) > 0:
            best_idx, best_nav_goal = self._get_current_nav_goal(start, assigned_nav_goals)           

            min_goal_dist = 2 if isinstance(best_nav_goal, Frontier) else 4

            self.path = Planning.compute_to_goal(
                start, 
                self.one_map.navigable_map & (self.one_map.confidence_map > 0).cpu().numpy(),
                (self.one_map.confidence_map > 0).cpu().numpy(),
                best_nav_goal.get_descr_point(),
                self.obstcl_kernel_size, 
                min_goal_dist
            )

            if self.path is None:
                # remove the nav goal from the list, we don't know how to reach it
                assigned_nav_goals.pop(best_idx)

        if self.path is None:
            self.one_map.reset_checked_map()

        self.free_unattainable_goal_agent(start, best_nav_goal)

    def free_unattainable_goal_agent(self, start, best_nav_goal:Union[Frontier, Cluster]):
        nav_goal_coords = best_nav_goal.get_descr_point()

        if self.last_nav_goal is not None and not np.array_equal(self.last_nav_goal.get_descr_point(), nav_goal_coords):
            self.stuck_at_nav_goal_counter = 0

        elif self.last_pose is not None and self.path is not None:
            if np.array_equal(self.last_pose[:-1], start) and len(self.path) < 5:
                self.stuck_at_nav_goal_counter += 1

        if self.stuck_at_nav_goal_counter > 10:
            # We probably are trying to reach an unreachable goal, for instance a frontier to the void in habitat
            self.one_map.blacklisted_nav_goals.append(nav_goal_coords)
        self.last_nav_goal = best_nav_goal

    def _free_stuck_agent(self, px, py, yaw):
        if self.last_pose:
            if np.linalg.norm(np.array([px, py, yaw]) - np.array(self.last_pose)) < 0.01:
                if self.path is not None:
                    self.stuck_at_cell_counter += 1
            else:
                self.stuck_at_cell_counter = 0
        if self.stuck_at_cell_counter > 5:
            # we are stuck we need to add an obstacle right in front of us!
            dx = np.cos(yaw)
            dy = np.sin(yaw)

            # Round to nearest integer to get facing direction
            facing_dx = round(dx)
            facing_dy = round(dy)

            # Calculate coordinates of facing cell
            facing_px = px + facing_dx
            facing_py = py + facing_dy
            self.artificial_obstacles.add((facing_px, facing_py))

    def _build_similarity_mask(self, kernel_size=7):
        adjusted_score = self.one_map.similarity_map + 1.0
        similarity_threshold = np.percentile(adjusted_score[self.one_map.confidence_map > 0], self.config.planner.percentile_exploitation)
        similarity_mask = (adjusted_score > similarity_threshold).astype(np.uint8)

        similarity_mask[self.one_map.confidence_map == 0] = 0
        k = np.ones((kernel_size, kernel_size), np.uint8)
        
        return cv2.dilate(similarity_mask, k, iterations=1)

    def _convert_to_map_coordinates(self,depth:np.ndarray, odometry:np.ndarray, masks:np.ndarray)->Tuple[int,int,np.ndarray]:
        # Pixels representing the detected object on the image
        object_coords = np.argwhere(masks[0] & (depth != 0))
        yaw = np.arctan2(odometry[1, 0], odometry[0, 0])

        object_depths = depth[object_coords[:, 0], object_coords[:, 1]]

        y_world = -(object_coords[:, 1] - self.projection.fx_fy_cx_cy[2]) * object_depths / self.projection.fx_fy_cx_cy[0]
        x_world = object_depths
        r = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        x_rot, y_rot = np.dot(r, np.stack((x_world, y_world)))
        x_rot += odometry[0, 3]
        y_rot += odometry[1, 3]

        x_id = (x_rot / self.one_map.cell_size).astype(np.uint32) + self.projection.map_center_cells[0].item()
        y_id = (y_rot / self.one_map.cell_size).astype(np.uint32) + self.projection.map_center_cells[1].item()

        return x_id, y_id, object_depths

    def _consensus_filtering(self, adjusted_score, depths, x_id, y_id):
        similarity_mask = self._build_similarity_mask()

        similarity_mask_projections = similarity_mask[x_id, y_id]
        if not np.any(similarity_mask_projections):
            return False

        mask = similarity_mask_projections
        x_masked = x_id[mask == 1]
        y_masked = y_id[mask == 1]
        depths_masked = depths[mask == 1]
        
        closest_object_point = (x_masked[np.argmin(depths_masked)], y_masked[np.argmin(depths_masked)])

        if self.object_detected and adjusted_score[closest_object_point] < adjusted_score[self.chosen_detection] * 1.1:
            return False
        self.chosen_detection = closest_object_point
        return True

    def check_object_in_image(self, image: np.ndarray, depth: np.ndarray, odometry: np.ndarray)->bool:
        # Case: the queried object was detected in the current image
        detections = self.detector.detect(image.transpose(1, 2, 0))
        if len(detections["boxes"]) > 0:
            # wants rgb
            self.sam_predictor.set_image(image.transpose(1, 2, 0))
            for area, confidence in zip(detections["boxes"], detections['scores']):

                # Obtain the object masks.
                masks, *_ = self.sam_predictor.predict(
                    point_coords=None,                                     
                    point_labels=None,
                    box=np.array(area)[None, :],
                    multimask_output=False, 
                )

                dist_to_obj_center = depth[(int((area[3] + area[1]) // 2), int((area[2] + area[0]) // 2))]
                if not self.config.planner.filter_detections_depth or dist_to_obj_center < 2.5:
                    x_id, y_id, object_depths = self._convert_to_map_coordinates(depth, odometry, masks)

                    adjusted_score = self.one_map.similarity_map + 1.0  # only positive scores

                    # Check whether the detected object is valid
                    if self.config.planner.consensus_filtering:
                        object_valid = self._consensus_filtering(adjusted_score, object_depths, x_id, y_id)

                    else:
                        closest_object_point = (x_id[np.argmin(object_depths)], y_id[np.argmin(object_depths)])
                        if self.object_detected and adjusted_score[closest_object_point] < adjusted_score[self.chosen_detection] * 1.1:
                            object_valid = False
                        else:
                            self.chosen_detection = closest_object_point
                            object_valid = True

                    if object_valid:
                        old_path = self.path.copy() if self.path else self.path
                        px, py = self.projection.metric_to_px(odometry[0, 3], odometry[1, 3])
                        self.object_detected = self.compute_best_path_to_object(np.array([px, py]))
                        if not self.path:
                            self.path = old_path

        else:
            if not self.object_detected:
                self.chosen_detection = None
        return self.object_detected

    def check_object_reached(self, current_pos):
        # We are following an object we detected at previous iterations (or just now)
        if np.linalg.norm(current_pos - self.chosen_detection) <= self.max_detect_distance:
            self.object_detected = False
            return self.agent_id

        if self.config.planner.consensus_filtering:
            similarity_mask = self._build_similarity_mask()

            if not similarity_mask[tuple(self.chosen_detection)]:
                self.object_detected = False
                return -1
    
        if self.config.planner.allow_replan:
            self.object_detected = self.compute_best_path_to_object(current_pos)            

        if self.object_detected and len(self.path) < 3:
            self.object_detected = False
            return self.agent_id

        return -1

    def add_data(self, image: np.ndarray, depth: np.ndarray, odometry: np.ndarray,):
        """
        Adds data to the navigator
        :param image: RGB image of dimension [C, H, W]
        :param depth: depth image of dimension [H, W]
        :param odometry: 4x4 transformation matrix from camera to world
        :return: boolean indicating if the episode is over
        """
        
        yaw = np.arctan2(odometry[1, 0], odometry[0, 0])
        px, py = self.projection.metric_to_px(odometry[0, 3], odometry[1, 3])

        self._free_stuck_agent(px, py, yaw)
        # Update onemap with new information
        image_features = self.model.get_image_features(image[np.newaxis, ...]).squeeze(0)
        self.one_map.update(self.projection, image_features, depth, odometry, self.artificial_obstacles)
        self.update_map()

        # Ensure proper start of the agent. It will not try to look at its feet.
        if self.first_obs:
            self.one_map.confidence_map[px - 10:px + 10, py - 10:py + 10] += 10
            self.one_map.checked_conf_map[px - 10:px + 10, py - 10:py + 10] += 10
            self.first_obs = False

    def update_map(self, reset=False) -> None:
        """updates the similarity map given the query text"""
        if self.query_text_features is None:
            raise ValueError("No query text set")
        
        if reset:
            self.one_map.set_similarity_map(None)

        mask = self.one_map.updated_mask
        if mask.max() == 0:
            return
        
        if self.one_map.similarity_map is not None:
            map_features = self.one_map.feature_map[mask, :].permute(1, 0).unsqueeze(0)
        else:
            map_features = self.one_map.feature_map.permute(2, 0, 1).unsqueeze(0)

        similarity = self.model.compute_similarity(map_features, self.query_text_features)[0].cpu().numpy()
        if self.one_map.similarity_map is None:
            self.one_map.set_similarity_map(similarity)
        else:
            # then, similarity is only updated where the mask is true, otherwise it is the previous similarity
            self.one_map.set_similarity_map(similarity, mask=mask)
        self.one_map.reset_updated_mask()
        
if __name__ == "__main__":
    pass
