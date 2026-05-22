# abstract classes
from abc import ABC, abstractmethod

from typing import Dict, Tuple, List

import numpy as np

from config import EvalConf

# MON
from vision_models.clip_dense import ClipModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from vision_models.yolov7_model import YOLOv7Detector


from .mapping import Navigator, OneMap, Projection
from .planning import Planning
# scipy
from scipy.spatial.transform import Rotation as R

def _gen_camera_matrix(hfov, res_x, res_y):
    hfov = np.deg2rad(hfov)
    focal_length = (res_x / 2) / np.tan(hfov / 2)
    principal_point_x = res_x / 2
    principal_point_y = res_y / 2
    return np.array([
        [focal_length, 0, principal_point_x],
        [0, focal_length, principal_point_y],
        [0, 0, 1]
    ])

class Actor(ABC):
    @abstractmethod
    def act(self,
            observations: Dict[str, any]) -> Tuple[Dict, bool]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_query(self, query: str):
        pass

class MONActor(Actor):
    one_map: OneMap
    mappers: list[Navigator]
    def __init__(self, config:EvalConf):
        model = ClipModel("weights/clip.pth", jetson=False)
        detector = YOLOWorldDetector(config.planner.yolo_confidence) if config.planner.using_ov \
            else YOLOv7Detector(config.planner.yolo_confidence)

        self.mode = config.mode
        self.fallback_mode = config.fallback_mode

        self.n_agents = config.n_agents
        self.init = 36*2 * config.n_agents

        self.one_map = OneMap(config.sort_seed, config.n_agents, model.feature_dim, config.mapping, map_device="cpu")
        self.projection = Projection(model.feature_dim, config.mapping)
        self.mappers = [Navigator(model, detector, self.one_map, self.projection, config, agent_id) for agent_id in range(config.n_agents)]
        
        K = _gen_camera_matrix(90 if config.square_im else 97, 640, 640 if config.square_im else 480)
        for mapper in self.mappers:
            mapper.projection.set_camera_matrix(K)

    def act(self, images:List[np.ndarray], depths:List[np.ndarray], odometries: List[np.ndarray]) -> Tuple[Dict, bool]:
        current_poses = [np.array(self.projection.metric_to_px(odom[0, 3], odom[1, 3]), dtype=int) for odom in odometries]
        yaws = [np.arctan2(odom[1, 0], odom[0, 0]) for odom in odometries]

        # Observe
        for a, mapper in enumerate(self.mappers):
            mapper.one_map.update_agent_pose((*current_poses[a],yaws[a]), mapper.agent_id)
            mapper.add_data(images[a], depths[a], odometries[a])
            mapper.check_object_in_image(images[a], depths[a], odometries[a])

        nav_goals = self.one_map.compute_frontiers_and_POIs(self.mode, self.fallback_mode)
        following_previous = [self.mappers[a].try_previous_frontier(current_poses[a], nav_goals) for a in range(self.n_agents)]
        exploiting_agents = self.one_map.assign_roles(current_poses, self.mappers, nav_goals, following_previous)
        unavailable_agents = np.logical_or(exploiting_agents, following_previous)
        assigned_nav_goals = self.one_map.split_frontiers_and_POIs(unavailable_agents, nav_goals)

        # Plan
        obj_found = -1
        for a, mapper in enumerate(self.mappers):

            if self.init == 0:
                if mapper.object_detected:
                    obj_found = max(mapper.check_object_reached(current_poses[a]), obj_found) #TODO, could be more technically correct

                elif mapper.config.planner.allow_replan and not unavailable_agents[a] and len(assigned_nav_goals[a]):
                    mapper.compute_best_path_to_frontier(current_poses[a], assigned_nav_goals[a])

            mapper.last_pose = (*current_poses[a], yaws[a])

        # Act on plans
        paths = []
        for a, mapper in enumerate(self.mappers):
            if self.init > 0:
                path = np.array([])
                self.init -= 1
            else:
                path = mapper.path

                if path and len(path) > 0:
                    path = Planning.simplify_path(np.array(path))
                    path = np.array(path).astype(np.float32)
                    for i in range(path.shape[0]):
                        path[i, :] = self.projection.px_to_metric(path[i, 0], path[i, 1])

                else:
                    path = np.array([])
            paths.append(path)
        
        if obj_found != -1:
            for mapper in self.mappers:
                mapper.last_nav_goal = None

        return paths, obj_found

    def reset(self):
        for mapper in self.mappers:
            mapper.reset()
        self.init = 36*2 * self.n_agents

    def set_query(self, query: str):
        for mapper in self.mappers:
            mapper.set_query([query])
