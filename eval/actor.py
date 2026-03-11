# abstract classes
from abc import ABC, abstractmethod

from typing import Dict, Tuple

import numpy as np
from collections import defaultdict

# MON
from vision_models.clip_dense import ClipModel
from vision_models.yolo_world_detector import YOLOWorldDetector
# from vision_models.grounding_dino_detector import GroundingDinoDetector
# from vision_models.yolov8_model import YoloV8Detector
# from vision_models.point_nav_policy import WrappedPointNavResNetPolicy
# from vision_models.yolov6_model import YOLOV6Detector
from vision_models.yolov7_model import YOLOv7Detector

from mapping import Navigator, OneMap
from planning import Planning, Controllers
# scipy
from scipy.spatial.transform import Rotation as R

import rerun as rr

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

def _transformation_matrix(pos, orientation):
    q0 = orientation.x
    q1 = orientation.y
    q2 = orientation.z
    q3 = orientation.w

    r = R.from_quat([q0, q1, q2, q3])
    # r to euler
    yaw, *_ = r.as_euler("yxz")
    # pitch is actually around z
    r = R.from_euler("xyz", [0, 0, yaw])
    r = r.as_matrix()
    transformation_matrix = np.hstack((r, pos))
    transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))

    return yaw, transformation_matrix

def rotate_frame(points):
    return [[y, x] for (x, y) in points]

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
    def __init__(self, config):
        model = ClipModel("weights/clip.pth", jetson=False)
        detector = YOLOWorldDetector(config.planner.yolo_confidence) if config.planner.using_ov \
            else YOLOv7Detector(config.planner.yolo_confidence)

        self.n_agents = config.n_agents
        self.policy = None
        self.action_lookup = [None, 'move_forward', 'turn_left', 'turn_right']

        self.init = 36*2 * config.n_agents

        self.one_map = OneMap(model.feature_dim, config.mapping, map_device="cpu")
        self.mappers = [Navigator(model, detector, self.one_map, config, agent_id) for agent_id in range(config.n_agents)]
        
        K = _gen_camera_matrix(90 if config.square_im else 97, 640, 640 if config.square_im else 480)
        for mapper in self.mappers:
            mapper.set_camera_matrix(K)
        self.controller = Controllers.HabitatController(None, config.controller)

    # consider that obs has the obs for all the agents
    def act(self, observations: Dict[str, any]) -> Tuple[Dict, bool]:
        return_act = defaultdict(lambda:defaultdict(dict))
        obj_found = False
        for a, mapper in enumerate(self.mappers):
            state = observations[a]["state"]

            pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))
            yaw, transformation_matrix = _transformation_matrix(pos, state.rotation)
            
            obj_found = obj_found or mapper.add_data(
                observations[a]["rgb"][:, :, :-1].transpose(2, 0, 1),
                observations[a]["depth"].astype(np.float32),
                transformation_matrix
            )
            if self.init > 0:
                return_act['discrete'][a] = 'turn_left'
                self.init -= 1
            else:
                path = mapper.get_path()
                if isinstance(path, str):
                    if path == "L":
                        return_act['discrete'][a] = 'turn_left'
                        continue
                    elif path == "R":
                        return_act['discrete'][a] = 'turn_right'
                        continue

                if path and len(path) > 0:
                    if self.policy is not None:
                        goal_pt = self.one_map.px_to_metric(path[-1][0], path[-1][1])
                        action = self.policy.act(observations[a]['depth'], pos[:2, 0], yaw, goal_pt)
                        # action 2 means turn left, action 3 means turn right action 1 means move forward?
                        return_act['discrete'][a] = self.action_lookup[action.item()]
                    else:
                        path = Planning.simplify_path(np.array(path))
                        path = np.array(path).astype(np.float32)
                        rr.log(f"map/agent_{a}/path_simplified",  rr.LineStrips2D(
                            rotate_frame(path), 
                            colors=np.repeat(np.array([0, 0, 255])[np.newaxis, :],
                            path.shape[0], axis=0)
                        ))
                        for i in range(path.shape[0]):
                            path[i, :] = self.one_map.px_to_metric(path[i, 0], path[i, 1])
                        ang, lin = self.controller.control(a, pos, yaw, path, False)
                        return_act['continuous']['linear'][a] = lin
                        return_act['continuous']['angular'][a] = ang
                else:
                    return_act['discrete'][a] = 'move_forward'

        return return_act, obj_found

    def reset(self):
        for mapper in self.mappers:
            mapper.reset()
        self.init = 36*2 * self.n_agents

    def set_query(self, query: str):
        for mapper in self.mappers:
            mapper.set_query([query])
