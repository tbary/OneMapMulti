"""
This script is used to test the habitat-sim library together with OneMap
"""
import time

# habitat
import habitat_sim
from habitat_sim.utils import common as utils

# numpy
import numpy as np

# rerun
import rerun as rr
import rerun.blueprint as rrb
from habitat_sim import ActionSpec, ActuationSpec
from numpy.lib.function_base import angle

# keyboard input
# from pynput import keyboard

# scipy
from scipy.spatial.transform import Rotation as R

# MON
from mapping import Navigator
from vision_models.clip_dense import ClipModel
from vision_models.yolo_world_detector import YOLOWorldDetector

# from onemap_utils import log_map_rerun
from planning import Planning, Controllers
from config import *
from mapping import rerun_logger, OneMap

# Global variables
running = True

def _normalize(x):
    return (x-x.min())/(x.max()-x.min())

def _pitch_yaw_roll(state):
    orientation = state.rotation
    q0 = orientation.x
    q1 = orientation.y
    q2 = orientation.z
    q3 = orientation.w

    r = R.from_quat([q0, q1, q2, q3])
    # r to euler
    return r.as_euler("yxz")

def _reset_agent_pos(sim: habitat_sim.Simulator, agent_id: int):
    agent = sim.get_agent(agent_id)

    state = agent.get_state()
    state.position = sim.pathfinder.get_random_navigable_point()

    agent.set_state(state)

def transformation_matrix(state):
    pitch = _pitch_yaw_roll(state)[0]
    # pitch is actually around z
    r = R.from_euler("xyz", [0, 0, pitch])
    r = r.as_matrix()
    transformation_matrix = np.hstack((r, pos))
    transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))

    return transformation_matrix

def configure_habitat(hm3d_path, hfov=90, res_x=640, res_y=640, n_agents=1):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = hm3d_path + "/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb"
    backend_cfg.scene_dataset_config_file = hm3d_path + "/hm3d_annotated_basis.scene_dataset_config.json"

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.hfov = hfov
    rgb.position = np.array([0, 0.88, 0])
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [res_y, res_x]

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.hfov = hfov
    depth.position = np.array([0, 0.88, 0])
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [res_y, res_x]

    focal_length = (res_x / 2) / np.tan(np.deg2rad(hfov) / 2)
    K = np.array([
        [focal_length, 0, res_x / 2],
        [0, focal_length, res_y / 2],
        [0, 0, 1]
    ])

    agent_cfgs = []
    for i in range(n_agents):
        agent_cfgs.append(habitat_sim.agent.AgentConfiguration(action_space=dict(
            move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
            turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
        )))
        agent_cfgs[i].sensor_specifications = [rgb, depth]


    sim_cfg = habitat_sim.Configuration(backend_cfg, agent_cfgs)
    sim = habitat_sim.Simulator(sim_cfg)
    objects = sim.semantic_scene.objects
    categories = [ob.category.name() for ob in objects]
    scene_categories = sim.semantic_scene.categories
    scene_categories = [cat.name() for cat in scene_categories]

    for cat in set(categories) - set(scene_categories):
        print("Object category not in scene categories:", cat)

    for cat in set(scene_categories) - set(categories):
        print("Scene category not in object categories:", cat)

    print("Unique categories:", len(set(categories)))

    if n_agents > 1:
        _reset_agent_pos(sim, 1)

    return sim, K

if __name__ == "__main__":
    config = load_config().Conf
    agents_ids = list(range(config.n_agents))
    qs = ["A fridge", "A TV", "A toilet", "A Couch", "A bed"] 

    if type(config.controller) == HabitatControllerConf:
        pass
    else:
        raise NotImplementedError("Spot controller not suited for habitat sim")

    model = ClipModel("weights/clip.pth")
    detector = YOLOWorldDetector(0.3)

    onemap = OneMap(model.feature_dim, config.mapping, map_device="cpu")
    logger = rerun_logger.RerunLogger(onemap, False, "", config.n_agents, debug=False) if config.log_rerun else None

    sim, K = configure_habitat("datasets/scene_datasets/hm3d", hfov=90, res_x=640, res_y=640, n_agents=config.n_agents)
    controller = Controllers.HabitatController(sim, config.controller)

    mappers = []
    for a in agents_ids:
        mapper = Navigator(model, detector, onemap, config, a)
        mapper.set_query(["A Couch"])
        mapper.set_camera_matrix(K)
        mappers.append(mapper)

    initial_sequences = [["turn_left"] * 56 for _ in agents_ids]

    running = True
    obj_found = False
    
    while running:
        if len(initial_sequences[0]):
            actions = [init_seq.pop(0) for init_seq in initial_sequences]
            observations = sim.step(dict(zip(agents_ids, actions)))

            if not len(initial_sequences[0]):
                [mapper.one_map.reset_checked_map() for mapper in mappers]

        else:
            for a, mapper in enumerate(mappers):
                state = sim.get_agent(a).get_state()

                current_pos = np.array([[-state.position[2]], [-state.position[0]], [state.position[1]]])
                path = mapper.get_path()

                if path and len(path) > 1:
                    path = Planning.simplify_path(np.array(path))
                    path = path.astype(np.float32)
                    for i in range(path.shape[0]):
                        path[i, :] = mapper.one_map.px_to_metric(path[i, 0], path[i, 1])
                    # pitch is actually around z
                    # orientation is pitch!
                    controller.control(a, current_pos, _pitch_yaw_roll(state)[0], path)
            observations = sim.get_sensor_observations(agent_ids=agents_ids)

        for a, mapper in enumerate(mappers):
            state = sim.get_agent(a).get_state()
            pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))

            t = time.time()
            obj_found = obj_found or mapper.add_data(observations[a]["rgb"][..., :-1].transpose(2, 0, 1), observations[a]["depth"].astype(np.float32), transformation_matrix(state))
            print("Time taken to add data: ", time.time() - t)
            mapper.update_map()

            if logger:
                rr.log(f"agent_{a}/camera/rgb", rr.Image(observations[a]["rgb"]))
                rr.log(f"agent_{a}/camera/depth", rr.Image(_normalize(observations[a]["depth"])))
                rr.log(f"agent_{a}/camera/target", rr.Points2D(positions=[[125, 10]], labels=[f"Target: {mapper.query_text[0]}"], colors=[[255,255,255]]))
                logger.log_map()
                logger.log_pos(pos[0, 0], pos[1, 0], a)

        if obj_found:
            obj_found = False
            if len(qs):
                new_query = qs.pop(0)
                for mapper in mappers:
                    mapper.set_query([new_query])
            else:
                running = False
