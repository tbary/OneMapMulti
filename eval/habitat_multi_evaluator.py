# eval utils
from eval import get_closest_dist
from eval.actor import MONActor
from mapping import rerun_logger, rotate_frame
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *

import matplotlib.pyplot as plt
import tqdm

# os / filsystem
import os

# cv2
import cv2

# numpy
import numpy as np

# typing
from typing import Dict

# habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

# rerun
import rerun as rr

# scipy
from scipy.spatial.transform import Rotation as R

def _get_pose(state):
    pose = np.zeros((4,))
    pose[0] = -state.position[2]
    pose[1] = -state.position[0]
    pose[2] = state.position[1]
    # yaw
    orientation = state.rotation
    q0 = orientation.x
    q1 = orientation.y
    q2 = orientation.z
    q3 = orientation.w
    r = R.from_quat([q0, q1, q2, q3])
    # r to euler
    yaw, *_ = r.as_euler("yxz")
    pose[3] = yaw
    return pose

def _normalize(x:np.ndarray):
    return (x-x.min())/(x.max()-x.min())

def _is_stuck(poses, agents_ids, threshold = 0.05):
    return np.max([np.linalg.norm(poses[agent_id][-1][:2] - poses[agent_id][-10][:2]) for agent_id in agents_ids]) < threshold

class Metrics:
    def __init__(self, ep_id) -> None:
        self.ep_id = ep_id
        self.sequence_results:list[Result] = []
        self.sequence_agent_finding:list[int] = []
        self.sequence_poses:list[list[np.ndarray]] = []
        self.sequence_object:list[str] = []

    def add_sequence(self, sequences: list, result: Result, agent_id:int, target_object: str) -> None:
        start_id = 0
        if len(self.sequence_poses) > 0:
            start_id = sum([len(seq[0]) for seq in self.sequence_poses])
        seq_poses = [np.array(seq)[start_id:, :] for seq in sequences]
        self.sequence_poses.append(seq_poses)
        self.sequence_results.append(result)
        self.sequence_agent_finding.append(agent_id)
        self.sequence_object.append(target_object)

    def get_progress(self):
        return self.sequence_results.count(Result.SUCCESS) /SEQ_LEN

class HabitatMultiEvaluator:
    def __init__(self,
                 config: EvalConf,
                 actor: MONActor,
                 ) -> None:
        self.config = config
        self.multi_object = config.multi_object
        self.controller = config.controller
        self.mapping = config.mapping
        self.planner = config.planner
        self.log_rerun = config.log_rerun
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.scene_data = {}
        self.episodes = []
        self.is_gibson = config.is_gibson
        self.n_agents = config.n_agents

        self.sim: habitat_sim.Simulator = None
        self.actor = actor
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.control_frequency = config.controller.control_freq
        self.max_vel = config.controller.max_vel
        self.max_ang_vel = config.controller.max_ang_vel
        self.time_step = 1.0 / self.control_frequency
        self.square = config.square_im

        if self.multi_object:
            self.episodes, self.scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes,
                                                                                       self.scene_data,
                                                                                       self.object_nav_path)
        else:
            raise RuntimeError("You are running the multi object evaluation with a single object config.")
        if self.actor is not None:
            self.logger = rerun_logger.RerunLogger(self.actor.one_map, False, "",  self.n_agents) if self.log_rerun else None
        self.results_path = "/home/finn/active/MON/results_gibson_multi" if self.is_gibson else self.config.results_path

    def load_scene(self, scene_id: str):
        if self.sim is not None:
            self.sim.close()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path + scene_id

        backend_cfg.scene_dataset_config_file = self.scene_path + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        hfov = 90 if self.square else 79
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.hfov = hfov
        rgb.position = np.array([0, 0.88, 0])
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        res_x = 640
        res_y = 640 if self.square else 480
        rgb.resolution = [res_y, res_x]

        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.hfov = hfov
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.position = np.array([0, 0.88, 0])
        depth.resolution = [res_y, res_x]

        agents_cfgs:list[habitat_sim.agent.AgentConfiguration] = []
        for i in range(self.n_agents):
            agents_cfgs.append(habitat_sim.agent.AgentConfiguration(action_space=dict(
                move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
                turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
                turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
            )))
            agents_cfgs[i].sensor_specifications = [rgb, depth]
        sim_cfg = habitat_sim.Configuration(backend_cfg, agents_cfgs)
        self.sim = habitat_sim.Simulator(sim_cfg)
        if self.scene_data[scene_id].objects_loaded:
            return
        self.scene_data = HM3DDataset.load_hm3d_objects(self.scene_data, self.sim.semantic_scene.objects, scene_id)

    def execute_action(self, action: Dict):
        if 'discrete' in action.keys():
            # We have a discrete actor
            self.sim.step(action['discrete'])
            if self.log_rerun:
                rr.log("actions_updates",rr.TextLog(f"Discrete actions: {action['discrete']}."))

        if 'continuous' in action.keys():
            agents_ids = list(action['continuous'].keys())
            for agent_id in agents_ids:
                # We have a continuous actor
                self.vel_control.angular_velocity = action['continuous'][agent_id]['angular']
                self.vel_control.linear_velocity = action['continuous'][agent_id]['linear']
                agent_state = self.sim.get_agent(agent_id).state
                previous_rigid_state = habitat_sim.RigidState(
                    utils.quat_to_magnum(agent_state.rotation), agent_state.position
                )

                # manually integrate the rigid state
                target_rigid_state = self.vel_control.integrate_transform(
                    self.time_step, previous_rigid_state
                )

                # snap rigid state to navmesh and set state to object/sim
                # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
                end_pos = self.sim.step_filter(
                    previous_rigid_state.translation, target_rigid_state.translation
                )

                # set the computed state
                agent_state.position = end_pos
                agent_state.rotation = utils.quat_from_magnum(
                    target_rigid_state.rotation
                )
                self.sim.get_agent(agent_id).set_state(agent_state)

                if self.log_rerun:
                    rr.log("actions_updates",rr.TextLog(f"Agent {agent_id} action command: {self.vel_control.angular_velocity} (angular), {self.vel_control.linear_velocity} (linear)."))
            self.sim.step_physics(self.time_step)

    def save_final_sims(self, episode_id, sequence_id, poses):
        final_sim = (self.actor.one_map.similarity_map + 1.0) / 2.0
        final_sim = monochannel_to_inferno_rgb(final_sim)

        confs = (self.actor.one_map.confidence_map > 0).cpu().squeeze().numpy()
        
        final_sim[~confs, :] = [0, 0, 0]
        min_x = np.min(np.where(confs)[0])
        max_x = np.max(np.where(confs)[0])
        min_y = np.min(np.where(confs)[1])
        max_y = np.max(np.where(confs)[1])
        final_sim = final_sim[min_x:max_x, min_y:max_y]
        final_sim = final_sim.transpose((1, 0, 2))
        final_sim = np.flip(final_sim, axis=0)                        # get min and max x and y of confs


        cv2.imwrite(f"{self.results_path}/similarities/final_sim_{episode_id}_{sequence_id}.png", final_sim)

        for agent_id in range(self.n_agents):
            # Create the plot
            fig = plt.figure(figsize=(10, 10))
            poses_ = np.array([self._metric_to_px(*pos[:2]) for pos in poses[agent_id]])
            poses_[:, 0] -= min_x
            poses_[:, 1] -= min_y
            plt.imshow(final_sim[:, :, ::-1], interpolation='nearest', aspect='equal',
                    extent=(0, final_sim.shape[1], 0, final_sim.shape[0]))

            plt.plot(poses_[:, 0], poses_[:, 1], 'b-o')  # 'b-o' means blue line with circle markers

            # Set equal aspect ratio to ensure accurate positions
            plt.axis('equal')

            # Add labels and title
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.title(f'Path of Poses for agent {agent_id}')

            # Add grid for better readability
            plt.grid(True)

            # Save the plot as SVG
            plt.savefig(f"{self.results_path}/similarities/path_{episode_id}_{sequence_id}_{agent_id}.svg", format='svg', dpi=300, bbox_inches='tight')

            # Display the plot (optional, comment out if not needed)
            # plt.show()
            plt.close(fig)

    def _metric_to_px(self, x, y):
        return self.actor.projection.metric_to_px(x,y)

    def _px_to_metric(self, px, py):
        return self.actor.projection.px_to_metric(px,py)

    def _log_ground_truth(self, episode:Episode, current_obj):
        pts = []
        for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
            if not self.is_gibson:
                pt = obj.bbox.center[[0, 2]]
                pt = (-pt[1], -pt[0])
                pts.append(self._metric_to_px(*pt))
            else:
                for pt_ in obj:
                    pt = (pt_[0], pt_[1])
                    pts.append(self._metric_to_px(*pt))
        pts = np.array(pts)
        if self.log_rerun:
            rr.log("map/ground_truth", rr.Points2D(rotate_frame(pts), colors=[[255, 255, 255]], radii=[1]))

    def evaluate(self, from_scratch=True):
        results:list[Metrics] = []
        agents_ids = list(range(self.n_agents))

        starting_point = 214 if from_scratch else len(os.listdir(os.path.join(self.results_path, "state")))

        for n_ep, episode in enumerate(self.episodes[starting_point:]):
            poses = [[] for _ in agents_ids]
            results.append(Metrics(episode.episode_id))

            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)
            
            for agent_id in agents_ids:
                self.sim.initialize_agent(agent_id, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
            self.actor.reset()

            pbar = tqdm.tqdm(total=None)
            
            sequence_id = 0
            failed = False
            while not failed and sequence_id < len(episode.obj_sequence):
                current_obj = episode.obj_sequence[sequence_id]
                self.actor.set_query(current_obj)

                if self.log_rerun:
                    self._log_ground_truth(episode, current_obj)

                steps = 0
                agent_called_found = -1
                while steps < self.config.max_steps and agent_called_found == -1:
                    observations = self.sim.get_sensor_observations(agent_ids=agents_ids)
                    for agent_id in agents_ids:
                        observations[agent_id]['state'] = self.sim.get_agent(agent_id=agent_id).get_state()
                        poses[agent_id].append(_get_pose(observations[agent_id]['state']))

                        if self.log_rerun:
                            cam_x = -self.sim.get_agent(agent_id).get_state().position[2]
                            cam_y = -self.sim.get_agent(agent_id).get_state().position[0]
                            rr.log(f"agent_{agent_id}/camera/rgb", rr.Image(observations[agent_id]["rgb"]))
                            rr.log(f"agent_{agent_id}/camera/depth", rr.Image(_normalize(observations[agent_id]["depth"])))
                            rr.log(f"agent_{agent_id}/camera/target", rr.Points2D(positions=[[125, 10]], labels=[f"Target: {current_obj}"], colors=[[255,255,255]]))
                            self.logger.log_pos(self.actor.mappers[agent_id].projection, cam_x, cam_y, agent_id)
                    actions, agent_called_found, nav_goals = self.actor.act(observations)
                    self.execute_action(actions)

                    if self.log_rerun:
                        self.logger.log_map()

                    if steps % 100 == 0:
                        dists = [get_closest_dist(
                            self.sim.get_agent(agent_id).get_state().position[[0, 2]],
                            self.scene_data[episode.scene_id].object_locations[current_obj],
                            self.is_gibson
                        ) for agent_id in agents_ids]
                        pbar.desc = f"Step {steps}, current object: {current_obj}, episode_id: {episode.episode_id + 1}/{len(self.episodes)}, distance to closest object: {np.min(dists)}"
                    steps += 1
                    pbar.update(1)

                if agent_called_found != -1:
                    dists = [get_closest_dist(self.sim.get_agent(agent_id).get_state().position[[0, 2]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj],
                                            self.is_gibson) for agent_id in agents_ids]
                    if np.min(dists) < self.config.max_dist:
                        result = Result.SUCCESS
                        pbar.write(f"Object {current_obj} found!")
                    else:
                        failed = True
                        dists_detect = []
                        for mapper in self.actor.mappers:
                            pos = mapper.chosen_detection
                            if pos is not None:
                                pos_metric = self._px_to_metric(pos[0], pos[1])
                                dists_detect.append(get_closest_dist(
                                    [-pos_metric[1], -pos_metric[0]],
                                    self.scene_data[episode.scene_id].object_locations[current_obj],
                                    self.is_gibson
                                ))
                        if np.min(dists_detect) < self.config.max_dist:
                            result = Result.FAILURE_NOT_REACHED
                        else:
                            result = Result.FAILURE_MISDETECT
                        pbar.write(f"Object {current_obj} not found! Dist {np.min(dists)}, detect dist: {np.min(dists_detect)}.")
                    
                else:
                    failed = True

                    if _is_stuck(poses, agents_ids):
                        result = Result.FAILURE_STUCK
                    elif len(nav_goals) == 0:
                        result = Result.FAILURE_ALL_EXPLORED
                    else:
                        result = Result.FAILURE_OOT
                    
                    pbar.write(f"Out of time to find object {current_obj}!")

                results[-1].add_sequence(poses, result, agent_called_found, current_obj)

                self.save_final_sims(episode.episode_id, sequence_id, poses)

                sequence_id += 1

            for seq_id, seq in enumerate(results[n_ep].sequence_poses):
                for agent_id, agent_seq in enumerate(seq):
                    np.savetxt(f"{self.results_path}/trajectories/poses_{episode.episode_id}_{seq_id}_{agent_id}.csv", agent_seq, delimiter=",")

            pbar.write(f"Average progress: {sum([m.get_progress() for m in results]) / (len(results))}")
            pbar.close()

            with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
                f.write(','.join(str(results[n_ep].sequence_results[i].value) for i in range(len(results[n_ep].sequence_results))))
                f.write('\n')
                f.write(','.join(str(results[n_ep].sequence_agent_finding[i]) for i in range(len(results[n_ep].sequence_results))))
