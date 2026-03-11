# eval utils
from eval import get_closest_dist, FMMPlanner
from eval.actor import MONActor
from eval.dataset_utils.gibson_dataset import load_gibson_episodes
from mapping import rerun_logger
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns

# os / filsystem
import bz2
import os
from os import listdir
import gzip
import json
import pathlib

# cv2
import cv2

# numpy
import numpy as np

# skimage
import skimage

# dataclasses
from dataclasses import dataclass

# quaternion
import quaternion

# typing
from typing import Tuple, List, Dict
import enum

# habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

# tabulate
from tabulate import tabulate

# rerun
import rerun as rr

# pandas
import pandas as pd

# pickle
import pickle

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

def _normalize(x):
    return (x-x.min())/(x.max()-x.min())

def _is_stuck(poses, agents_ids, threshold = 0.05):
    return np.max([np.linalg.norm(poses[agent_id][-1][:2] - poses[agent_id][-10][:2]) for agent_id in agents_ids]) < threshold

def rotate_frame(points):
    return [[y, x] for (x, y) in points]

SEQ_LEN = 3
class Result(enum.Enum):
    SUCCESS = 1
    FAILURE_MISDETECT = 2
    FAILURE_STUCK = 3
    FAILURE_OOT = 4
    FAILURE_NOT_REACHED = 5
    FAILURE_ALL_EXPLORED = 6


class Metrics:
    def __init__(self, ep_id) -> None:
        self.ep_id = ep_id
        # self.sequence_lengths:list[float] = []
        self.sequence_results:list[Result] = []
        self.sequence_poses:list[list[np.ndarray]] = []
        self.sequence_object:list[str] = []

    def add_sequence(self, sequences: list, result: Result, target_object: str) -> None:
        start_id = 0
        if len(self.sequence_poses) > 0:
            start_id = sum([len(seq) for seq in self.sequence_poses])
        seq_poses = [np.array(seq)[start_id:, :] for seq in sequences]
        self.sequence_poses.append(seq_poses)
        self.sequence_results.append(result)
        # Careful when uncommenting, faulty but unnused component...
        # self.sequence_lengths.append(np.linalg.norm(seq_poses[1:, :2] - seq_poses[:-1, :2]))
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
        self.max_steps = config.max_steps
        self.max_dist = config.max_dist
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
        self.num_seq = SEQ_LEN
        self.square = config.square_im

        if self.multi_object:
            self.episodes, self.scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes,
                                                                                       self.scene_data,
                                                                                       self.object_nav_path)
        else:
            raise RuntimeError("You are running the multi object evaluation with a single object config.")
        if self.actor is not None:
            self.logger = rerun_logger.RerunLogger(self.actor.one_map, False, "",  self.n_agents) if self.log_rerun else None
        self.results_path = "/home/finn/active/MON/results_gibson_multi" if self.is_gibson else "results_multi/"

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

        elif 'continuous' in action.keys():
            agents_ids = list(range(self.n_agents))
            for agent_id in agents_ids:
                # We have a continuous actor
                self.vel_control.angular_velocity = action['continuous']['angular'][agent_id]
                self.vel_control.linear_velocity = action['continuous']['linear'][agent_id]
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
            self.sim.step_physics(self.time_step)

    def display_results(self, data, sort_by):
        def printkey():
            print("\nReading key:")
            print("\tSUCCESS: Object was reached by one of the agents.")
            print(f"\tFAILURE_OOT: Agents ran out of iterations ({self.max_steps}) before finding the object.")
            print("\tFAILURE_MISDETECT: An agent misdetected the target object.")
            print("\tFAILURE_ALL_EXPLORED: Agents explored the whole scene and couldn't find object (object exists in scene).")
            print("\tFAILURE_NOT_REACHED: An agent detected the object, but could not get to it.")
            print("\tFAILURE_STUCK: All agents soft locked themselves.")
            print("\tProgress: Average proportion of objects found in the episode.")
            print("\tSPL: Success weighted by path length. Penalize successes with deviation from optimal path a priori.")
            print("\topt_PL: Average optimal path length to reach target.")
            print("\ts: All the objects of the scene where found.")
            print("\ts_spl: All object in scene found penalized with deviation from optimal path a priori.")
            print("\n")

        def calc_prog_per_episode(group):
            successes = (group.groupby('experiment')['state'].apply(lambda x: (x == 1).sum()))
            progress = successes / self.num_seq
            return progress

        def calc_spl_per_episode(group):
            spls_per_exp = group.groupby('experiment')['spl'].sum()
            return spls_per_exp

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in data["state"].unique()})
            progress = calc_prog_per_episode(group)
            spl = calc_spl_per_episode(group)
            s = progress[progress == 1]
            result['Progress'] = progress.mean()
            result['SPL'] = spl.mean()
            result['opt_PL'] = group['opt_path'].mean()
            result['Map Size'] = group['map_size'].mean() / 100
            result['s'] = s.sum() / len(progress)
            result['s_spl'] = spl[progress == 1].sum()/len(progress)

            # Calculate average SPL and multiply by 100
            # avg_spl = group['spl'].mean()
            # result['Average SPL'] = avg_spl

            return result

        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        def has_success(group, seq_id):
            return group[(group['sequence'] == seq_id) & (group['state'] == 1)].shape[0] > 0
        
        # Per-object results
        object_results = data.groupby('object').apply(calculate_percentages, include_groups=False).reset_index()
        object_results = object_results.rename(columns={'object': 'Object'})

        # Per-scene results
        scene_results = data.groupby('scene').apply(calculate_percentages, include_groups=False).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        # Overall results
        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{'Object': 'Overall'} | overall_percentages.to_dict()])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Sorting
        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        # Apply formatting to all columns except the first one (Object/Scene)
        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].map(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].map(format_percentages))

        printkey()

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        data_per_scene = data.groupby('scene')
        sr_per_scene = []
        spl_per_scene = []
        for scene, scene_data in data_per_scene:
            print(f"\nScene: {scene}")
            success_rates = []
            spl_values = []
            seq_numbers = []
            for i in range(self.num_seq):
                sequences = scene_data[scene_data['sequence'] == i]
                if len(sequences) > 0:
                    successful_experiments = sequences[sequences['state'] == 1]
                    spl = sequences['spl'].mean() * SEQ_LEN
                    success_rate = len(successful_experiments) / len(sequences)

                    success_rates.append(success_rate)
                    spl_values.append(spl)
                    seq_numbers.append(i)
                    print(f"  Sequence {i}:")
                    print(f"    Num of experiments: {len(sequences)}")
                    print(f"    Overall SPL: {spl:.4f}")
                    print(f"    Fraction of successful experiments: {success_rate:.2%}")
                else:
                    print(f"  Sequence {i}: No data")
                    success_rates.append(0)
                    spl_values.append(0)
            sr_per_scene.append(success_rates)
            spl_per_scene.append(spl_values)
        print(f"SPL per scene: {np.mean(np.array(spl_per_scene), axis=0)}, Success Rate per scene: {np.mean(np.array(sr_per_scene), axis=0)}")

        sr_df = pd.DataFrame(sr_per_scene).T.stack().reset_index()
        sr_df.columns = ["sequence","experiment","success_rate"]

        sns.lineplot(data=sr_df, x="sequence", y="success_rate",
                    estimator="mean", errorbar=("ci",95), marker="o", ax=ax1)

        spl_df = pd.DataFrame(spl_per_scene).T.stack().reset_index()
        spl_df.columns = ["sequence","experiment","spl"]

        sns.lineplot(data=spl_df, x="sequence", y="spl",
                    estimator="mean", errorbar=("ci",95), marker="o", ax=ax2)

        # Set up Success Rate subplot
        ax1.set_xlabel('Sequence Number')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate per Sequence')
        # ax1.legend()
        ax1.grid(True)

        # Set up SPL subplot
        ax2.set_xlabel('Sequence Number')
        ax2.set_ylabel('SPL')
        ax2.set_title('SPL per Sequence')
        # ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('output_plot.png')

        # selected_experiment_ids = successful_experiments['experiment'].unique()
        # experiments_with_second_success = successful_experiments.groupby('experiment').filter(
        #     lambda x: has_success(x, 1))
        # successful_second_ids = experiments_with_second_success['experiment'].unique()
        # fraction_successful = len(successful_second_ids) / len(selected_experiment_ids) if len(
        #     selected_experiment_ids) > 0 else 0
        #
        # # Calculate conditional SPL for each experiment
        # second_sequences = data[(data['state'] == 1) & (data['sequence'] == 1)]
        # conditional_spl = second_sequences['spl'].mean()
        # print(f"\nOverall Conditional SPL (second sequence, given first success): {conditional_spl:.4f}")
        #
        # print(f"Fraction of successful first experiments: {len(selected_experiment_ids)/len(all_ids):.2%}")
        # print(f"Fraction of successful second, conditioned on first: {fraction_successful:.2%}")

    def read_results(self, path, sort_by, data_pkl=None):
        # !!! SPL is not properly defined for multi agents... Find another metric
        def compute_spl(sim, poses, scenes, experiment_num, seq_num):
            spl_agents = []
            for pose in poses:
                path_length = np.linalg.norm(pose[1:, :3] - pose[:-1, :3], axis=1).sum()
                start_pos = pose[0, :3]
                pos = np.array([-start_pos[1], start_pos[2], -start_pos[0]])
                
                floor_data = scenes[self.episodes[experiment_num].scene_id].floors[self.episodes[experiment_num].floor_id]
                possible_objs = floor_data.objects[self.episodes[experiment_num].obj_sequence[seq_num]]
                
                best_dist = np.inf
                obj_found = False
                for obj in possible_objs:
                    dist, _ = object_nav_gen.get_geodesic(pos, sim, obj, correct_start=True)
                    if dist is not None:
                        obj_found = True
                        if dist < best_dist:
                            best_dist = dist

                if not obj_found:
                    pbar.write(f"Warning: No object found for sequence {seq_num} in experiment {experiment_num}")
                spl_agents.append(min(1.0, best_dist/ max(path_length, best_dist)))
            return np.max(spl_agents)

        if data_pkl is not None:
            with open(data_pkl, 'rb') as f:
                data = pickle.load(f)
            self.display_results(data, sort_by)
            return data

        from eval.dataset_utils import gen_multiobject_dataset
        from eval.dataset_utils.object_nav_utils import object_nav_gen
              
        state_dir = os.path.join(path, 'state')
        pose_dir = os.path.join(path, "trajectories")

        # Iterate through all files in the state directory
        data = []
        episodes, scene_data = HM3DDataset.load_hm3d_episodes(episodes:=[], scene_data:={}, gen_multiobject_dataset.path_to_hm3d_objectnav_v2)
        gen_multiobject_dataset.load_scenes(episodes, scene_data, {}, {}, scenes:={})
        loaded_scenes = set()
        sim = None
        for filename in (pbar:=tqdm.tqdm(sorted(os.listdir(state_dir)))):
            if filename.startswith('state_') and filename.endswith('.txt'):
                try:
                    # Extract the experiment number from the filename
                    experiment_num = int(filename[6:-4])  # removes 'state_' and '.txt'

                    # Read the content of the file
                    with open(os.path.join(state_dir, filename), 'r') as file:
                        content = file.read().strip()

                    state_values = [int(val) for val in content.split(',')]

                    for seq_num, value in enumerate(state_values):
                        if value == 1:
                            poses = [np.genfromtxt(os.path.join(pose_dir, f"poses_{experiment_num}_{seq_num}_{agent_id}.csv"), delimiter=",") for agent_id in range(self.n_agents)]
                            if len(poses[0].shape) == 1:
                                poses = [pose.reshape((1, 4)) for pose in poses]

                            if sim is None or not sim.curr_scene_name in self.episodes[experiment_num].scene_id:
                                if sim is not None:
                                    sim.close()
                                sim = gen_multiobject_dataset.build_sim(gen_multiobject_dataset.path_to_hm3d_v0_2, self.episodes[experiment_num].scene_id, gen_multiobject_dataset.start_poses_tilt_angle, True)

                            if self.episodes[experiment_num].scene_id not in loaded_scenes:
                                needs_save = gen_multiobject_dataset.load_all_scene_data(
                                    self.episodes[experiment_num].scene_id,
                                    scenes, scene_data,
                                    viewpoint_conf=object_nav_gen.VPConf(1.0, 0.1, 0.05), 
                                    sim=sim
                                )
                                if needs_save:
                                    pbar.write(f"Storing viewpoints for scene {self.episodes[experiment_num].scene_id}...")
                                    gen_multiobject_dataset.store_viewpoints(scenes, self.episodes[experiment_num].scene_id,"datasets/multi_object_data")
                                loaded_scenes.add(self.episodes[experiment_num].scene_id)
                            
                            spl = compute_spl(sim, poses, scenes, experiment_num, seq_num)

                            top_down_map = maps.get_topdown_map(
                                            sim.pathfinder,
                                            height=poses[0][0,1], #assumes same height for all agents
                                            map_resolution=512,
                                            draw_border=True,
                                        )
                            map_size = top_down_map.shape[0] * top_down_map.shape[1]

                        else:
                            spl = 0
                            map_size = 0

                        data.append({
                            'experiment': experiment_num,
                            'sequence': seq_num,
                            'state': value,
                            'spl': spl / self.num_seq,
                            'map_size': map_size,
                            'opt_path': sum([d[0] for d in self.episodes[experiment_num].best_dist]),
                            'object': self.episodes[experiment_num].obj_sequence[seq_num],
                            'scene': self.episodes[experiment_num].scene_id[15:-10]
                        })

                    if self.episodes[experiment_num].episode_id != experiment_num:
                        pbar.write(f"Warning, experiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")
                except ValueError:
                    pbar.write(f"Warning: Skipping {filename} due to invalid format")
        pbar.close()

        data = pd.DataFrame(data)
        self.display_results(data, sort_by)

        return data

    def save_final_sims(self, episode_id, sequence_id, poses):
        final_sim = (self.actor.one_map.get_similarity_map() + 1.0) / 2.0
        final_sim = monochannel_to_inferno_rgb(final_sim[0])

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
        return self.actor.one_map.metric_to_px(x,y)

    def _px_to_metric(self, px, py):
        return self.actor.one_map.px_to_metric(px,py)

    def _log_ground_truth(self, episode, current_obj):
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
        rr.log("map/ground_truth", rr.Points2D(rotate_frame(pts), colors=[[255, 255, 0]], radii=[1]))

    def evaluate(self):
        results:list[Metrics] = []
        agents_ids = list(range(self.n_agents))

        for n_ep, episode in enumerate(self.episodes):
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
                called_found = False
                while steps < self.max_steps and not called_found:
                    observations = self.sim.get_sensor_observations(agent_ids=agents_ids)
                    for agent_id in agents_ids:
                        observations[agent_id]['state'] = self.sim.get_agent(agent_id=agent_id).get_state()
                        poses[agent_id].append(_get_pose(observations[agent_id]['state']))

                    if self.log_rerun:
                        for agent_id in agents_ids:
                            cam_x = -self.sim.get_agent(agent_id).get_state().position[2]
                            cam_y = -self.sim.get_agent(agent_id).get_state().position[0]
                            rr.log(f"agent_{agent_id}/camera/rgb", rr.Image(observations[agent_id]["rgb"]))
                            rr.log(f"agent_{agent_id}/camera/depth", rr.Image(_normalize(observations[agent_id]["depth"])))
                            rr.log(f"agent_{agent_id}/camera/target", rr.Points2D(positions=[[125, 10]], labels=[f"Target: {current_obj}"], colors=[[255,255,255]]))
                            self.logger.log_pos(cam_x, cam_y, agent_id)

                    actions, called_found = self.actor.act(observations)
                    self.execute_action(actions)

                    if self.log_rerun:
                        self.logger.log_map()

                    if steps % 100 == 0:
                        dists = [get_closest_dist(
                            self.sim.get_agent(agent_id).get_state().position[[0, 2]],
                            self.scene_data[episode.scene_id].object_locations[current_obj],
                            self.is_gibson
                        ) for agent_id in agents_ids]
                        pbar.desc = f"Step {steps}, current object: {current_obj}, episode_id: {episode.episode_id}/{len(self.episodes)}, distance to closest object: {np.min(dists)}"
                    steps += 1
                    pbar.update(1)

                if called_found:
                    dists = [get_closest_dist(self.sim.get_agent(agent_id).get_state().position[[0, 2]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj],
                                            self.is_gibson) for agent_id in agents_ids]
                    if np.min(dists) < self.max_dist:
                        result = Result.SUCCESS
                        pbar.write(f"Object {current_obj} found!")
                    else:
                        failed = True
                        dists_detect = []
                        for mapper in self.actor.mappers:
                            pos = mapper.chosen_detection
                            if pos is not None:
                                pos_metric = self._px_to_metric(pos[0], pos[1])
                                dists_detect.append(get_closest_dist([-pos_metric[1], -pos_metric[0]],
                                                            self.scene_data[episode.scene_id].object_locations[current_obj],
                                                            self.is_gibson))
                        if np.min(dists_detect) < self.max_dist:
                            result = Result.FAILURE_NOT_REACHED
                        else:
                            result = Result.FAILURE_MISDETECT
                        pbar.write(f"Object {current_obj} not found! Dist {np.min(dists)}, detect dist: {np.min(dists_detect)}.")
                    
                else:
                    failed = True
                    num_frontiers = np.sum([len(mapper.nav_goals) for mapper in self.actor.mappers])

                    if _is_stuck(poses, agents_ids):
                        result = Result.FAILURE_STUCK
                    elif num_frontiers == 0:
                        result = Result.FAILURE_ALL_EXPLORED
                    else:
                        result = Result.FAILURE_OOT
                    
                    pbar.write(f"Out of time to find object {current_obj}!")

                results[-1].add_sequence(poses, result, current_obj)

                self.save_final_sims(episode.episode_id, sequence_id, poses)

                sequence_id += 1

            for seq_id, seq in enumerate(results[n_ep].sequence_poses):
                for agent_id, agent_seq in enumerate(seq):
                    np.savetxt(f"{self.results_path}/trajectories/poses_{episode.episode_id}_{seq_id}_{agent_id}.csv", agent_seq, delimiter=",")

            pbar.write(f"Average progress: {sum([m.get_progress() for m in results]) / (len(results))}")
            pbar.close()

            with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
                f.write(','.join(str(results[n_ep].sequence_results[i].value) for i in range(len(results[n_ep].sequence_results))))
