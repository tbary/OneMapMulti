# eval utils
from config import EvalConf
from eval.dataset_utils import *
from habitat.utils.visualizations import maps

from eval.dataset_utils import gen_multiobject_dataset
from eval.dataset_utils.object_nav_utils import object_nav_gen

import matplotlib.pyplot as plt
import tqdm
import seaborn as sns

# os / filsystem
import os

# numpy
import numpy as np

# typing
from typing import List, Dict, Set

# habitat
import habitat_sim

# tabulate
from tabulate import tabulate

# pandas
import pandas as pd

# pickle
import pickle

def _load_sim(sim:habitat_sim.Simulator|None, 
              scene_id:str, 
              loaded_scenes:Set[str], 
              scenes:Dict[str, SceneAccumulated], 
              scene_data:Dict[str, SceneData],
              pbar:tqdm.tqdm)->habitat_sim.Simulator:

    if sim is None or not sim.curr_scene_name in scene_id:
        if sim is not None:
            sim.close()
        sim = gen_multiobject_dataset.build_sim(gen_multiobject_dataset.path_to_hm3d_v0_2, scene_id, gen_multiobject_dataset.start_poses_tilt_angle, True)

    if scene_id not in loaded_scenes:
        needs_save = gen_multiobject_dataset.load_all_scene_data(
            scene_id,
            scenes, scene_data,
            viewpoint_conf=object_nav_gen.VPConf(1.0, 0.1, 0.05), 
            sim=sim
        )
        if needs_save:
            pbar.write(f"Storing viewpoints for scene {scene_id}...")
            gen_multiobject_dataset.store_viewpoints(scenes, scene_id,"datasets/multi_object_data")
        loaded_scenes.add(scene_id)

    return sim


class MultiResultsReader:
    def __init__(self, config: EvalConf):
        self.config = config
        self.num_seq = SEQ_LEN
        self.episodes = []

        if config.multi_object:
            self.episodes, _ = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes, {}, config.object_nav_path)
        else:
            raise RuntimeError("You are running the multi object evaluation with a single object config.")

    def display_results(self, data:pd.DataFrame, sort_by:str):
        def printkey():
            print("\nReading key:")
            print("\tSUCCESS: Object was reached by one of the agents.")
            print(f"\tFAILURE_OOT: Agents ran out of iterations ({self.config.max_steps}) before finding the object.")
            print("\tFAILURE_MISDETECT: An agent misdetected the target object.")
            print("\tFAILURE_ALL_EXPLORED: Agents explored the whole scene and couldn't find object (object exists in scene).")
            print("\tFAILURE_NOT_REACHED: An agent detected the object, but could not get to it.")
            print("\tFAILURE_STUCK: All agents soft locked themselves.")
            print("\tProgress: Average proportion of objects found in the episode.")
            print("\tSPL: Success weighted by path length. Penalize successes with deviation from optimal path a priori.")
            print("\topt_PL: Average optimal path length to reach target.")
            print("\tEpisode Success: All the objects of the scene where found.")
            print("\tEpisode SPL: All object in scene found penalized with deviation from optimal path a priori.")
            print("\n")

        def calculate_percentages(group:pd.DataFrame)->pd.Series:
            def calc_prog_per_episode(group:pd.DataFrame):
                return group.groupby('experiment')['state'].apply(lambda x: (x == 1).sum()) / self.num_seq

            def calc_spl_per_episode(group:pd.DataFrame):
                return group.groupby('experiment')['spl'].sum()
            
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / len(group) for state in data["state"].unique()})
            progress = calc_prog_per_episode(group)
            spl = calc_spl_per_episode(group)

            result['Progress'] = progress.mean()
            result['SPL'] = spl.mean() # avg per experience
            result['Opt. Path Length'] = group['opt_path'].mean()
            result['Map Size'] = group['map_size'].mean() / 100
            result['Episode Success'] = len(progress[progress == 1]) / len(progress)
            result['Episode SPL'] = spl[progress == 1].sum()/len(progress)

            return result
        
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
        object_results = object_results.drop(columns=["Progress", "SPL", "Opt. Path Length", "Map Size", "Episode Success", "Episode SPL"])

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Sorting
        object_results = object_results.sort_values(by="SUCCESS", ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        # Apply formatting to all columns except the first one (Object/Scene)
        format_percentages = lambda val: f"{val:.2%}" if isinstance(val, float) else val
        object_table = object_results.iloc[:, 0].to_frame().join(object_results.iloc[:, 1:].map(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(scene_results.iloc[:, 1:].map(format_percentages))

        printkey()

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        data["success"] = data["state"].apply(lambda x:(x == 1))
        data["spl"] *= self.num_seq

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.lineplot(data=data, x="sequence", y="success",
                    estimator="mean", errorbar=("ci",95), marker="o", ax=ax1)

        sns.lineplot(data=data, x="sequence", y="spl",
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

    def read_results(self, path, sort_by, data_pkl=None, save_pkl=True):      
        # !!! SPL is not properly defined for multi agents... Find another metric
        def compute_spl(sim: habitat_sim.Simulator, poses: List[np.ndarray], scenes:Dict[str, SceneAccumulated], experiment_num:int, seq_num:int, exploiting_agent:int):
            def get_geo_dist(pos, sim, possible_objs):
                dists = []
                for obj in possible_objs:
                    dist=object_nav_gen.get_geodesic(pos, sim, obj, correct_start=True)[0]
                    dists.append(dist if dist is not None else np.inf)
                return dists
            
            floor_data = scenes[self.episodes[experiment_num].scene_id].floors[self.episodes[experiment_num].floor_id]
            possible_objs = floor_data.objects[self.episodes[experiment_num].obj_sequence[seq_num]]

            agent_path_length = np.linalg.norm(poses[exploiting_agent][1:, :3] - poses[exploiting_agent][:-1, :3], axis=1).sum()

            start_poses = np.array([pose[0, [1, 2, 0]] * [-1, 1, -1] for pose in poses])
            path_length_to_objects = np.array([get_geo_dist(start_pos, sim, possible_objs) for start_pos in start_poses]).flatten()

            if not len(path_length_to_objects):
                pbar.write(f"Warning: No object found for sequence {seq_num} in experiment {experiment_num}")

            optimal_path_length = np.min(path_length_to_objects)

            if max(agent_path_length, optimal_path_length) in [0, np.inf]:
                return 1

            return optimal_path_length / max(agent_path_length, optimal_path_length)

        if data_pkl is not None:
            print("Pickle detected!")
            with open(data_pkl, 'rb') as f:
                data = pickle.load(f)
            self.display_results(data, sort_by)
            return data
             
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
                # Extract the experiment number from the filename
                experiment_num = int(filename[6:-4])  # removes 'state_' and '.txt'
                episode_id = self.episodes[experiment_num].episode_id
                if episode_id != experiment_num:
                    pbar.write(f"Warning: experiment_num {experiment_num} does not correctly resolve to episode_id {episode_id}...")

                # Read the content of the file
                with open(os.path.join(state_dir, filename), 'r') as file:
                    content = file.readlines()
                state_values = [int(val) for val in content[0].split(',')]
                finding_agents = [int(val) for val in content[1].split(',')]

                scene_id = self.episodes[experiment_num].scene_id
                sim = _load_sim(sim, scene_id, loaded_scenes, scenes, scene_data, pbar)

                for seq_num, (value, agent) in enumerate(zip(state_values, finding_agents)):
                    if value == 1:
                        poses = [
                            np.genfromtxt(os.path.join(pose_dir, f"poses_{experiment_num}_{seq_num}_{agent_id}.csv"), delimiter=",") 
                            for agent_id in range(self.config.n_agents)
                        ]
                        if len(poses[0].shape) == 1:
                            poses = [pose.reshape((1, 4)) for pose in poses]

                        spl = compute_spl(sim, poses, scenes, experiment_num, seq_num, agent)

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
                        'scene': scene_id[15:-10],
                        'exploit_agent':agent
                    })

        pbar.close()

        data = pd.DataFrame(data)
        if save_pkl:
            data.to_pickle(os.path.join(path, "data.pkl"))

        self.display_results(data, sort_by)

        return data
