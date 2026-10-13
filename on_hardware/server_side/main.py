# eval utils
import os
from .actor import MONActor
from .ros_topics import SubscriptionsListener, BroadcastManager
from config import EvalConf, load_eval_config
from mapping import rerun_logger

import matplotlib.pyplot as plt
import tqdm
import time

import rclpy # type: ignore

# cv2
import cv2

import rerun as rr
import numpy as np

def _normalize(x:np.ndarray):
    return (x-x.min())/(x.max()-x.min())

def warmup(listener, eval_config, timeout=300):
    print("Waiting for first messages...")

    for attempt in range(timeout):
        state = listener.listen()

        all_ready = True
        for n in range(eval_config.n_agents):
            missing = [k for k, v in state[n].items() if v is None]
            if missing:
                if attempt % 10 == 0:
                    print(f"Agent {n} waiting for: {missing}")
                all_ready = False

        if all_ready:
            print("We're ready to go!")
            return

        time.sleep(1)

    raise ValueError("Some agent topics seem to not broadcast data...")

def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a monochannel float32 image to an RGB representation using the Inferno
    colormap.

    Args:
        image (numpy.ndarray): The input monochannel float32 image.

    Returns:
        numpy.ndarray: The RGB image with Inferno colormap.
    """
    # Normalize the input image to the range [0, 1]
    min_val, max_val = np.min(image), np.max(image)
    peak_to_peak = max_val - min_val
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / peak_to_peak

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    return inferno_colormap

def main(actor: MONActor, config: EvalConf, listener:SubscriptionsListener, publisher:BroadcastManager, obj_sequence, results_path, logger:rerun_logger.RerunLogger):
    agents_ids = list(range(config.n_agents))

    poses = [[] for _ in agents_ids]
    
    actor.reset()

    pbar = tqdm.tqdm(total=None)
    
    sequence_id = 0
    while sequence_id < len(obj_sequence) and rclpy.ok():
        current_obj = obj_sequence[sequence_id]
        actor.set_query(current_obj)

        steps = 0
        agent_called_found = -1
        while steps < config.max_steps and agent_called_found == -1:
            last_messages = listener.listen()
            images, depths, odometries = [],[],[]
            for a in range(config.n_agents):
                agent_messages = last_messages[a]
                images.append(agent_messages["rgbs"])
                depths.append(agent_messages["depths"])
                odom = agent_messages["odometries"]
                odometries.append(odom)
                poses[a].append(np.array([odom[0,3], odom[1,3], odom[2,3], np.arctan2(odom[1, 0], odom[0, 0])]))

                if config.log_rerun:
                    cam_x = agent_messages["odometries"][0,3]
                    cam_y = agent_messages["odometries"][1,3]

                    rr.log(f"agent_{a}/camera/rgb", rr.Image(agent_messages["rgbs"].transpose(1, 2, 0)))
                    rr.log(f"agent_{a}/camera/depth", rr.Image(_normalize(agent_messages["depths"])))
                    rr.log(f"agent_{a}/camera/target", rr.Points2D(positions=[[125, 10]], labels=[f"Target: {current_obj}"], colors=[[255,255,255]]))
                    logger.log_pos(actor.mappers[a].projection, cam_x, cam_y, a)
            paths, agent_called_found = actor.act(images, depths, odometries)
            publisher.broadcast(paths, current_obj, spin=actor.spinning)

            if config.log_rerun:
                logger.log_map()
                rr.log("step", rr.TextLog(steps))
                rr.log("path_updates", rr.TextLog(steps))

            if steps % 100 == 0:
                pbar.desc = f"Step {steps}, current object: {current_obj}"
            steps += 1
            pbar.update(1)

        if agent_called_found != -1:
            pbar.write(f"Object {current_obj} found!")

        else:               
            pbar.write(f"Out of time to find object {current_obj}!")

        save_final_sims(config.n_agents, actor, "hardware", sequence_id, poses, results_path)

        sequence_id += 1

    pbar.close()

def save_final_sims(n_agents, actor: MONActor, episode_id, sequence_id, poses, results_path):
    def _metric_to_px(x, y):
        return actor.projection.metric_to_px(x,y)
    final_sim = (actor.one_map.similarity_map + 1.0) / 2.0
    final_sim = monochannel_to_inferno_rgb(final_sim)

    confs = (actor.one_map.confidence_map > 0).cpu().squeeze().numpy()
    
    final_sim[~confs, :] = [0, 0, 0]
    min_x = np.min(np.where(confs)[0])
    max_x = np.max(np.where(confs)[0])
    min_y = np.min(np.where(confs)[1])
    max_y = np.max(np.where(confs)[1])
    final_sim = final_sim[min_x:max_x, min_y:max_y]
    final_sim = final_sim.transpose((1, 0, 2))
    final_sim = np.flip(final_sim, axis=0)                        # get min and max x and y of confs


    cv2.imwrite(f"{results_path}/similarities/final_sim_{episode_id}_{sequence_id}.png", final_sim)

    for agent_id in range(n_agents):
        # Create the plot
        fig = plt.figure(figsize=(10, 10))
        poses_ = np.array([_metric_to_px(*pos[:2]) for pos in poses[agent_id]])
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
        plt.savefig(f"{results_path}/similarities/path_{episode_id}_{sequence_id}_{agent_id}.svg", format='svg', dpi=300, bbox_inches='tight')

        # Display the plot (optional, comment out if not needed)
        plt.close(fig)

if __name__ == "__main__":
    eval_config = load_eval_config().EvalConf
    print(f"{eval_config.n_agents} agents")
    rclpy.init()
    listener = SubscriptionsListener(eval_config.n_agents)
    publisher = BroadcastManager(eval_config.n_agents)
    
    actor = MONActor(eval_config)
    results_path = "hardware_test"
    os.makedirs(results_path, exist_ok=True)
    obj_sequence = ["chair"]

    run_name = os.environ.get("ONEMAP_RUN_NAME", "onemap_run")
    rrd_path = f"{results_path}/{run_name}.rrd"
    logger = rerun_logger.RerunLogger(actor.one_map, True, rrd_path, eval_config.n_agents) if eval_config.log_rerun else None

    warmup(listener, eval_config)

    try:
        main(actor, eval_config, listener, publisher, obj_sequence, results_path, logger)
    except KeyboardInterrupt: pass
    finally:
        actor = None
        listener = None
        publisher = None

        import gc
        gc.collect()

        if rclpy.ok():
            rclpy.shutdown()