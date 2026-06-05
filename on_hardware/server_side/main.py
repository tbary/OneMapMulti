# eval utils
import os
from .actor import MONActor
from .ros_topics import SubscriptionsListener, BroadcastManager
from config import EvalConf, load_eval_config

import matplotlib.pyplot as plt
import tqdm
import time

import rclpy # type: ignore

# cv2
import cv2

# numpy
import numpy as np

def warmup(listener, eval_config):
    print("Waiting for first messages...")

    for attempt in range(5):
        state = listener.listen()

        all_ready = True
        print(state)
        for n in range(eval_config.n_agents):
            if any(v is None for v in state[n].values()):
                print(f"Agent {n} is still warming up!")
                all_ready = False

        if all_ready:
            print("We're ready to go!")
            return

        time.sleep(1)
    else:
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

def main(actor: MONActor, config: EvalConf, listener:SubscriptionsListener, publisher:BroadcastManager, obj_sequence, results_path):
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
                odometries.append(agent_messages["odometries"])

            paths, agent_called_found = actor.act(images, depths, odometries)
            publisher.broadcast(paths, current_obj)

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

    warmup(listener, eval_config)

    try:
        main(actor, eval_config, listener, publisher, obj_sequence, results_path)
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()