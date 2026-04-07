# numpy
import numpy as np

# cv2
import cv2

# rerun
import rerun as rr

from sklearn.manifold import TSNE

import colorsys
from typing import List

def generate_distinguishable_colors(n):
    """
    Generates n visually distinguishable colors in RGB format.
    The first color is always white [256, 256, 256].
    """
    if n < 1:
        return []

    colors = [[256, 256, 256]]  # First color is always white

    if n == 1:
        return colors

    # Evenly distribute colors in HSV space
    for i in range(1, n):
        h = i / n  # hue
        s = 0.7 + 0.3 * np.random.random()  # saturation between 0.7-1.0 for vividness
        v = 0.7 + 0.3 * np.random.random()  # value between 0.7-1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])

    return colors


def log_dino_embeddings_tsne(
    embedding_lists:List[np.ndarray],
    perplexity:int=30,
    random_state:int=42,
):
    num_sets = len(embedding_lists)

    colors = generate_distinguishable_colors(num_sets)

    all_embeddings = np.concatenate(embedding_lists, axis=0)

    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
    )
    reduced = tsne.fit_transform(all_embeddings)

    split_indices = np.cumsum([len(e) for e in embedding_lists])[:-1]
    reduced_lists = np.split(reduced, split_indices)

    all_points = []
    all_colors = []

    for emb, color in zip(reduced_lists, colors):
        all_points.append(emb)
        all_colors.append(np.tile(color, (emb.shape[0], 1)))

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    rr.log(
        "embeddings",
        rr.Points3D(
            positions=all_points,
            colors=all_colors,
        ),
    )


def log_map_rerun(map_, path, needs_orientation=False):
    """
    Applies the inferno colormap to the map and logs it to rerun at the given path
    :param map_: 2D array
    :param path: logging path
    :param needs_orientation:
    :return:
    """
    if needs_orientation:
        map_ = map_.transpose((1, 0))
        map_ = np.flip(map_, axis=0)
    map_ = monochannel_to_inferno_rgb(map_)
    rr.log(path, rr.Image(np.flip(map_, axis=-1)).compress(jpeg_quality=50))


def publish_sim_map(sim_map, br, publisher):
    sim_map = sim_map.transpose((1, 0))
    sim_map = np.flip(sim_map, axis=0)
    sim_map = monochannel_to_inferno_rgb(sim_map)
    # upscale to 1000x1000
    sim_map = cv2.resize(sim_map, (1000, 1000))
    img_msg = br.cv2_to_imgmsg(sim_map, encoding="bgr8")
    publisher.publish(img_msg)

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
