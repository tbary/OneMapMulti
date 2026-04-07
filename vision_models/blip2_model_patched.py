from typing import List

import numpy as np
import requests

from .blip2_model import BLIP2Model

# rerun
import rerun as rr

# torch
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def geometric(x):
    return (1 - 4 ** x) / (1 - 4)


def ceildiv(a, b):
    return -(a // -b)


class BLIP2PatchedModel(BLIP2Model):
    def __init__(self):
        super().__init__()
        self.set_batch_size(21)

    def get_image_features(self, image: np.ndarray) -> torch.Tensor:
        n_levels = 3
        assert n_levels > 0
        if len(image.shape) == 3:
            # add batch channel to numpy image
            image = np.expand_dims(image, axis=0)
        n_images = int(geometric(n_levels))
        pad_dims = []
        images = torch.zeros((n_images, 3, 224, 224)).to(self.device)

        images[0] = self.preprocess_image(image)
        level_ids = np.zeros((n_levels,), dtype=int)
        img = torch.Tensor(image).to(self.device)
        index = 1
        for i in range(1, n_levels):
            level_ids[i] = index
            patch_size_x = ceildiv(img.shape[2], (2 ** i))
            patch_size_y = ceildiv(img.shape[3], (2 ** i))

            padding_size_x = patch_size_x * (2 ** i) - img.shape[2]
            padding_size_y = patch_size_y * (2 ** i) - img.shape[3]
            padded_img = F.pad(img, (0, padding_size_y, 0, padding_size_x))
            unfolded = padded_img.unfold(2, patch_size_x, patch_size_x).unfold(3, patch_size_y, patch_size_y)
            patches = unfolded.contiguous().view(3, -1, patch_size_x, patch_size_y)
            patches = patches.permute(1, 0, 2, 3)

            patches_transformed = self.preprocess_image(patches)
            images[index:index + patches_transformed.shape[0]] = torch.Tensor(patches_transformed)
            pad_dims.append((padding_size_x, padding_size_y))
            index += patches_transformed.shape[0]

        image_feats = self.process_image(images)
        image_feats = image_feats.mean(dim=1)
        image_feats = F.normalize(image_feats, dim=-1)
        plot_level = -1

        patch_feats = torch.zeros((n_images, 256), dtype=torch.float32).to(self.device)
        patch_feats[0] = image_feats[0]

        index = 1
        n_img = 1

        def xy_to_index(x, y, n):
            return x * n + y

        for i in range(1, n_levels):
            n_img *= 4
            n_patch_row = 2 ** i
            if i == plot_level or plot_level == -1:
                image_feats_patched = image_feats[index:index + n_img].reshape(n_patch_row, n_patch_row, -1)
                kk = 0
                for x in range(n_patch_row):
                    for y in range(n_patch_row):
                        id = xy_to_index(x, y, n_patch_row) + level_ids[i]
                        # pixel_sims[x_coord_0:x_coord_1, y_coord_0:y_coord_1] += sims_patched[x, y]
                        if plot_level == -1:
                            patch_feats[id] = image_feats_patched[x, y]
                            parent_patch = xy_to_index(x // 2, y // 2, n_patch_row // 2) + level_ids[i - 1]

                            patch_feats[id] += patch_feats[parent_patch]

                        kk += 1

            index += n_img
        # the last level is the interesting level, has 4**(n_levels - 1) patches
        n_row = 2 ** (n_levels - 1)
        # patch_feats = patch_feats[level_ids[-1]:].reshape((n_row, n_row, -1))/n_levels
        patch_feats = image_feats[level_ids[-1]:].reshape((n_row, n_row, -1))
        return patch_feats.permute(2, 0, 1).unsqueeze(0)

    def compute_similarity(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        # image_feats = F.normalize(image_feats, dim=1)  # B C H W, normalize along C
        # text_feats = F.normalize(text_feats, dim=1)
        corr = torch.einsum('bchw, bc -> bhw', image_feats, text_feats)
        return corr

if __name__ == "__main__":
    pass