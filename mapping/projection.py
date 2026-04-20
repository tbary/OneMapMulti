from .varying_blur import precompute_gaussian_kernel_components, precompute_gaussian_sum_els, gaussian_kernel_sum, compute_gaussian_kernel_components
from config import MappingConf
from onemap_utils import ceildiv

from enum import Enum
import numpy as np
from typing import Tuple
import torch
import warnings
from functools import wraps

def check_camera_initialized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "camera_initialized", False):
            raise RuntimeError("Camera matrix not set, please set camera matrix first")
        return func(self, *args, **kwargs)
    return wrapper

def rotate_pcl(pointcloud: torch.Tensor,tf_camera_to_episodic: torch.Tensor) -> torch.Tensor:
    # TODO We might be interested in a complete 3d rotation if the camera is not perfectly horizontal
    rotation_matrix = tf_camera_to_episodic[:3, :3]

    yaw = torch.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    r = torch.tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]], dtype=torch.float32).to("cuda")
    pointcloud[:, :2] = (r @ pointcloud[:, :2].T).T
    return pointcloud

def _ensure_cuda(values: torch.Tensor, depth:torch.Tensor):
        if not values.is_cuda:
            warnings.warn("Values tensor moved to CUDA")
            values = values.cuda()

        if not depth.is_cuda:
            warnings.warn("Depth tensor moved to CUDA")
            depth = depth.cuda()

        return values, depth

def _patch_averaged_depth(values: torch.Tensor, depth:torch.Tensor):
    nh = values.shape[0]
    nw = values.shape[1]
    h = depth.shape[0]
    w = depth.shape[1]
    # TODO: this is possibly inaccurate, the patch_size might not add up and introduce errors
    patch_size_h = ceildiv(h, nh)
    patch_size_w = ceildiv(w, nw)

    pad_h = patch_size_h * nh - h
    pad_w = patch_size_w * nw - w
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    depth_padded = np.pad(depth, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)))
    return depth_padded.reshape(nh, patch_size_h, nw, patch_size_w).mean(axis=(1, 3))

def _smooth_depth(depth: torch.Tensor, kernel_size=11):
    smoothed_depth = depth.clone()

    # TODO this will be wrong for sub-sampled as e.g. fx will be wrong
    mask = smoothed_depth == float('inf')
    smoothed_depth[mask] = smoothed_depth[~mask].max()

    smoothed_depth = -torch.nn.functional.max_pool2d(
        -smoothed_depth.unsqueeze(0), 
        kernel_size,
        padding=kernel_size // 2,
        stride=1
    ).squeeze(0)

    # TODO Gaussian Blur temporarily disabled
    # smoothed_depth = F.gaussian_blur(smoothed_depth, [31, 31], sigma=4.0)

    return smoothed_depth

class DenseProjectionType(Enum):
    INTERPOLATE = "interpolate"
    SUBSAMPLE = "subsample"

class Projection:
    def __init__(self,
                 feature_dim: int,
                 config: MappingConf,
                 dense_projection: DenseProjectionType = DenseProjectionType.INTERPOLATE,
                 ) -> None:

        assert isinstance(dense_projection, DenseProjectionType), "Invalid dense_projection. It should be one of DenseProjection."

        self.config = config

        self.dense_projection = dense_projection

        self.map_center_cells = torch.tensor([self.config.n_points // 2, self.config.n_points // 2], dtype=torch.int32).to("cuda")
        self.cell_size = self.config.size / self.config.n_points
        self.feature_dim = feature_dim

        self.fx_fy_cx_cy = None
        self.camera_initialized = False

        self.kernel_half = int(np.round(config.blur_kernel_size / self.cell_size))
        self.kernel_size = self.kernel_half * 2 + 1
        self.kernel_components_sum = precompute_gaussian_sum_els(self.kernel_size).to("cuda")
        self.kernel_components = precompute_gaussian_kernel_components(self.kernel_size).to("cuda")
        self.kernel_ids = torch.arange(-self.kernel_half, self.kernel_half + 1).to("cuda")
        self.kernel_ids_x, self.kernel_ids_y = torch.meshgrid(self.kernel_ids, self.kernel_ids)
        self.kernel_ids_x = self.kernel_ids_x.unsqueeze(0)
        self.kernel_ids_y = self.kernel_ids_y.unsqueeze(0)

    def _align_values_and_depth(self, values: torch.Tensor, depth: torch.Tensor):
        if values.shape[0:2] == depth.shape[0:2]:
            # our values align with the depth pixels
            return values.reshape((-1, values.shape[-1])), depth

        # our values are to be considered "patch wise" where we need to project each patch, by averaging the
        # depth values within that patch
        if self.dense_projection == DenseProjectionType.SUBSAMPLE:
            depth_aligned = _patch_averaged_depth(values, depth)
            return values.reshape((-1, values.shape[-1])), depth_aligned
        
        if self.dense_projection == DenseProjectionType.INTERPOLATE:
            values = torch.nn.functional.interpolate(
                values.permute(2, 0, 1).unsqueeze(0),
                size=depth.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            return values.reshape((-1, values.shape[-1])), depth
        
        raise ValueError("Unsupported Dense Projection Mode.")

    def _compute_projection_scores(self, depth_image_smoothed, depth: torch.Tensor, kernel_size=11):
        dx = torch.gradient(depth_image_smoothed, dim=1)[0] / (self.fx_fy_cx_cy[0] / depth.shape[1])
        dy = torch.gradient(depth_image_smoothed, dim=0)[0] / (self.fx_fy_cx_cy[1] / depth.shape[0])

        gradient_magnitude = torch.nn.functional.max_pool2d(
            torch.sqrt(dx ** 2 + dy ** 2).unsqueeze(0), 
            kernel_size, 
            stride=1,
            padding=kernel_size//2
        ).squeeze(0)
        
        scores = (
            (1 - torch.tanh(gradient_magnitude * self.config.gradient_factor)) *
            torch.exp(-((self.config.optimal_object_distance - depth) / self.config.optimal_object_factor) ** 2 / 3.0)
        )
        
        return scores.reshape(-1)

    def set_camera_matrix(self,
                          camera_matrix: np.ndarray
                          ) -> None:
        """
        Sets the camera matrix for the map
        :param camera_matrix: 3x3 numpy array representing the camera matrix
        :return:
        """
        self.camera_initialized = True
        self.fx_fy_cx_cy = camera_matrix[(0, 1, 0, 1), (0, 1, 2, 2)]

    @check_camera_initialized
    @torch.no_grad()
    def project_dense(self,
                      values: torch.Tensor,
                      depth: torch.Tensor,
                      tf_camera_to_episodic: torch.Tensor,
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects the dense features into the map
        TODO We could get rid of sparse tensors entirely and instead use arrays of indices and values to reduce overhead
        :param values: torch tensor of values, shape (hf, wf, feature_dim)
        :param depth: torch tensor of depth values, shape (h, w)
        :param tf_camera_to_episodic:
        :return: (confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped), sparse COO tensor in map coordinates
        """
        values, depth = _ensure_cuda(values, depth)

        values_aligned, depth_aligned = self._align_values_and_depth(values, depth)
        
        depth_image_smoothed = _smooth_depth(depth_aligned)
        
        scores_aligned = self._compute_projection_scores(depth_image_smoothed, depth)

        projected_depth, hole_mask = self._project_depth_camera(depth_aligned, depth.shape)

        rotated_pcl = rotate_pcl(projected_depth, tf_camera_to_episodic)
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        rotated_pcl[:, :2] += torch.tensor([cam_x, cam_y], device='cuda')

        pcl_grid_ids = torch.floor(rotated_pcl[:, :2] / self.cell_size).to(torch.int32)
        pcl_grid_ids[:, 0] += self.map_center_cells[0]
        pcl_grid_ids[:, 1] += self.map_center_cells[1]

        # Filter valid updates
        mask = (
            (depth_aligned.flatten() != float('inf')) & 
            (depth_aligned.flatten() != 0) & 
            (pcl_grid_ids[:, 0] >= self.kernel_half + 1) & 
            (pcl_grid_ids[:, 0] < self.config.n_points - self.kernel_half - 1) & 
            (pcl_grid_ids[:, 1] >= self.kernel_half + 1) & 
            (pcl_grid_ids[:, 1] < self.config.n_points - self.kernel_half - 1) 
        ) # for value map

        z = rotated_pcl[:, 2]
        if hole_mask.nelement() == 0:
            mask_obstacle = (
                mask & 
                (z > self.config.obstacle_min) & 
                (z < self.config.obstacle_max)
            )
        else:
            mask_obstacle = mask & ((z > self.config.obstacle_min) & (z < self.config.obstacle_max) | hole_mask)

        mask &= (scores_aligned > 1e-5)
        mask_obstacle_masked = mask_obstacle[mask]
        scores_masked = scores_aligned[mask]

        pcl_grid_ids_masked = pcl_grid_ids[mask].T
        values_to_add = values_aligned[mask] * scores_masked.unsqueeze(1)

        combined_data = torch.cat((
            values_to_add,
            mask_obstacle_masked.unsqueeze(1),
            torch.ones((values_to_add.shape[0], 1), dtype=torch.uint8, device="cuda"),
            scores_masked.unsqueeze(1)),
            dim=1)  # prepare to aggregate doubles (values pointing to the same grid cell)

        # define the map from unique ids to all ids
        pcl_grid_ids_masked_unique, pcl_mapping = pcl_grid_ids_masked.unique(dim=1, return_inverse=True)
        # coalesce the data
        coalesced_combined_data = torch.zeros((pcl_grid_ids_masked_unique.shape[1], combined_data.shape[-1]),
                                              dtype=torch.float32, device="cuda")
        coalesced_combined_data.index_add_(0, pcl_mapping, combined_data)

        # Extract the data
        data_dim = combined_data.shape[-1]
        obstacle_mapped = coalesced_combined_data[:, data_dim - 3]
        scores_mapped = coalesced_combined_data[:, data_dim - 1].unsqueeze(1)
        sums_per_cell = coalesced_combined_data[:, data_dim - 2].unsqueeze(1)
        new_map = coalesced_combined_data[:, :data_dim - 3]

        # Normalize (from sum to mean)
        new_map /= scores_mapped
        scores_mapped /= sums_per_cell
        obstcl_confidence_mapped = scores_mapped


        # Get all the ids that are affected by the kernel (depth noise blurring)
        ids = pcl_grid_ids_masked_unique
        all_ids_ = torch.zeros((2, ids.shape[1], self.kernel_size, self.kernel_size), device="cuda")
        all_ids_[0] = (ids[0].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_x)
        all_ids_[1] = (ids[1].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_y)
        all_ids, mapping = all_ids_.reshape(2, -1).unique(dim=1, return_inverse=True)

        # Compute the corresponding depths
        depths = ((all_ids - self.map_center_cells.unsqueeze(1)) * self.cell_size - torch.tensor([cam_x, cam_y],
                                                                                 dtype=torch.float32, device="cuda")
                  .unsqueeze(1))

        # And the depth noise
        depth_noise = torch.sqrt(torch.sum(depths ** 2, dim=0)) * self.config.depth_factor / self.cell_size

        # Compute the sum for each kernel centered around a grid cell
        kernel_sums = gaussian_kernel_sum(self.kernel_components_sum, depth_noise).unsqueeze(-1)  # all unique ids

        # remap the depths to all the id's to kernels centered around the original points in ids and
        # compute the sparse inverse kernel elements
        kernels = compute_gaussian_kernel_components(self.kernel_components, depth_noise[mapping].reshape(-1,
                                                                                  self.kernel_size, self.kernel_size))

        coalesced_map_data = torch.zeros((all_ids.shape[1], self.feature_dim), dtype=torch.float32, device="cuda")
        coalesced_scores = torch.zeros((all_ids.shape[1], 1), dtype=torch.float32, device="cuda")
        # Compute the blurred map and blurred scores
        coalesced_map_data.index_add_(0, mapping, (kernels.unsqueeze(-1) *
                                                   new_map.unsqueeze(1).unsqueeze(1)).reshape(-1, self.feature_dim))
        coalesced_scores.index_add_(0, mapping, (kernels * scores_mapped.unsqueeze(1)).reshape(-1, 1))

        # Free up memory to avoid OOM
        torch.cuda.empty_cache()

        # Normalize the map and scores
        coalesced_map_data /= kernel_sums
        coalesced_scores /= kernel_sums

        # Compute the obstacle map
        obstacle_mapped[:] = (obstacle_mapped > 0).to(torch.float32)

        obstacle_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstacle_mapped.unsqueeze(1), (self.config.n_points, self.config.n_points, 1), is_coalesced=True).cpu()
        obstcl_confidence_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstcl_confidence_mapped, (self.config.n_points, self.config.n_points, 1), is_coalesced=True).cpu()

        return torch.sparse_coo_tensor(all_ids, coalesced_scores, (self.config.n_points, self.config.n_points, 1), is_coalesced=True).cpu(), torch.sparse_coo_tensor(all_ids, coalesced_map_data, (self.config.n_points, self.config.n_points, self.feature_dim), is_coalesced=True).cpu(), obstacle_mapped.cpu(), obstcl_confidence_mapped.cpu()

    @check_camera_initialized
    def _project_depth_camera(self,
                             depth: torch.Tensor,
                             camera_resolution: Tuple[int, int],
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the depth into 3D pointcloud. Camera resolution is passed if the depth is subsampled,
        to match value array resolution.
        :param depth: torch Tensor of shape (h, w), not necessarily the same as camera resolution
        :param camera_resolution: tuple of original camera resolution to correct depth if necessary (w, h)
        :return: a point cloud of shape (h * w, 3), where x is depth (points into the image),
                                                          y is horizontal (points left),
                                                          z is vertical (points up)
        """
        # TODO are the "-1" necessary?
        x = torch.arange(0, depth.shape[1], device="cuda") * (camera_resolution[1] - 1) / (depth.shape[1] - 1)
        y = torch.arange(0, depth.shape[0], device="cuda") * (camera_resolution[0] - 1) / (depth.shape[0] - 1)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()
        x_world = (xx - self.fx_fy_cx_cy[2]) * zz / self.fx_fy_cx_cy[0]
        y_world = (yy - self.fx_fy_cx_cy[3]) * zz / self.fx_fy_cx_cy[1]
        z_world = zz
        point_cloud = torch.vstack((z_world, -x_world, -y_world)).T
        if self.config.filter_stairs:
            hole_mask = -y_world < self.config.floor_threshold # todo threshold parameter
            if hole_mask.any():
                scale_factor = self.config.floor_level / -y_world[hole_mask]
                point_cloud[hole_mask] *= scale_factor.unsqueeze(-1)
                return point_cloud, hole_mask

        return point_cloud, torch.empty((0,))

    def metric_to_px(self, x, y):
        epsilon = 1e-9  # Small value to account for floating-point imprecision

        return (
            int(x / self.cell_size + self.map_center_cells[0].item() + epsilon),
            int(y / self.cell_size + self.map_center_cells[1].item() + epsilon))

    def px_to_metric(self, px, py):
        return ((px - self.map_center_cells[0].item()) * self.cell_size,
                (py - self.map_center_cells[1].item()) * self.cell_size)
