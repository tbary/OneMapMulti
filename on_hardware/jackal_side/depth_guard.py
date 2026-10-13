import numpy as np


class DepthGuard:
    """
    Reactive emergency stop from the forward RealSense depth image.

    Looks only at the 3D points inside the corridor the robot will drive through
    (robot width + margin, between just above the floor and the robot's top) and
    returns the distance to the nearest one. limit() caps the forward speed:
    full speed while the bumper gap is beyond slow_gap, then a linear ramp down
    (never below min_vel, the Jackal's drive deadband is ~0.1 m/s), zero at stop_gap.
    Rotation is never limited, so the planner can still turn away.

    nearest() returns the distance from the camera; the gaps are at the front bumper,
    bumper_offset ahead of the camera (in run_03 the door touched at ~0.28 m depth).
    """

    def __init__(
        self,
        # camera above the floor (m). Floor fit on run_03 depth gives ~0.52 m (0.50-0.56) and
        # ~1 deg up, not the 0.30 m in mapping_conf_jackal.yaml. Keep this at or slightly below
        # the real height: too high makes the floor look like an obstacle.
        cam_height: float = 0.50,
        half_width: float = 0.32,     # Jackal half-width 0.22 m + 0.10 m margin
        min_height: float = 0.06,     # ignore the floor (and small bumps) below this
        max_height: float = 0.60,     # robot top 19" (0.48 m) + margin; desks (~0.7 m) can be driven under
        bumper_offset: float = 0.28,  # camera -> front bumper (m)
        stop_gap: float = 0.15,       # stop with this much space in front of the bumper
        slow_gap: float = 0.75,       # start slowing down at this bumper gap
        min_vel: float = 0.15,        # slowest commanded forward speed before a full stop
        min_points: int = 30,         # need this many points closer than a distance to believe it
        min_valid_frac: float = 0.20, # less valid depth than this in the corridor = blind = unsafe
        stride: int = 4,              # subsample the image (640x480 -> 160x120)
    ):
        self.cam_height = cam_height
        self.half_width = half_width
        self.min_height = min_height
        self.max_height = max_height
        self.stop_dist = bumper_offset + stop_gap   # camera distances
        self.slow_dist = bumper_offset + slow_gap
        self.min_vel = min_vel
        self.min_points = min_points
        self.min_valid_frac = min_valid_frac
        self.stride = stride
        self._rays = None  # (x/z, y/z) per subsampled pixel, from the intrinsics

    def set_intrinsics(self, fx, fy, cx, cy, width, height):
        s = self.stride
        u = np.arange(0, width, s, dtype=np.float32)
        v = np.arange(0, height, s, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        self._rays = ((uu - cx) / fx, (vv - cy) / fy)

    @property
    def ready(self):
        return self._rays is not None

    def nearest(self, depth_m: np.ndarray) -> float:
        """Distance (m) to the nearest obstacle in the driving corridor; 0.0 if blind, inf if clear."""
        z = depth_m[:: self.stride, :: self.stride]
        rx, ry = self._rays
        if z.shape != rx.shape:  # resolution changed without new intrinsics
            return 0.0
        x = rx * z                      # right
        h = self.cam_height - ry * z    # height above floor (optical y points down)
        corridor = (np.abs(x) < self.half_width) & (h > self.min_height) & (h < self.max_height)

        # pixels whose ray stays inside the corridor near the robot, valid or not
        near_rays = (np.abs(rx) < self.half_width / self.slow_dist)
        valid = z > 0
        if near_rays.any() and (valid & near_rays).sum() < self.min_valid_frac * near_rays.sum():
            return 0.0

        d = np.sort(z[corridor & valid])
        if d.size < self.min_points:
            return float("inf")
        return float(d[self.min_points - 1])  # robust nearest: ignore a few speckle pixels

    def limit(self, linear_x: float, dist: float) -> float:
        if linear_x <= 0.0:
            return linear_x
        if dist <= self.stop_dist:
            return 0.0
        if dist >= self.slow_dist:
            return linear_x
        frac = (dist - self.stop_dist) / (self.slow_dist - self.stop_dist)
        return min(linear_x, max(self.min_vel, linear_x * frac))
