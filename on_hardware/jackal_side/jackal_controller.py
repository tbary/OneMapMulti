import numpy as np


class JackalController:
    """
    Pure pursuit controller for the Clearpath Jackal.
    Outputs (linear_x m/s, angular_z rad/s) to be placed directly into a Twist message.
    Logic mirrors planning/controllers.py:HabitatController.compute_velocity but
    maps to the flat 2-DOF Twist space instead of habitat's 3-axis arrays.
    """

    def __init__(
        self,
        max_vel: float = 0.5,
        max_ang_vel: float = 1.0,
        angle_threshold: float = 0.1,
        lookahead: int = 1,
        min_ang_vel: float = 0.5,
        goal_tolerance: float = 0.10,
    ):
        self.max_vel = max_vel
        self.max_ang_vel = max_ang_vel
        self.angle_threshold = angle_threshold
        self.lookahead = lookahead
        # Below this magnitude the Jackal's drive motors don't actually turn
        # the wheels (measured deadband is somewhere between 0.3-0.6 rad/s),
        # so a shrinking angle correction can stall forever just above
        # angle_threshold. Floor the commanded magnitude to stay out of it.
        self.min_ang_vel = min_ang_vel
        # Within this distance of the final waypoint the robot is there: stop. Without it,
        # the target is the point the robot sits on, its direction is noise, and the robot
        # spins and pulses in place (run_05, 52-100 s).
        self.goal_tolerance = goal_tolerance

    def control(self, pos: np.ndarray, yaw: float, path: np.ndarray):
        """
        Args:
            pos:  [x, y] robot position in metric odom frame
            yaw:  robot heading in radians
            path: [N, 2] array of waypoints in the same metric odom frame

        Returns:
            (linear_x, angular_z) — plug straight into geometry_msgs/Twist
        """
        if path is None or len(path) == 0:
            return 0.0, 0.0

        if np.linalg.norm(path[-1] - pos[:2]) < self.goal_tolerance:
            return 0.0, 0.0

        distances = np.linalg.norm(path - pos[:2], axis=1)
        closest = int(np.argmin(distances))
        next_id = min(closest + self.lookahead, len(path) - 1)
        next_pos = path[next_id]

        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]
        desired_angle = np.arctan2(dy, dx)
        angle_diff = (desired_angle - yaw + np.pi) % (2 * np.pi) - np.pi

        if abs(angle_diff) < self.angle_threshold:
            dist_to_goal = np.linalg.norm([dx, dy])
            linear_x = float(np.clip(dist_to_goal, 0.0, self.max_vel))
            angular_z = 0.0
        else:
            linear_x = 0.0
            angular_z = float(np.clip(angle_diff, -self.max_ang_vel, self.max_ang_vel))
            angular_z = np.sign(angular_z) * max(abs(angular_z), self.min_ang_vel)

        return linear_x, angular_z
