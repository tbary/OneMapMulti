"""
Jackal-side bridge between OneMap's server and the Jackal base.

RGB/depth are NOT relayed here — full_stack.launch.py's topic_tools relay
nodes publish rgbs_0/depths_0 directly from the camera.

Subscribes:
  path_0                                 (std_msgs/Float32MultiArray)  ← server waypoints
  spin_0                                 (std_msgs/Bool)               ← server: initial 360 deg scan
  /j100_0751/platform/odom/filtered      (nav_msgs/Odometry)           ← robot pose
  /camera/camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image) ← emergency stop
  /camera/camera/color/camera_info       (sensor_msgs/CameraInfo)      ← emergency stop

Publishes:
  odometries_0                           (std_msgs/String / JSON)      → server odom input
  /j100_0751/platform/cmd_vel_unstamped  (geometry_msgs/Twist)         → Jackal base
  estop_0                                (std_msgs/Float32)            → nearest obstacle (m) in
                                                                         the driving corridor, for bags

Emergency stop (depth_guard.py): forward speed ramps down (min 0.15 m/s) once the gap in
front of the bumper is under 0.75 m and is zero at 0.15 m (camera 0.28 m behind the bumper;
obstacles 6 cm-0.60 m above the floor). No depth for 0.5 s, or a mostly-invalid depth image, also
blocks forward motion. Turning in place is always allowed.

No path_0 message for 1 s (server stopped, crashed, Wi-Fi dropped) drops the path: the robot stops.
spin_0=True from the server (initial 360 deg scan, actor.py) turns the robot in place at 0.6 rad/s.
"""

import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from depth_guard import DepthGuard
from jackal_controller import JackalController

DEPTH_TIMEOUT = 0.5  # s without a depth image -> no forward motion
PATH_TIMEOUT = 1.0   # s without a path_0 message (server sends ~3 Hz) -> drop the path and stop
SPIN_RATE = 0.6      # rad/s for the server's initial 360 deg scan (~10 s), above the ~0.5 rad/s deadband
SPIN_TIMEOUT = 0.5   # s without spin_0=True -> stop spinning


class JackalNode(Node):
    def __init__(self):
        super().__init__("jackal_bridge")

        self._path: np.ndarray | None = None
        self._last_path_t = 0.0
        self._last_spin_t = 0.0      # last spin_0=True from the server
        self._spinning = False
        self._odom: Odometry | None = None

        self._controller = JackalController(
            max_vel=0.8,
            max_ang_vel=1.5,
            angle_threshold=0.1,
            lookahead=1,
        )
        self._guard = DepthGuard()
        self._obstacle_dist = 0.0       # unknown until the first depth image -> blocked
        self._last_depth_t = 0.0
        self._was_limited = False

        # subscriptions
        self.create_subscription(Float32MultiArray, "path_0", self._path_cb, 10)
        self.create_subscription(Bool, "spin_0", self._spin_cb, 10)
        self.create_subscription(Odometry, "/j100_0751/platform/odom/filtered", self._odom_cb, 10)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self._info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self._depth_cb, qos_profile_sensor_data)

        # publishers
        self._odom_pub  = self.create_publisher(String, "odometries_0", 10)
        self._cmd_pub   = self.create_publisher(Twist, "/j100_0751/cmd_vel", 10)
        self._estop_pub = self.create_publisher(Float32, "estop_0", 10)

        self.create_timer(0.1, self._tick)  # 10 Hz

    # ------------------------------------------------------------------
    def _path_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        self._path = arr.reshape(-1, 2) if arr.size >= 2 else None
        self._last_path_t = time.monotonic()

    def _spin_cb(self, msg: Bool):
        if msg.data:
            self._last_spin_t = time.monotonic()

    def _odom_cb(self, msg: Odometry):
        self._odom = msg

    def _info_cb(self, msg: CameraInfo):
        if not self._guard.ready:
            self._guard.set_intrinsics(msg.k[0], msg.k[4], msg.k[2], msg.k[5], msg.width, msg.height)
            self.get_logger().info(f"emergency stop armed ({msg.width}x{msg.height} depth)")

    def _depth_cb(self, msg: Image):
        if not self._guard.ready or msg.encoding not in ("16UC1", "mono16"):
            return
        depth = np.frombuffer(msg.data, dtype=">u2" if msg.is_bigendian else "<u2")
        depth = depth.reshape(msg.height, msg.step // 2)[:, : msg.width].astype(np.float32) / 1000.0
        self._obstacle_dist = self._guard.nearest(depth)
        self._last_depth_t = time.monotonic()

    # ------------------------------------------------------------------
    def _tick(self):
        if self._odom is None:
            return

        p = self._odom.pose.pose.position
        o = self._odom.pose.pose.orientation

        # publish odometry JSON for the server
        odom_json = json.dumps(
            {
                "robot_name": "jackal",
                "pose": {
                    "position": {"x": p.x, "y": p.y, "z": p.z},
                    "orientation": {"x": o.x, "y": o.y, "z": o.z, "w": o.w},
                },
            }
        )
        self._odom_pub.publish(String(data=odom_json))

        # extract yaw from quaternion
        rot = R.from_quat([o.x, o.y, o.z, o.w])
        yaw = float(rot.as_euler("xyz")[2])
        pos = np.array([p.x, p.y], dtype=np.float32)

        # server gone quiet (stopped, crashed, Wi-Fi dropped): don't keep driving an old path
        if self._path is not None and time.monotonic() - self._last_path_t > PATH_TIMEOUT:
            self.get_logger().warn(f"no path from the server for {PATH_TIMEOUT:.0f} s, stopping")
            self._path = None

        # run controller and publish cmd_vel
        linear_x, angular_z = self._controller.control(pos, yaw, self._path)

        # server's initial 360 deg scan: turn in place while spin_0 keeps arriving
        spinning = time.monotonic() - self._last_spin_t < SPIN_TIMEOUT
        if spinning != self._spinning:
            self.get_logger().info("initial scan: turning in place" if spinning else "initial scan: done")
            self._spinning = spinning
        if spinning:
            linear_x, angular_z = 0.0, SPIN_RATE

        # emergency stop: cap forward speed by the nearest obstacle ahead
        dist = self._obstacle_dist if time.monotonic() - self._last_depth_t < DEPTH_TIMEOUT else 0.0
        limited_x = self._guard.limit(linear_x, dist)
        self._estop_pub.publish(Float32(data=float(min(dist, 99.0))))
        limited = linear_x > 0.0 and limited_x < linear_x
        if limited and not self._was_limited:
            reason = "no depth" if dist == 0.0 else f"obstacle at {dist:.2f} m"
            self.get_logger().warn(f"emergency stop: {reason}, forward {linear_x:.2f} -> {limited_x:.2f} m/s")
        self._was_limited = limited
        linear_x = limited_x

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self._cmd_pub.publish(twist)


def main():
    rclpy.init()
    node = JackalNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
