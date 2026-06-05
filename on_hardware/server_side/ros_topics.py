import rclpy # type: ignore
from rclpy.node import Node # type: ignore
from std_msgs.msg import String, Float32MultiArray # type: ignore
from sensor_msgs.msg import Image # type: ignore
from cv_bridge import CvBridge # type: ignore
from rclpy.executors import MultiThreadedExecutor # type: ignore

from scipy.spatial.transform import Rotation as R
import numpy as np
import json

def ros_msg_to_numpy(msg):
    """
    Converts common ROS messages to numpy arrays / dicts.
    Supports: Float32MultiArray, Image, String (JSON-serialized detections or pose)
    """
    msg_type = type(msg).__name__

    if msg_type == 'Float32MultiArray':
        arr = np.array(msg.data, dtype=np.float32)
        if msg.layout.dim:
            dims = [d.size for d in msg.layout.dim]
            arr = arr.reshape(dims)
        return arr

    elif msg_type == 'Image':
        HWC_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if len(HWC_img.shape) == 3:
            return np.transpose(HWC_img, (2, 0, 1))
        # Convert 16UC1 depth from mm to meters
        if msg.encoding == '16UC1':
            return HWC_img.astype(np.float32) / 1000.0
        return HWC_img

    elif msg_type == 'String':
        data = json.loads(msg.data)

        # Pose: {'robot_name': str, 'pose': {'position': {...}, 'orientation': {...}}}
        if isinstance(data, dict) and 'pose' in data:
            p = data['pose']['position']
            o = data['pose']['orientation']
            return {
                'robot_name': data['robot_name'],
                'position':    np.array([p['x'], p['y'], p['z']], dtype=np.float32),
                'orientation': np.array([o['x'], o['y'], o['z'], o['w']], dtype=np.float32),
            }

        # Detections: [{'id': str, 'bbox': {...}, 'results': [...]}, ...]
        elif isinstance(data, list):
            detections = []
            for det in data:
                bbox = det['bbox']
                detections.append({
                    'id':    det['id'],
                    'bbox':  np.array([
                                bbox['center_x'], bbox['center_y'],
                                bbox['size_x'],   bbox['size_y']
                             ], dtype=np.float32),
                    'results': [
                        {'class_id': r['class_id'], 'score': float(r['score'])}
                        for r in det['results']
                    ]
                })
            return detections

        # Unknown JSON structure — return raw
        return data

    return None

def convert_format(msg, topic_type):
    if topic_type == "odometries":
        position = np.asarray(msg["position"], dtype=np.float32)
        quat = np.asarray(msg["orientation"], dtype=np.float32)

        # scipy quaternion format = [x, y, z, w]
        rotation_matrix = R.from_quat(quat).as_matrix()

        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = position

        return T
    return msg

class BaseSubscriber(Node):
    def __init__(self, datatype, topic_name, qos=10):
        super().__init__(f'{topic_name}_subscriber')

        self.latest_msg = None
        self.topic_name = topic_name.split("_")[0]

        self.sub = self.create_subscription(
            datatype,
            topic_name,
            self.callback,
            qos
        )
    
    def callback(self, msg):
        # self.get_logger().info(f"Got message on {self.get_name()}: {type(msg)}")  # will print to console
        self.latest_msg = convert_format(ros_msg_to_numpy(msg), self.topic_name)  # just store

class SubscriptionsListener:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.subscriptions = {}

        self.executor = MultiThreadedExecutor()

        for a in range(n_agents):

            agent_topics = {
                # "detections": BaseSubscriber(String,                      f"detections_{a}"),
                # "extracted_features": BaseSubscriber(Float32MultiArray,   f"extracted_features_{a}"),
                "depths":     BaseSubscriber(Image,                       f"depths_{a}"),
                "rgbs":     BaseSubscriber(Image,                         f"rgbs_{a}"),
                "odometries": BaseSubscriber(String,                      f"odometries_{a}"),
            }

            self.subscriptions[a] = agent_topics

            # IMPORTANT: register nodes in executor
            for node in agent_topics.values():
                self.executor.add_node(node)

    def listen(self):
        self.executor.spin_once(timeout_sec=0.1)

        last_messages = {}

        for a in range(self.n_agents):
            agent_last_messages = {
                key: value.latest_msg
                for key, value in self.subscriptions[a].items()
            }
            last_messages[a] = agent_last_messages

        return last_messages

class BasePublisher(Node):
    def __init__(self, datatype, topic_name, qos=10):
        super().__init__(f'{topic_name}_publisher')

        self.datatype = datatype

        self.pub = self.create_publisher(
            datatype,
            topic_name,
            qos
        )

    def publish(self, data):
        """
        Accept either:
        - already-constructed ROS message
        - raw python value
        """

        # Case 1: already ROS message
        if isinstance(data, self.datatype):
            msg = data

        # Case 2: convert from raw value
        else:
            msg = self.datatype()

            # Generic handling for std_msgs/String-like messages
            if hasattr(msg, "data"):
                msg.data = data
            else:
                raise ValueError(
                    f"Cannot auto-convert data for type {self.datatype}"
                )

        self.pub.publish(msg)

class BroadcastManager:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.publishers = {}

        self.executor = MultiThreadedExecutor()

        for a in range(n_agents):
            agent_publishers = {
                "path": BasePublisher(Float32MultiArray, f"path_{a}"),
                "target_object": BasePublisher(String, f"target_object_{a}"),
            }

            self.publishers[a] = agent_publishers

            for node in agent_publishers.values():
                self.executor.add_node(node)

    def broadcast(self, paths, target_object):
        for a in range(self.n_agents):
            self.publishers[a]["path"].publish(np.array(paths[a]).astype(np.float32).flatten().tolist())
            self.publishers[a]["target_object"].publish(target_object[a])

        self.executor.spin_once(timeout_sec=0.1)

if __name__ == "__main__":
    rclpy.init()

    listener = SubscriptionsListener(1)
    broadcaster = BroadcastManager(1)
    try:
        while rclpy.ok():

            broadcaster.broadcast(
                paths=[[[0, 1], [1, 1]]],
                target_object=["chair"]
            )

            state = listener.listen()
            try:
                print(state[0]["depths"].shape)
                print(state[0]["rgbs"].shape)
                print(state[0]["odometries"].shape)

            except:pass

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            rclpy.shutdown()