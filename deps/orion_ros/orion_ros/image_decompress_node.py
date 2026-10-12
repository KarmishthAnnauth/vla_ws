#!/usr/bin/env python3
# Decompress a JPEG CompressedImage into a raw sensor_msgs/Image.
#
# Runs on the CONSUMER machine (the Orin). It subscribes to the JPEG-compressed
# camera topic (small — ~150 KB — so it crosses the LAN cheaply) and republishes
# a raw bgr8 Image LOCALLY, so orion_lite_node / orion_withpid_node can subscribe
# to a normal raw Image exactly as before. The heavy 1600x900 raw frame
# (~4.3 MB) never crosses the network — only the JPEG does.
#
# Pair with an image_transport republisher on the CARLA side that produces the
# compressed topic from carla_ros_bridge's raw camera:
#   ros2 run image_transport republish raw compressed \
#     --ros-args -r in:=/carla/hero/rgb_0/image \
#                -r out/compressed:=/carla/hero/rgb_0/image/compressed

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class ImageDecompressNode(Node):
    def __init__(self):
        super().__init__('image_decompress_node')
        self.declare_parameter('in_topic',  '/carla/hero/rgb_0/image/compressed')
        self.declare_parameter('out_topic', '/carla/hero/rgb_0/image_decompressed')
        in_topic  = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value

        self.bridge = CvBridge()
        # Compressed frames are small, so RELIABLE depth=1 transfers fine and
        # keeps only the freshest frame (matches the consumer's latest-only use).
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self._pub = self.create_publisher(Image, out_topic, qos)
        self.create_subscription(CompressedImage, in_topic, self._cb, qos)
        self.get_logger().info(f'Decompressing {in_topic} -> {out_topic} (bgr8)')

    def _cb(self, msg: CompressedImage):
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn('JPEG decode failed', throttle_duration_sec=5.0)
            return
        out = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        # Preserve the sim-time stamp + frame_id — the downstream control
        # timestamp depends on it.
        out.header = msg.header
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImageDecompressNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
