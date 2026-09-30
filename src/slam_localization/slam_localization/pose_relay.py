"""Republish the slam_toolbox localisation result in the particle filter's format.

slam_toolbox publishes the map -> odom transform. For every /odom message this node looks up
map -> laser and publishes it as nav_msgs/Odometry on /pf/pose/odom (frame 'map', pose of the
laser, twist = wheel odometry speed and yaw rate), which is what particle_filter publishes.

slam_toolbox corrects the pose in steps (up to ~30 cm on this car), so the output goes
through PoseSmoother: it moves with the odometry and blends each correction in over
smoothing_time seconds. smoothing_time: 0.0 publishes the raw pose.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
import tf2_ros

from slam_localization.smoothing import PoseSmoother


class PoseRelay(Node):
    def __init__(self):
        super().__init__('pose_relay')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('pose_frame', 'laser')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('output_topic', '/pf/pose/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('smoothing_time', 0.3)
        self.map_frame = self.get_parameter('map_frame').value
        self.pose_frame = self.get_parameter('pose_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.smoother = PoseSmoother(tau=self.get_parameter('smoothing_time').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(Odometry, self.get_parameter('output_topic').value, 1)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_cb, 1)
        self.warned = False

    def odom_cb(self, msg):
        try:
            t_map = self.tf_buffer.lookup_transform(self.map_frame, self.pose_frame, Time())
            t_odom = self.tf_buffer.lookup_transform(self.odom_frame, self.pose_frame, Time())
        except tf2_ros.TransformException as e:
            if not self.warned:
                self.get_logger().warn('waiting for %s -> %s: %s' % (self.map_frame, self.pose_frame, e))
                self.warned = True
            return
        stamp = Time.from_msg(t_odom.header.stamp).nanoseconds * 1e-9
        x, y, th = self.smoother.update(stamp, to_pose(t_map), to_pose(t_odom))
        out = Odometry()
        out.header.stamp = t_odom.header.stamp
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.pose_frame
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.orientation.z = math.sin(th / 2)
        out.pose.pose.orientation.w = math.cos(th / 2)
        out.twist.twist = msg.twist.twist
        self.pub.publish(out)


def to_pose(t):
    q = t.transform.rotation
    return (t.transform.translation.x, t.transform.translation.y,
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def main(args=None):
    rclpy.init(args=args)
    node = PoseRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
