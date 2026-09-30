"""Estimate the car's start pose on the map from one LiDAR scan.

slam_toolbox localisation needs a start pose within about 0.5 m of the truth. This grid-searches
the laser pose around the start box that best lines the current /scan up with the map walls,
and prints the matching base_link pose in the form expected by
    ros2 launch slam_localization localize_launch.py start_pose:=x,y,yaw

    ros2 run slam_localization fit_start_pose [--map track_20260930] [--x -0.5 2.0] [--y -0.6 0.6]
"""
import argparse
import math
import os

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

LASER_X = 0.27  # base_link -> laser offset from f1tenth_stack bringup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='track_20260930')
    ap.add_argument('--x', nargs=2, type=float, default=[-0.5, 2.0], help='laser x search range (m)')
    ap.add_argument('--y', nargs=2, type=float, default=[-0.6, 0.6], help='laser y search range (m)')
    ap.add_argument('--yaw', type=float, default=20.0, help='yaw search half-width (deg)')
    args, ros_args = ap.parse_known_args()

    maps = os.path.join(get_package_share_directory('particle_filter'), 'maps')
    meta = yaml.safe_load(open(os.path.join(maps, args.map + '.yaml')))
    img = cv2.imread(os.path.join(maps, os.path.basename(meta['image'])), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    res, ox, oy = meta['resolution'], meta['origin'][0], meta['origin'][1]
    dist = cv2.distanceTransform((img >= 100).astype(np.uint8), cv2.DIST_L2, 5) * res

    rclpy.init(args=ros_args)
    node = Node('fit_start_pose')
    scans = []
    node.create_subscription(LaserScan, '/scan', scans.append, 1)
    while not scans:
        rclpy.spin_once(node, timeout_sec=0.1)
    s = scans[-1]
    r = np.asarray(s.ranges, np.float32)
    a = s.angle_min + np.arange(len(r)) * s.angle_increment
    ok = np.isfinite(r) & (r > 0.05) & (r < 10.0)
    r, a = r[ok][::3], a[ok][::3]

    def cost(x, y, th):
        c = ((x + r * np.cos(a + th) - ox) / res).astype(int)
        row = (h - (y + r * np.sin(a + th) - oy) / res).astype(int)
        inside = (c >= 0) & (c < w) & (row >= 0) & (row < h)
        e = np.ones(len(r))
        e[inside] = np.minimum(dist[row[inside], c[inside]], 1.0)
        return e.mean(), (e < 0.1).mean()

    best = None
    for x in np.arange(args.x[0], args.x[1] + 1e-9, 0.05):
        for y in np.arange(args.y[0], args.y[1] + 1e-9, 0.05):
            for th in np.radians(np.arange(-args.yaw, args.yaw + 1e-9, 1.5)):
                e, _ = cost(x, y, th)
                if best is None or e < best[0]:
                    best = (e, x, y, th)
    _, bx, by, bth = best
    for x in np.arange(bx - 0.05, bx + 0.051, 0.01):
        for y in np.arange(by - 0.05, by + 0.051, 0.01):
            for th in np.arange(bth - 0.03, bth + 0.031, 0.005):
                e, _ = cost(x, y, th)
                if e < best[0]:
                    best = (e, x, y, th)
    e, x, y, th = best
    hit = cost(x, y, th)[1]
    node.get_logger().info('laser at (%.2f, %.2f, %.1f deg), %.0f%% of scan points on walls'
                           % (x, y, math.degrees(th), hit * 100))
    if hit < 0.7:
        node.get_logger().warn('poor fit: is the car in the start box and nobody next to it?')
    print('start_pose:=%.3f,%.3f,%.4f' % (x - LASER_X * math.cos(th), y - LASER_X * math.sin(th), th))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
