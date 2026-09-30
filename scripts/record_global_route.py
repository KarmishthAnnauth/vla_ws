#!/usr/bin/env python3
"""Record a global route for SimLingo from the car's localiser pose.

Writes the ``x,y,velocity`` CSV (map frame, real metres, base_link, NO header
line) that both consumers read:

  * ``waypoint_visualiser_node`` (vla_ws/pure_pursuit, on the car) -> latched
    ``nav_msgs/Path`` on ``/global_path``;
  * ``route_csv:=`` of ``simlingo_realworld_node`` (Orin).

The route only provides SimLingo's two *target points* (the next entries at
least 0.75 m real ahead); the car follows the model's own predicted trajectory,
not this file.  Keep it sparse: CARLA route plans are nodes metres apart.

Live, on the car (localiser running, drive the route by joystick):

    python3 scripts/record_global_route.py --out routes/lap_ccw.csv
    ...drive...  Ctrl-C writes the file.

Offline, from a recorded bag (no ROS graph needed, sqlite3 + rclpy only):

    python3 scripts/record_global_route.py --bag data/bags/sl_lap_145833 --out routes/lap_145833.csv

``/pf/pose/odom`` is the *laser* pose; ``--pose-offset-x -0.27`` (default, the
static base_link -> laser transform) moves it to base_link.  The velocity
column is informational (pure_pursuit_node reads it, the SimLingo bridge does
not); by default it is the ``/odom`` speed at that point, or ``--velocity``.
"""

import argparse
import glob
import math
import os
import sqlite3
import sys
from typing import List, Optional, Tuple


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RouteBuilder:
    """Keeps every pose at least ``min_spacing`` metres (straight line) from the last kept one."""

    def __init__(self, min_spacing: float, pose_offset_x: float, default_velocity: float) -> None:
        self.min_spacing = float(min_spacing)
        self.pose_offset_x = float(pose_offset_x)
        self.default_velocity = float(default_velocity)
        self.points: List[Tuple[float, float, float]] = []
        self.n_in = 0
        self.speed: Optional[float] = None

    def add_odom(self, msg) -> None:
        self.speed = float(msg.twist.twist.linear.x)

    def add_pose(self, msg) -> bool:
        self.n_in += 1
        yaw = yaw_of(msg.pose.pose.orientation)
        x = msg.pose.pose.position.x + self.pose_offset_x * math.cos(yaw)
        y = msg.pose.pose.position.y + self.pose_offset_x * math.sin(yaw)
        if self.points:
            lx, ly, _ = self.points[-1]
            if math.hypot(x - lx, y - ly) < self.min_spacing:
                return False
        v = self.speed if self.speed is not None else self.default_velocity
        self.points.append((x, y, v))
        return True

    def length(self) -> float:
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.points, self.points[1:]))

    def write(self, path: str, close_loop: bool) -> None:
        pts = list(self.points)
        if close_loop and len(pts) >= 2:
            pts.append(pts[0])
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as fh:            # no header: the C++ reader stod()s every field
            for x, y, v in pts:
                fh.write(f"{x:.4f},{y:.4f},{v:.3f}\n")
        print(f"wrote {path}: {len(pts)} waypoints, {self.length():.1f} m, from {self.n_in} poses")


# ── offline: read the bag's sqlite directly ─────────────────────────────────

def record_from_bag(bag: str, builder: RouteBuilder, pose_topic: str, speed_topic: str) -> None:
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry

    dbs = sorted(glob.glob(os.path.join(bag, "*.db3")))
    if not dbs:
        sys.exit(f"no .db3 in {bag}")
    for db in dbs:
        uri = "file:" + os.path.abspath(db) + "?mode=ro&immutable=1"   # read-only, no -wal/-shm side files
        con = sqlite3.connect(uri, uri=True)
        ids = {name: tid for tid, name in con.execute("select id, name from topics")}
        wanted = {ids[t]: t for t in (pose_topic, speed_topic) if t in ids}
        if ids.get(pose_topic) is None:
            sys.exit(f"{pose_topic} not in {db} (topics: {sorted(ids)})")
        q = "select topic_id, data from messages where topic_id in (%s) order by timestamp" % \
            ",".join(str(i) for i in wanted)
        for tid, data in con.execute(q):
            msg = deserialize_message(data, Odometry)
            if wanted[tid] == pose_topic:
                builder.add_pose(msg)
            else:
                builder.add_odom(msg)
        con.close()


# ── live: subscribe on the car ───────────────────────────────────────────────

def record_live(builder: RouteBuilder, pose_topic: str, speed_topic: str) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry

    rclpy.init()
    node = Node("record_global_route")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

    def on_pose(msg):
        if builder.add_pose(msg):
            x, y, v = builder.points[-1]
            node.get_logger().info(f"#{len(builder.points):3d}  x={x:+7.2f} y={y:+7.2f} v={v:.2f}  "
                                   f"({builder.length():.1f} m)")

    node.create_subscription(Odometry, pose_topic, on_pose, qos)
    node.create_subscription(Odometry, speed_topic, builder.add_odom, qos)
    node.get_logger().info(f"recording {pose_topic} (base_link offset {builder.pose_offset_x:+.2f} m, "
                           f"spacing {builder.min_spacing:.2f} m). Drive the route; Ctrl-C to write.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output CSV (x,y,velocity; no header)")
    ap.add_argument("--bag", help="rosbag2 directory to read instead of the live topics")
    ap.add_argument("--pose-topic", default="/pf/pose/odom")
    ap.add_argument("--speed-topic", default="/odom")
    ap.add_argument("--min-spacing", type=float, default=1.0,
                    help="metres between kept waypoints (real; 1.0 = 10 model m for SimLingo)")
    ap.add_argument("--pose-offset-x", type=float, default=-0.27,
                    help="laser -> base_link along the heading (bringup static TF is +0.27)")
    ap.add_argument("--velocity", type=float, default=0.5,
                    help="velocity column when no /odom speed is available")
    ap.add_argument("--close-loop", action="store_true", help="append the first waypoint at the end")
    a = ap.parse_args()

    b = RouteBuilder(a.min_spacing, a.pose_offset_x, a.velocity)
    if a.bag:
        record_from_bag(a.bag, b, a.pose_topic, a.speed_topic)
    else:
        record_live(b, a.pose_topic, a.speed_topic)
    if len(b.points) < 2:
        sys.exit(f"only {len(b.points)} waypoint(s) recorded; nothing written")
    b.write(a.out, a.close_loop)


if __name__ == "__main__":
    main()
