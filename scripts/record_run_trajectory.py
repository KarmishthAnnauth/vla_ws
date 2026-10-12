#!/usr/bin/env python3
"""Driven-trajectory recorder for a route run (runs in the Orin container, started by
run_simlingo_route.sh / run_orion_route.sh / run_minddrive_route.sh next to the e-stop keeper).

Writes what scripts/eval_route_tracking.py needs to measure how far the car strayed
from the global route, one CSV row per localiser pose:

    t,x,y,yaw,v,estop
    1790950105.183,0.0731,-0.1402,0.0123,0.00,1

  * ``t``      receive time on the Orin (s, wall clock; the car's clock may differ);
  * ``x,y``    base_link in ``map`` (real metres): ``/pf/pose/odom`` is the laser pose,
               moved by ``--pose-offset-x`` along the heading (-0.27, as the bridges and
               record_global_route.py do, so it is comparable with the route CSVs);
  * ``yaw``    rad; ``v`` the last ``/odom`` speed (m/s);
  * ``estop``  1 while the run's e-stop is engaged (also before its first message).

Route changes are ``#`` lines, so a file covers every route of one model load:

    # route route_3.csv 0.0738 -0.1377;1.0765 -0.1113;...

the initial one from ``--route-csv``, then one per ``nav_msgs/Path`` on ``/global_path``
(named after the routes/route_*.csv it matches, ``global_path`` otherwise). Each row is
flushed at once: the Orin has frozen hard during runs, and the rows up to then survive.

    record_run_trajectory.py --out log/orion/route_1_..._traj.csv \\
        --route-csv routes/route_1.csv --estop-topic /orion/estop
"""

import argparse
import glob
import math
import os
import signal
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Bool


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_route_csv(path: str) -> np.ndarray:
    """``x,y[,v]`` rows; header lines and '#' comments are skipped (as route_planner.load_route_csv)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = [p for p in line.strip().replace(";", ",").split(",") if p.strip()]
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                continue
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def route_name(points: np.ndarray, routes_dir: str) -> str:
    """The routes/route_*.csv holding these points (1 cm), or 'global_path'."""
    for path in sorted(glob.glob(os.path.join(routes_dir, "route_*.csv"))):
        try:
            r = load_route_csv(path)
        except OSError:
            continue
        if r.shape == points.shape and np.abs(r - points).max() < 0.01:
            return os.path.basename(path)
    return "global_path"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--route-csv", help="the route the run starts with")
    ap.add_argument("--routes-dir", help="where to name /global_path routes from (default: dir of --route-csv)")
    ap.add_argument("--estop-topic", default="/simlingo/estop")
    ap.add_argument("--pose-topic", default="/pf/pose/odom")
    ap.add_argument("--speed-topic", default="/odom")
    ap.add_argument("--route-topic", default="/global_path")
    ap.add_argument("--pose-offset-x", type=float, default=-0.27,
                    help="laser -> base_link along the heading (bringup static TF is +0.27)")
    a = ap.parse_args()
    routes_dir = a.routes_dir or (os.path.dirname(a.route_csv) if a.route_csv else ".")

    # "w" on a file the run script created keeps its (host user) owner
    fh = open(a.out, "w")

    def write(line: str) -> None:
        fh.write(line + "\n")
        fh.flush()

    def write_route(name: str, pts: np.ndarray) -> None:
        write(f"# route {name} " + ";".join(f"{x:.4f} {y:.4f}" for x, y in pts))

    write(f"# recorded by record_run_trajectory.py from {a.pose_topic} (base_link offset "
          f"{a.pose_offset_x:+.2f} m), {a.speed_topic}, {a.estop_topic}, {a.route_topic}")
    write("t,x,y,yaw,v,estop")
    if a.route_csv:
        write_route(os.path.basename(a.route_csv), load_route_csv(a.route_csv))

    rclpy.init()
    node = rclpy.create_node("run_trajectory_recorder")
    state = {"estop": True, "v": 0.0, "n": 0, "last_sync": time.time()}

    def on_pose(msg: Odometry) -> None:
        yaw = yaw_of(msg.pose.pose.orientation)
        x = msg.pose.pose.position.x + a.pose_offset_x * math.cos(yaw)
        y = msg.pose.pose.position.y + a.pose_offset_x * math.sin(yaw)
        write(f"{time.time():.3f},{x:.4f},{y:.4f},{yaw:.4f},{state['v']:.2f},{int(state['estop'])}")
        state["n"] += 1
        if time.time() - state["last_sync"] > 1.0:       # survives a hard freeze of the Orin
            os.fsync(fh.fileno())
            state["last_sync"] = time.time()

    def on_speed(msg: Odometry) -> None:
        state["v"] = float(msg.twist.twist.linear.x)

    def on_estop(msg: Bool) -> None:
        state["estop"] = bool(msg.data)

    def on_route(msg: Path) -> None:
        pts = np.asarray([[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=np.float64)
        if len(pts) >= 2:
            write_route(route_name(pts, routes_dir), pts)

    latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(Odometry, a.pose_topic, on_pose, qos_profile_sensor_data)
    node.create_subscription(Odometry, a.speed_topic, on_speed, qos_profile_sensor_data)
    node.create_subscription(Bool, a.estop_topic, on_estop, 10)
    node.create_subscription(Path, a.route_topic, on_route, latched)
    node.get_logger().info(f"recording {a.pose_topic} to {a.out}")

    signal.signal(signal.SIGTERM, lambda *_: rclpy.try_shutdown())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        fh.flush()
        os.fsync(fh.fileno())
        fh.close()
        print(f"{state['n']} poses written to {a.out}", flush=True)     # rosout is gone after Ctrl-C
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
