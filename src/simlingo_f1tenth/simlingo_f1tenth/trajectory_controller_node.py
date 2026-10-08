#!/usr/bin/env python3
"""Trajectory -> AckermannDriveStamped controller for the F1TENTH stack.

Subscriptions
  plan_topic    autoware_planning_msgs/Trajectory   SimLingo's plan, MAP frame, real metres,
                                                    longitudinal_velocity_mps per point
  pose_topic    nav_msgs/Odometry                   particle filter pose (/pf/pose/odom, map frame)
  speed_topic   nav_msgs/Odometry                   VESC odometry (/odom), twist.linear.x = m/s
  estop_topic   std_msgs/Bool                       true -> command zero speed until false

Publication
  drive_topic   ackermann_msgs/AckermannDriveStamped  -> ackermann_mux "navigation" input (/drive)

Runs at control_hz and publishes on *every* tick: the mux drops the navigation
source 0.2 s after its last message, so a controller that goes quiet is the
same as one commanding zero.  Every failure mode (no plan, stale plan, plan
driven to its end, no pose, e-stop) resolves to an explicit zero-speed command.

Why this exists instead of SimLingo's own PID: the CARLA node runs SimLingo's
throttle/brake/steer PID and hands CARLA pedal positions.  The VESC stack does
not take pedals -- it takes a speed (its own closed loop tracks it) and a
steering angle -- so the longitudinal PID has no place here: the plan's speed
is commanded directly.  Laterally, two options:

  pure_pursuit (default)  geometric Ackermann tracking with the car's real
                          wheelbase; the principled trajectory-to-steering-angle
                          conversion for a car whose geometry is known.
  simlingo_pid            SimLingo's LateralPIDController as tuned for CARLA,
                          run in model units.  Its [-1, 1] output is turned into
                          the path curvature it produces on the CARLA ego and then
                          into this car's steering angle for that curvature at
                          world_scale (see _carla_curvature).

The plan is anchored in the map frame at the pose it was predicted from, and
tracked from the car's *current* pose each tick, so the model's latency shows
up as the plan being stale rather than as a steering artefact.  This is the
arrangement the reference node arrived at after publishing one control per
prediction spun the car out.
"""

from __future__ import annotations

import math
import signal
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from autoware_planning_msgs.msg import Trajectory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool

from simlingo_f1tenth.route_planner import map_to_ego, yaw_from_quaternion


# The CARLA ego as agent_simlingo.py::bicycle_model_forward models it (the
# World on Rails fit the upstream agent itself uses).
_CARLA_FRONT_WB = -0.090769015
_CARLA_REAR_WB = 1.4178275
_CARLA_STEER_GAIN = 0.36848336


def _carla_curvature(steer_norm: float) -> float:
    """Path curvature (1/model m, same sign as steer) of the CARLA ego at steer in [-1, 1]."""
    beta = math.atan(_CARLA_REAR_WB / (_CARLA_FRONT_WB + _CARLA_REAR_WB)
                     * math.tan(_CARLA_STEER_GAIN * steer_norm))
    return math.sin(beta) / _CARLA_REAR_WB


class TrajectoryControllerNode(Node):

    def __init__(self) -> None:
        super().__init__("trajectory_controller")

        # ── topics ───────────────────────────────────────────────────────────
        self.declare_parameter("plan_topic",  "/simlingo/plan")
        self.declare_parameter("pose_topic",  "/pf/pose/odom")
        self.declare_parameter("speed_topic", "/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("estop_topic", "/simlingo/estop")
        # true: hold zero speed until estop_topic receives false (arming step on the real car)
        self.declare_parameter("start_estopped", False)
        # /pf/pose/odom is the pose of the *laser* (map -> laser TF); base_link
        # sits pose_offset_x metres along the heading from it (bringup_launch.py
        # static TF base_link -> laser is +0.27 m, so laser -> base_link is -0.27).
        self.declare_parameter("pose_offset_x", -0.27)

        # ── loop / safety ────────────────────────────────────────────────────
        self.declare_parameter("control_hz",        20.0)
        self.declare_parameter("max_plan_age_sec",  3.0)      # brake if newest plan is older
        self.declare_parameter("pose_timeout_sec",  0.5)      # brake if localisation stops
        self.declare_parameter("max_speed_mps",     1.0)      # hard cap on commanded speed
        self.declare_parameter("speed_scale",       1.0)      # multiplier on the plan's speed
        self.declare_parameter("min_speed_mps",     0.0)      # floor while a plan is being tracked
        self.declare_parameter("brake_below_mps",   0.04)     # plan speed below this = stop
        self.declare_parameter("max_steering_rad",  0.34)     # joy_teleop scale: +-0.34 rad
        self.declare_parameter("wheelbase_m",       0.25)     # vesc.yaml wheelbase
        # First-order low-pass on the steering command, time constant in s; 0 = off.
        # Every new plan (2 Hz) moves the path under the car: on 2026-10-02 the pure
        # pursuit angle jumped by a median of 5-7 deg, 17-19 deg in one switch out of
        # ten, whatever the lookahead.  For pure pursuit: behind simlingo_pid, which
        # steers on a point ~0.25 m ahead, 0.2 s made the car swing from side to side.
        self.declare_parameter("steer_smoothing_sec", 0.0)
        # Stuck / creep recovery, after agent_simlingo.py::run_step (upstream: 40 s at
        # < 0.1 m/s, then throttle 0.4 for 0.75 s).  At standstill the model keeps
        # predicting standstill; this is what gets it rolling.  creep_after_sec 0 = off.
        # Two things differ from upstream because this is a real drivetrain behind a
        # 2 Hz planner: the kick (creep_speed_mps) lasts until the car measurably rolls
        # (creep_release_speed_mps) -- 0.5 m/s for 0.75 s left it at 0.1 m/s -- and the
        # window then continues at creep_hold_speed_mps, long enough (creep_duration_sec
        # in total) for the model to plan from a moving car; the plans in flight were
        # made at standstill and would brake it again.  The hold is capped by
        # max_speed_mps, the kick is not: it ends as soon as the car rolls, and with the
        # cap at 1.0 m/s a kick of 1.0 m/s (released at 0.3 m/s, hold 0.5 m/s) did not
        # get the car going on 2026-10-02.
        self.declare_parameter("creep_after_sec",    0.0)
        self.declare_parameter("creep_speed_mps",    1.0)
        self.declare_parameter("creep_release_speed_mps", 0.3)
        self.declare_parameter("creep_hold_speed_mps", 0.5)
        self.declare_parameter("creep_duration_sec", 2.0)
        self.declare_parameter("stuck_speed_mps",    0.05)

        # ── lateral controller ───────────────────────────────────────────────
        self.declare_parameter("lateral_controller",  "pure_pursuit")   # | simlingo_pid
        self.declare_parameter("pp_lookahead_min_m",  0.5)
        self.declare_parameter("pp_lookahead_max_m",  1.5)
        self.declare_parameter("pp_lookahead_gain_s", 0.6)    # Ld = clip(gain * v, min, max)
        # simlingo_pid only: model metres per real metre, must equal the
        # inference node's world_scale so lookahead indices mean the same thing.
        # Speed is scaled by it too, whatever speed_world_scale the model is fed:
        # the lookahead is about how fast the car moves through this geometry.
        self.declare_parameter("world_scale",         10.0)
        # simlingo_pid only: multiplier on the PID's gains (k_p, k_i, k_d); 1.0 = as tuned
        # upstream for CARLA.  At 1.0 the car's steering limit is reached at ~13 deg of
        # heading error to a point ~0.25 m ahead (6 cm sideways), and the path moves by
        # about that much with every new plan: the steering was at full lock in 40-60 %
        # of the samples on the car (2026-10-01/02).
        self.declare_parameter("pid_gain",            1.0)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._pose_offset_x = float(p("pose_offset_x"))
        self._control_hz    = float(p("control_hz"))
        self._max_plan_age  = float(p("max_plan_age_sec"))
        self._pose_timeout  = float(p("pose_timeout_sec"))
        self._max_speed     = float(p("max_speed_mps"))
        self._speed_scale   = float(p("speed_scale"))
        self._min_speed     = float(p("min_speed_mps"))
        self._brake_below   = float(p("brake_below_mps"))
        self._max_steer     = float(p("max_steering_rad"))
        self._wheelbase     = float(p("wheelbase_m"))
        self._steer_tau     = max(0.0, float(p("steer_smoothing_sec")))
        self._creep_after   = float(p("creep_after_sec"))
        self._creep_speed   = float(p("creep_speed_mps"))
        self._creep_release = float(p("creep_release_speed_mps"))
        self._creep_hold    = float(p("creep_hold_speed_mps"))
        self._creep_time    = float(p("creep_duration_sec"))
        self._stuck_speed   = float(p("stuck_speed_mps"))
        self._lateral       = str(p("lateral_controller")).lower()
        self._ld_min        = float(p("pp_lookahead_min_m"))
        self._ld_max        = float(p("pp_lookahead_max_m"))
        self._ld_gain       = float(p("pp_lookahead_gain_s"))
        self._world_scale   = float(p("world_scale"))
        self._pid_gain      = float(p("pid_gain"))
        if self._pid_gain <= 0.0:
            raise ValueError("pid_gain must be > 0")
        if self._lateral not in ("pure_pursuit", "simlingo_pid"):
            raise ValueError(f"lateral_controller must be pure_pursuit or simlingo_pid, got {self._lateral}")

        self._pid = None
        if self._lateral == "simlingo_pid":
            from simlingo_ros.simlingo_pid import SimLingoPIDController
            self._pid = SimLingoPIDController(control_hz=self._control_hz, window_rate_compensation=False)
            tc = self._pid.turn_controller
            tc.k_p, tc.k_i, tc.k_d = tc.k_p * self._pid_gain, tc.k_i * self._pid_gain, tc.k_d * self._pid_gain

        # ── state ────────────────────────────────────────────────────────────
        self._plan_xy: Optional[np.ndarray] = None     # [N, 2] map frame
        self._plan_v:  Optional[np.ndarray] = None     # [N]
        self._plan_seq = 0
        self._plan_rx_sec: Optional[float] = None
        self._progress = 0
        self._progress_seq = -1
        self._exhausted_seq = -1
        self._pose: Optional[np.ndarray] = None        # [x, y, yaw] of base_link, map frame
        self._pose_rx_sec: Optional[float] = None
        self._speed = 0.0
        self._steer_out = 0.0           # last steering command, the low-pass state
        self._estop = bool(p("start_estopped"))
        self._stuck_time = 0.0
        self._force_move_until = 0.0
        self._creep_open = False        # a creep window is running
        self._creep_rolling = False     # ...and the car has reached creep_release_speed_mps
        self._last_reason = ""

        # ── ROS I/O ──────────────────────────────────────────────────────────
        self._drive_pub = self.create_publisher(AckermannDriveStamped, str(p("drive_topic")), 10)
        self.create_subscription(Trajectory, str(p("plan_topic")), self._plan_cb, 10)
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        # The particle filter publishes with default (reliable) QoS; a
        # best-effort subscriber matches both reliable and best-effort publishers.
        self.create_subscription(Odometry, str(p("pose_topic")), self._pose_cb, be)
        self.create_subscription(Odometry, str(p("speed_topic")), self._speed_cb, be)
        self.create_subscription(Bool, str(p("estop_topic")), self._estop_cb, 10)
        self.create_timer(1.0 / self._control_hz, self._control_cb)

        self.get_logger().info(
            f"trajectory_controller: {self._lateral}, {self._control_hz:.0f} Hz, "
            f"max_speed={self._max_speed:.2f} m/s, max_steer={self._max_steer:.2f} rad, "
            f"steer smoothing {self._steer_tau:.2f} s, "
            f"wheelbase={self._wheelbase:.2f} m -> {p('drive_topic')}")
        if self._pid is not None:
            tc = self._pid.turn_controller
            self.get_logger().info(
                f"lateral PID: gain x{self._pid_gain:g} -> k_p={tc.k_p:.3f} k_i={tc.k_i:.3f} k_d={tc.k_d:.3f}")
        if self._estop:
            self.get_logger().warn(f"starting E-STOPPED: publish false on {p('estop_topic')} to release")

    # ── callbacks ────────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _plan_cb(self, msg: Trajectory) -> None:
        if len(msg.points) < 2:
            return
        self._plan_xy = np.array([[pt.pose.position.x, pt.pose.position.y] for pt in msg.points])
        self._plan_v  = np.array([pt.longitudinal_velocity_mps for pt in msg.points], dtype=np.float64)
        self._plan_seq += 1
        self._plan_rx_sec = self._now()

    def _pose_cb(self, msg: Odometry) -> None:
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        x = msg.pose.pose.position.x + self._pose_offset_x * math.cos(yaw)
        y = msg.pose.pose.position.y + self._pose_offset_x * math.sin(yaw)
        self._pose = np.array([x, y, yaw])
        self._pose_rx_sec = self._now()

    def _speed_cb(self, msg: Odometry) -> None:
        self._speed = float(msg.twist.twist.linear.x)

    def _estop_cb(self, msg: Bool) -> None:
        if bool(msg.data) != self._estop:
            self.get_logger().warn(f"e-stop {'ENGAGED' if msg.data else 'released'}")
        self._estop = bool(msg.data)

    # ── control loop ─────────────────────────────────────────────────────────

    def _control_cb(self) -> None:
        now = self._now()
        reason, detail = None, ""
        if self._estop:
            reason = "e-stop"
        elif self._pose is None or self._pose_rx_sec is None:
            reason = "no pose"
        elif now - self._pose_rx_sec > self._pose_timeout:
            reason, detail = "pose stale", f"{now - self._pose_rx_sec:.1f}s"
        elif self._plan_xy is None:
            reason = "no plan"
        elif 0.0 < self._max_plan_age < now - self._plan_rx_sec:
            reason, detail = "plan stale", f"{now - self._plan_rx_sec:.1f}s"
        if reason is not None:
            self._stop(reason, detail)
            return

        pose = self._pose
        ego_pos, ego_yaw = pose[:2], pose[2]
        plan_ego = map_to_ego(self._plan_xy, ego_pos, ego_yaw)      # x fwd, y left

        # Progress: nearest waypoint, monotonic per plan (see reference node
        # for why a half-plane test is wrong here).
        if self._plan_seq != self._progress_seq:
            self._progress_seq = self._plan_seq
            self._progress = 0
        nearest = int(np.argmin(np.linalg.norm(plan_ego, axis=1)))
        self._progress = max(self._progress, nearest)
        ahead = plan_ego[self._progress + 1:]
        ahead_v = self._plan_v[self._progress + 1:]
        if ahead.shape[0] < 2:
            if self._plan_seq != self._exhausted_seq:
                self._exhausted_seq = self._plan_seq
                self.get_logger().warn(
                    f"OUTRUN: plan {self._plan_seq} consumed after "
                    f"{now - self._plan_rx_sec:.2f}s at {self._speed:.2f} m/s -- "
                    "stopping until the next prediction.")
            self._stop("plan exhausted")
            return

        # ── lateral ──────────────────────────────────────────────────────────
        if self._lateral == "pure_pursuit":
            steer, ld_idx = self._pure_pursuit(ahead, self._speed)
        else:
            steer, ld_idx = self._simlingo_lateral(ahead, self._speed)
        if self._steer_tau > 0.0:
            dt = 1.0 / self._control_hz
            steer = self._steer_out + dt / (self._steer_tau + dt) * (steer - self._steer_out)
        self._steer_out = steer

        # ── longitudinal: the plan's own speed, scaled and capped ────────────
        v_plan = float(ahead_v[min(ld_idx, len(ahead_v) - 1)])
        if v_plan < self._brake_below:
            speed = 0.0
        else:
            speed = float(np.clip(v_plan * self._speed_scale, self._min_speed, self._max_speed))

        if self._creep_after > 0.0:
            if abs(self._speed) < self._stuck_speed:
                self._stuck_time += 1.0 / self._control_hz
            else:
                self._stuck_time = 0.0
            if self._stuck_time > self._creep_after:
                self._stuck_time = 0.0
                self._force_move_until = now + self._creep_time
                self._creep_open, self._creep_rolling = True, False
                self.get_logger().warn(
                    f"CREEP: standing for {self._creep_after:.0f} s while tracking a plan -- "
                    f"kick {self._creep_speed:.2f} m/s until the car rolls at "
                    f"{self._creep_release:.2f} m/s, then {min(self._creep_hold, self._max_speed):.2f} m/s, "
                    f"{self._creep_time:.1f} s in total")
            if now < self._force_move_until:
                if not self._creep_rolling and self._speed >= self._creep_release:
                    self._creep_rolling = True
                    self.get_logger().info(
                        f"CREEP: rolling at {self._speed:.2f} m/s after "
                        f"{self._creep_time - (self._force_move_until - now):.2f} s")
                creep = min(self._creep_hold, self._max_speed) if self._creep_rolling else self._creep_speed
                speed = max(speed, creep)
            elif self._creep_open:
                self._creep_open = False
                if not self._creep_rolling:
                    self.get_logger().warn(
                        f"CREEP: the car did not reach {self._creep_release:.2f} m/s within "
                        f"{self._creep_time:.1f} s (now {self._speed:.2f} m/s)")

        self._publish(speed, steer)
        self._last_reason = ""
        self.get_logger().info(
            f"[CTRL] v_cmd={speed:.2f} (plan {v_plan:.2f}) steer={steer:+.3f} rad "
            f"v_act={self._speed:.2f} | plan {self._plan_seq} age={now - self._plan_rx_sec:.2f}s "
            f"{ahead.shape[0]}/{plan_ego.shape[0]} ahead",
            throttle_duration_sec=1.0)

    def _pure_pursuit(self, ahead: np.ndarray, speed: float):
        """Classic pure pursuit on the ego-frame path. Returns (steering_rad, index used)."""
        ld = float(np.clip(self._ld_gain * max(speed, 0.0), self._ld_min, self._ld_max))
        dists = np.linalg.norm(ahead, axis=1)
        idx = int(np.argmax(dists >= ld)) if np.any(dists >= ld) else len(ahead) - 1
        gx, gy = ahead[idx]
        d = max(float(dists[idx]), 1e-3)
        alpha = math.atan2(gy, gx)
        steer = math.atan2(2.0 * self._wheelbase * math.sin(alpha), d)
        return float(np.clip(steer, -self._max_steer, self._max_steer)), idx

    def _simlingo_lateral(self, ahead_ego: np.ndarray, speed: float):
        """SimLingo's LateralPIDController in model units; [-1, 1] -> steering angle.

        The PID's output means "this much steer on the CARLA ego", i.e. a path
        curvature.  The same path at 1:world_scale has world_scale times that
        curvature, and this car reaches it at atan(wheelbase * curvature).  Mapping
        [-1, 1] straight onto max_steering_rad gave about half the curvature the
        PID expects for its output; with this mapping the car's steering limit is
        reached at about half of the PID's range (its turning circle, scaled up,
        is twice the CARLA ego's).
        """
        route_model = np.stack([ahead_ego[:, 0], -ahead_ego[:, 1]], axis=1) * self._world_scale
        route_interp = self._pid.interpolate_waypoints(route_model)
        steer_norm = float(np.clip(self._pid.turn_controller.step(route_interp, speed * self._world_scale),
                                   -1.0, 1.0))
        curvature_real = _carla_curvature(steer_norm) * self._world_scale        # 1/m, > 0 = right
        # CARLA steer > 0 is right; ackermann steering_angle > 0 is left.
        steer = -math.atan(self._wheelbase * curvature_real)
        return float(np.clip(steer, -self._max_steer, self._max_steer)), 0

    # ── output ───────────────────────────────────────────────────────────────

    def _stop(self, reason: str, detail: str = "") -> None:
        """Zero-speed command. Logged once per reason, then a throttled reminder."""
        if reason != self._last_reason:
            self.get_logger().warn(f"commanding stop: {reason} {detail}".rstrip())
            self._last_reason = reason
        else:
            self.get_logger().warn(f"still stopped: {reason} {detail}".rstrip(),
                                   throttle_duration_sec=5.0)
        self._stuck_time = 0.0
        self._force_move_until = 0.0
        self._creep_open = False
        self._steer_out = 0.0
        self._publish(0.0, 0.0)

    def brake_on_exit(self) -> None:
        """Zero-speed commands on shutdown. ackermann_mux only stops forwarding when
        /drive goes quiet; nothing downstream of it commands the VESC to stop."""
        for _ in range(10):
            self._publish(0.0, 0.0)
            time.sleep(0.02)

    def _publish(self, speed: float, steer: float) -> None:
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed = float(speed)
        cmd.drive.steering_angle = float(steer)
        self._drive_pub.publish(cmd)


def main(args=None) -> None:
    # Own the signals: rclpy's handler shuts the context down first, after which
    # the zero-speed commands in brake_on_exit() could not be published.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TrajectoryControllerNode()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.brake_on_exit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
