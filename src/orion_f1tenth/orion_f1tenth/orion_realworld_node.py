#!/usr/bin/env python3
"""ORION inference on a real F1TENTH car (runs on the Jetson AGX Orin).

Subscriptions
  image_topic   sensor_msgs/CompressedImage | Image   front camera on the Orin
  pose_topic    nav_msgs/Odometry                      particle filter, map frame (/pf/pose/odom)
  speed_topic   nav_msgs/Odometry                      VESC odometry: twist.linear.x, twist.angular.z
  imu_topic     sensor_msgs/Imu                        optional ("" = none): can_bus accel + gyro
  route_topic   nav_msgs/Path                          global route, map frame (optional; see route_csv)

Publications
  plan_topic       autoware_planning_msgs/Trajectory  the prediction, MAP frame, real metres and
                                                      real m/s, anchored at the pose it was made from
  path_topic       nav_msgs/Path                      the same plan for RViz
  cot_topic        std_msgs/String                    the model's text output (empty on the
                                                      planning-only agent config)
  markers          visualization_msgs/MarkerArray     plan line + the near route node

No control is produced here: simlingo_f1tenth's trajectory_controller_node tracks
the plan and talks to the VESC, exactly as for SimLingo.  Structure follows
simlingo_f1tenth/simlingo_realworld_node.py (payload built on the executor
thread, inference on a single worker thread that owns the CUDA handles) with the
payload itself built by orion_f1tenth/orion_model.py the way
orion_ros/orion_withpid_node.py builds it for CARLA.

Six cameras from one
  ORION is a surround-view model.  The one camera feeds CAM_FRONT and (by
  default) the two front-side slots as well; the three rear slots are black.
  The model is thus told the same pixels sit at yaw 0 / -55 / +55 deg and that
  nothing is behind the car.  See side_views.

World scale
  The lab track is a 1:10 replica of the CARLA scene.  Positions, route
  waypoints and speed are multiplied by ``world_scale`` before they reach the
  model, and the predicted waypoints and speed are divided by it on the way
  out.  With world_scale=10 the route planner's 4 m / 50 m discard window
  becomes 0.4 m / 5 m on the floor.  ORION's temporal memory stores ego poses
  in model metres and stamps in real seconds, which stays consistent as long
  as speed_world_scale equals world_scale.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, Imu
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from simlingo_f1tenth.route_planner import (
    RoutePlanner, ego_to_map, load_route_csv, map_to_ego, model_to_map, thin_route, yaw_from_quaternion,
)
from orion_f1tenth.frame_saver import FrameSaver
from orion_f1tenth.orion_model import (
    CMD_FOLLOW, CMD_NAMES, ORION_AGENT_HZ, ORION_MEMORY_MAX_DT, ORION_TRAJ_DT, ROUTE_MAX_DIST,
    ROUTE_MIN_DIST, SPEED_PROFILES, TRAIN_CAM_H, TRAIN_CAM_W, OrionRunner, build_can_bus,
    FakeOrionRunner, command_from_route, extend_plan, format_camera_frame, jpeg_roundtrip, make_views,
)


class _RosLogAdapter:
    def __init__(self, logger) -> None:
        self._l = logger

    def info(self, m):  self._l.info(str(m))
    def warn(self, m):  self._l.warn(str(m))
    def warning(self, m):  self._l.warn(str(m))
    def error(self, m): self._l.error(str(m))


class OrionRealWorldNode(Node):

    def __init__(self) -> None:
        super().__init__("orion_realworld_node")

        # ── model ────────────────────────────────────────────────────────────
        # The container's PRE-BUILT checkout (compiled mmcv ops); the node chdir's
        # there so the config's relative 'ckpts/...' paths resolve.
        self.declare_parameter("orion_repo_path", "/root/Orion")
        self.declare_parameter("orion_config_path", "/root/Orion/adzoo/orion/configs/orion_stage3_agent.py")
        self.declare_parameter("orion_checkpoint_path", "/models/Orion/Orion.pth")
        self.declare_parameter("precision", "fp16")
        # baseline | eager | fast | int8 | lite  (orion_model.SPEED_PROFILES)
        self.declare_parameter("inference_mode", "fast")
        self.declare_parameter("llm_int8_stats",
                               "/benchmarking/alpamayo-autoware/src/orion_ros/engines/llm_act_stats.pt")
        self.declare_parameter("decode_workers", 6)
        self.declare_parameter("profile_stages", False)
        self.declare_parameter("inference_period_sec", 0.05)     # dispatch check; the model sets the pace
        # Build frame N+1's batch while frame N's forward runs (hides the ~100 ms
        # pipeline); started pipeline_margin_ms + <avg prep> before the forward
        # is expected to end so the input is as fresh as in the sequential path.
        self.declare_parameter("pipeline_prep", True)
        self.declare_parameter("pipeline_margin_ms", 40.0)
        # "sensor": the frame's real stamp (honest dt; memory kept while frames
        # are < 2 s apart, i.e. always at ~1 s per frame).  "agent": frame_idx/20
        # as OrionAgent does under synchronous CARLA (false dt here; A/B only).
        self.declare_parameter("timestamp_mode", "sensor")

        # ── topics ───────────────────────────────────────────────────────────
        self.declare_parameter("image_topic",    "/camera/front/image/compressed")
        self.declare_parameter("image_compressed", False)      # auto if topic ends in /compressed
        self.declare_parameter("pose_topic",     "/pf/pose/odom")
        self.declare_parameter("speed_topic",    "/odom")
        self.declare_parameter("imu_topic",      "")
        self.declare_parameter("route_topic",    "/global_path")
        self.declare_parameter("plan_topic",     "/orion/plan")
        self.declare_parameter("path_topic",     "/orion/plan_path")
        self.declare_parameter("cot_topic",      "/orion/cot")
        self.declare_parameter("marker_topic",   "/orion/markers")

        # ── geometry / scale ─────────────────────────────────────────────────
        self.declare_parameter("world_scale",    10.0)
        self.declare_parameter("speed_world_scale", 0.0)       # 0 = world_scale
        # /pf/pose/odom is the laser pose; base_link (rear axle) is 0.27 m behind it.
        # The agent localises from its GNSS at x=-1.4 m of the CARLA vehicle origin,
        # which is about the rear axle of the Lincoln MKZ, so base_link already is
        # the reference point; gnss_mount_offset_x (REAL metres, vehicle frame)
        # moves it further if ever needed.
        self.declare_parameter("pose_offset_x",  -0.27)
        self.declare_parameter("gnss_mount_offset_x", 0.0)
        self.declare_parameter("route_csv",      "")
        self.declare_parameter("route_loop",     False)
        self.declare_parameter("route_min_spacing_m", 0.0)     # REAL metres; 0 keeps every waypoint
        # Driving command (RoadOption 1..6) fed as ego_fut_cmd: "geometry" (default) = LEFT/RIGHT
        # when the route turns by more than command_turn_deg within command_lookahead_m
        # (model metres) ahead, else LANEFOLLOW; "static" = driving_command always.
        self.declare_parameter("command_source", "geometry")
        self.declare_parameter("driving_command", CMD_FOLLOW)
        self.declare_parameter("command_lookahead_m", 15.0)
        self.declare_parameter("command_turn_deg", 35.0)

        # ── image formatting ─────────────────────────────────────────────────
        # Horizontal FOV of the real camera; > 70 crops the central 70 deg the
        # CAM_FRONT intrinsics describe.  0 = unknown, feed the whole frame.
        self.declare_parameter("camera_hfov_deg", 0.0)
        self.declare_parameter("side_views", "copy")           # copy | black
        self.declare_parameter("replicate_jpeg_quality", 20)   # OrionAgent.tick's q20 re-encode

        # ── output ───────────────────────────────────────────────────────────
        # Extrapolate the plan this long past ORION's 3 s horizon (last segment's
        # velocity), so the controller does not run dry while the next ~1 s
        # forward runs.  The ego origin is always prepended.
        self.declare_parameter("plan_extend_sec", 2.0)

        # ── misc ─────────────────────────────────────────────────────────────
        self.declare_parameter("stale_image_sec", 1.0)
        self.declare_parameter("log_waypoints",   True)
        # Directory for one JPEG per plan: CAM_FRONT frame + predicted plan + route. "" = off.
        self.declare_parameter("save_frames_dir", "")
        self.declare_parameter("save_frames_every", 1)      # save every Nth plan
        # No model: FakeOrionRunner returns straight plans (pipeline / controller
        # tests without the weights).  Never for driving.
        self.declare_parameter("fake_model",      False)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._scale         = float(p("world_scale"))
        self._speed_scale   = float(p("speed_world_scale")) or self._scale
        self._pose_offset_x = float(p("pose_offset_x"))
        self._gnss_offset_x = float(p("gnss_mount_offset_x"))
        self._hfov          = float(p("camera_hfov_deg"))
        self._side_views    = str(p("side_views"))
        self._jpeg_q        = int(p("replicate_jpeg_quality"))
        self._stale_image   = float(p("stale_image_sec"))
        self._route_spacing = float(p("route_min_spacing_m"))
        self._log_waypoints = bool(p("log_waypoints"))
        self._timestamp_mode = str(p("timestamp_mode"))
        self._command_source = str(p("command_source"))
        self._driving_command = int(p("driving_command"))
        self._cmd_lookahead = float(p("command_lookahead_m"))
        self._cmd_turn_deg  = float(p("command_turn_deg"))
        self._plan_extend   = float(p("plan_extend_sec"))
        self._pipeline_prep = bool(p("pipeline_prep"))
        self._pipeline_margin_ms = float(p("pipeline_margin_ms"))
        mode = str(p("inference_mode"))
        if self._scale <= 0.0 or self._speed_scale <= 0.0:
            raise ValueError("world_scale must be > 0 and speed_world_scale >= 0")
        if self._timestamp_mode not in ("sensor", "agent"):
            raise ValueError("timestamp_mode must be 'sensor' or 'agent'")
        if self._command_source not in ("static", "geometry"):
            raise ValueError("command_source must be 'static' or 'geometry'")
        if not 1 <= self._driving_command <= 6:
            raise ValueError("driving_command must be a RoadOption 1..6")
        if mode not in SPEED_PROFILES:
            raise ValueError(f"inference_mode must be one of {sorted(SPEED_PROFILES)}")
        make_views(np.zeros((TRAIN_CAM_H, TRAIN_CAM_W, 3), np.uint8), self._side_views)   # validates
        if abs(self._speed_scale - self._scale) > 1e-9:
            self.get_logger().warn(
                f"speed_world_scale {self._speed_scale:g} != world_scale {self._scale:g}: ORION's "
                "temporal memory sees ego displacements that do not match the speed it is told")

        # ── state ────────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None        # 1600x900 BGR, formatted
        self._latest_img_wall: Optional[float] = None
        self._latest_img_stamp: float = 0.0
        self._pose: Optional[np.ndarray] = None               # [x, y, yaw] reference point, map, REAL m
        self._speed = 0.0                                     # REAL m/s
        self._yaw_rate = 0.0                                  # rad/s (odom twist.angular.z)
        self._imu: Optional[Imu] = None
        self._planner = RoutePlanner(min_distance=ROUTE_MIN_DIST, max_distance=ROUTE_MAX_DIST,
                                     loop=bool(p("route_loop")))
        self._frame_idx = 0
        self._route_serial = 0
        self._scene_token = "route-0000"
        self._pending_reset = False
        self._prev_sensor_ts: Optional[float] = None
        self._memoryless = 0
        self._durations: List[float] = []
        self._period_mark: Optional[float] = None
        # prep pipelining
        self._dispatch_lock = threading.Lock()
        self._prepared: Optional[dict] = None
        self._preparing = False
        self._fwd_expected_end: Optional[float] = None
        self._fwd_ms_avg: Optional[float] = None
        self._prep_ms_avg = 150.0
        self._plan_no = 0
        self._saver: Optional[FrameSaver] = None
        if str(p("save_frames_dir")):
            self._saver = FrameSaver(str(p("save_frames_dir")), int(p("save_frames_every")),
                                     logger=self.get_logger(), world_scale=self._scale)
            self.get_logger().info(f"saving plan frames to {p('save_frames_dir')}")

        # ── publishers ───────────────────────────────────────────────────────
        self._plan_pub   = self.create_publisher(Trajectory,  str(p("plan_topic")), 10)
        self._path_pub   = self.create_publisher(Path,        str(p("path_topic")), 10)
        self._cot_pub    = self.create_publisher(String,      str(p("cot_topic")), 10)
        self._marker_pub = self.create_publisher(MarkerArray, str(p("marker_topic")), 10)

        # ── subscribers ──────────────────────────────────────────────────────
        img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=2)
        image_topic = str(p("image_topic"))
        compressed = bool(p("image_compressed")) or image_topic.endswith("/compressed")
        if compressed:
            self.create_subscription(CompressedImage, image_topic, self._image_cb_compressed, img_qos)
        else:
            self.create_subscription(Image, image_topic, self._image_cb_raw, img_qos)
        self.get_logger().info(f"image ({'jpeg' if compressed else 'raw'}): {image_topic}, "
                               f"camera hfov {self._hfov:g} deg ({'crop to 70' if self._hfov > 70 else 'no crop'}), "
                               f"side views: {self._side_views}")

        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, str(p("pose_topic")),  self._pose_cb,  be)
        self.create_subscription(Odometry, str(p("speed_topic")), self._speed_cb, be)
        imu_topic = str(p("imu_topic"))
        if imu_topic:
            self.create_subscription(Imu, imu_topic, self._imu_cb, be)
            self.get_logger().info(f"imu: {imu_topic}")

        route_csv = str(p("route_csv"))
        if route_csv:
            self._set_route(load_route_csv(route_csv), f"csv {route_csv}")
        route_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                               depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, str(p("route_topic")), self._route_cb, route_qos)
        self.get_logger().info(f"route: topic {p('route_topic')}" + (f" (initial: {route_csv})" if route_csv else "")
                               + f", command {self._command_source}"
                               + (f" ({CMD_NAMES[self._driving_command]})" if self._command_source == "static" else ""))

        # ── model, on its own thread ─────────────────────────────────────────
        runner_cls = OrionRunner
        if bool(p("fake_model")):
            runner_cls = FakeOrionRunner
            self.get_logger().error("fake_model=true: NO ORION, straight plans only -- never drive on this")
        self._runner = runner_cls(
            repo_path=str(p("orion_repo_path")), config_path=str(p("orion_config_path")),
            checkpoint_path=str(p("orion_checkpoint_path")), precision=str(p("precision")),
            profile=mode, llm_int8_stats=str(p("llm_int8_stats")),
            decode_workers=int(p("decode_workers")), profile_stages=bool(p("profile_stages")),
            logger=_RosLogAdapter(self.get_logger()),
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active: Optional[Future] = None
        self.get_logger().info(f"Loading ORION ({mode}: ~200 s of weights off the SSD"
                               + (", then minutes of torch.compile in the warm-up" if SPEED_PROFILES[mode]["compile_targets"] else "")
                               + ")...")
        self._executor.submit(self._runner.load).result()

        self._infer_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(float(p("inference_period_sec")), self._timer_cb, callback_group=self._infer_group)
        self.get_logger().info(
            f"ready: world_scale={self._scale:g}, speed scale {self._speed_scale:g}, mode {mode}, "
            f"route window {ROUTE_MIN_DIST:g}-{ROUTE_MAX_DIST:g} model m "
            f"({ROUTE_MIN_DIST / self._scale:.2f}-{ROUTE_MAX_DIST / self._scale:.2f} real m), "
            f"plan extended {self._plan_extend:g} s past the horizon")

    # ── sensor callbacks ─────────────────────────────────────────────────────

    def _store_frame(self, bgr: np.ndarray, stamp) -> None:
        formatted = format_camera_frame(bgr, TRAIN_CAM_W, TRAIN_CAM_H, self._hfov)
        with self._lock:
            self._latest_bgr = formatted
            self._latest_img_wall = time.time()
            self._latest_img_stamp = stamp.sec + stamp.nanosec * 1e-9

    def _image_cb_compressed(self, msg: CompressedImage) -> None:
        bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn("undecodable compressed frame", throttle_duration_sec=2.0)
            return
        self._store_frame(bgr, msg.header.stamp)

    def _image_cb_raw(self, msg: Image) -> None:
        enc = msg.encoding.lower()
        ch = 4 if enc in ("rgba8", "bgra8") else 3
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, ch)
        if enc.startswith("rgb"):
            bgr = np.ascontiguousarray(img[:, :, 2::-1])
        else:
            bgr = np.ascontiguousarray(img[:, :, :3])
        self._store_frame(bgr, msg.header.stamp)

    def _pose_cb(self, msg: Odometry) -> None:
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        d = self._pose_offset_x + self._gnss_offset_x
        x = msg.pose.pose.position.x + d * math.cos(yaw)
        y = msg.pose.pose.position.y + d * math.sin(yaw)
        with self._lock:
            self._pose = np.array([x, y, yaw], dtype=np.float64)

    def _speed_cb(self, msg: Odometry) -> None:
        self._speed = float(msg.twist.twist.linear.x)
        self._yaw_rate = float(msg.twist.twist.angular.z)

    def _imu_cb(self, msg: Imu) -> None:
        self._imu = msg

    def _route_cb(self, msg: Path) -> None:
        if len(msg.poses) < 2:
            self.get_logger().warn("route Path with < 2 poses ignored")
            return
        wps = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses])
        self._set_route(wps, f"{len(wps)}-pose Path")

    def _set_route(self, wps_real: np.ndarray, source: str) -> None:
        wps_real = thin_route(wps_real, self._route_spacing)
        with self._lock:
            self._planner.set_route(wps_real * self._scale)
            # A new route = a new scene for ORION's temporal memory (the agent is
            # rebuilt per route upstream); applied on the inference thread.
            self._route_serial += 1
            self._scene_token = f"route-{self._route_serial:04d}"
            self._pending_reset = True
        length = float(np.linalg.norm(np.diff(wps_real, axis=0), axis=1).sum())
        self.get_logger().info(f"route set from {source}: {len(wps_real)} waypoints, {length:.1f} m real "
                               f"({length * self._scale:.0f} m model), loop={self._planner.loop}, "
                               f"scene {self._scene_token}")

    # ── scheduling ───────────────────────────────────────────────────────────

    def _timer_cb(self) -> None:
        try:
            if self._active is not None and not self._active.done():
                self._maybe_prepare_ahead()
                return
            self._dispatch()
        except Exception as exc:
            self.get_logger().error(f"_timer_cb failed: {type(exc).__name__}: {exc}", throttle_duration_sec=5.0)

    def _dispatch(self) -> None:
        """Start a forward if none is in flight: on the prepared payload, else
        on one built now.  Called from the timer and from the forward-done
        callback; the lock makes exactly one of them dispatch."""
        with self._dispatch_lock:
            if self._active is not None and not self._active.done():
                return
            if self._preparing:
                return                      # the prepare in progress dispatches when done
            payload, self._prepared = self._prepared, None
            built_now = payload is None
            if built_now:
                payload = self._build_payload()
            if payload is None:
                return
            self._fwd_expected_end = (time.time() + (self._fwd_ms_avg or 0.0) / 1e3
                                      if self._fwd_ms_avg is not None else None)
            self._active = self._executor.submit(self._run_inference, payload)
            self._active.add_done_callback(self._on_done)

    def _maybe_prepare_ahead(self) -> None:
        """While a forward is in flight: build the next payload so it is ready
        when the GPU frees up.  Started late on purpose (see pipeline_prep)."""
        if (not self._pipeline_prep or self._prepared is not None or self._preparing
                or self._fwd_expected_end is None or self._fwd_ms_avg is None
                or self._fwd_ms_avg < 2.0 * self._prep_ms_avg + self._pipeline_margin_ms):
            return
        if time.time() < self._fwd_expected_end - (self._prep_ms_avg + self._pipeline_margin_ms) / 1e3:
            return
        self._preparing = True
        try:
            payload = self._build_payload()
            if payload is not None:
                self._prepared = payload
        finally:
            self._preparing = False
        if self._prepared is not None:
            self._dispatch()

    # ── payload ──────────────────────────────────────────────────────────────

    def _build_payload(self) -> Optional[dict]:
        with self._lock:
            bgr, img_wall, img_stamp = self._latest_bgr, self._latest_img_wall, self._latest_img_stamp
            pose = None if self._pose is None else self._pose.copy()
            has_route = self._planner.has_route
        if bgr is None:
            self.get_logger().warn("waiting for image...", throttle_duration_sec=5.0)
            return None
        if pose is None:
            self.get_logger().warn("waiting for pose (/pf/pose/odom)...", throttle_duration_sec=5.0)
            return None
        if not has_route:
            self.get_logger().warn("waiting for route (Path topic or route_csv)...", throttle_duration_sec=5.0)
            return None
        age = time.time() - img_wall
        if age > self._stale_image:
            self.get_logger().warn(f"newest frame is {age:.1f}s old -- camera stalled? skipping",
                                   throttle_duration_sec=2.0)
            return None
        t0 = time.perf_counter()

        # ── ego state, model units ───────────────────────────────────────────
        ego_pos_model = pose[:2] * self._scale
        ego_yaw = float(pose[2])
        speed_real = self._speed
        speed_model = speed_real * self._speed_scale
        imu = self._imu
        if imu is not None:
            # body frame, x forward y left (ROS); accel scales with distance,
            # angular velocity is scale free.  Gyro y flipped as the reference
            # node does for the ros-bridge (agent: -angular_velocity over
            # CARLA's gyro; the bridge negates x and z only).
            acc = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y,
                            imu.linear_acceleration.z]) * self._scale
            gyro = np.array([imu.angular_velocity.x, -imu.angular_velocity.y, imu.angular_velocity.z])
        else:
            acc = np.zeros(3)
            gyro = np.array([0.0, 0.0, self._yaw_rate])
        can_bus, ego_pose, ego_pose_inv, l2g = build_can_bus(ego_pos_model, ego_yaw, speed_model, acc, gyro)

        # ── route: discard passed nodes, near node, command ──────────────────
        with self._lock:
            tp0, _ = self._planner.target_points(ego_pos_model, ego_yaw)   # model ego frame, x fwd y right
            route_idx, route_done = self._planner.index, self._planner.finished()
            if self._command_source == "geometry":
                command = command_from_route(self._planner.route, route_idx, self._cmd_lookahead,
                                             self._cmd_turn_deg, self._planner.loop)
            else:
                command = self._driving_command
            scene_token, frame_idx = self._scene_token, self._frame_idx
            self._frame_idx += 1
            route_ahead = None
            if self._saver is not None:   # the next 4 real m of the route, for the saved frame
                route = self._planner.route
                ahead = np.roll(route, -route_idx, axis=0) if self._planner.loop else route[route_idx:]
                dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ahead, axis=0), axis=1))])
                route_ahead = map_to_ego(ahead[dist <= 4.0 * self._scale], ego_pos_model, ego_yaw)
        if route_done:
            self.get_logger().info("route finished", throttle_duration_sec=5.0)

        if self._timestamp_mode == "agent":
            timestamp = frame_idx / ORION_AGENT_HZ
        else:
            timestamp = img_stamp
        self._check_memory_continuity(img_stamp)

        # ── images: the agent's JPEG-q20 pass, then the six slots ────────────
        front = jpeg_roundtrip(bgr, self._jpeg_q)
        views = make_views(front, self._side_views)
        batch = self._runner.build_batch(views, can_bus, ego_pose, ego_pose_inv, l2g, command,
                                         scene_token, frame_idx, timestamp)
        prep_ms = (time.perf_counter() - t0) * 1000.0
        self._prep_ms_avg = 0.8 * self._prep_ms_avg + 0.2 * prep_ms
        return {
            "batch": batch, "prep_ms": prep_ms, "command": command,
            "speed_model": speed_model, "speed_real": speed_real,
            "near_node": np.asarray(tp0, dtype=np.float64),
            "pose_real": pose, "route_idx": route_idx, "img_age": age,
            "frame_idx": frame_idx, "stamp": self.get_clock().now().to_msg(),
            "bgr": bgr, "route_ahead": route_ahead, "scene": scene_token,
        }

    def _check_memory_continuity(self, sensor_ts: float) -> None:
        prev, self._prev_sensor_ts = self._prev_sensor_ts, sensor_ts
        if prev is None or self._timestamp_mode != "sensor":
            return
        dt = abs(sensor_ts - prev)
        if dt < ORION_MEMORY_MAX_DT:
            return
        self._memoryless += 1
        self.get_logger().warn(
            f"temporal memory: {dt:.2f} s between frames exceeds ORION's {ORION_MEMORY_MAX_DT:.0f} s window, "
            f"memory zeroed, single-frame inference ({self._memoryless} frames so far)",
            throttle_duration_sec=10.0)

    # ── inference (worker thread) ────────────────────────────────────────────

    def _run_inference(self, payload: dict) -> dict:
        t_start = time.perf_counter()
        if self._pending_reset:
            self._runner.reset_memory()
            self._pending_reset = False
            self.get_logger().info(f"context reset for {self._scene_token}: temporal memory and frame counter")
        res = self._runner.infer(payload["batch"])
        period_ms = (t_start - self._period_mark) * 1000.0 if self._period_mark else None
        self._period_mark = t_start
        self._fwd_ms_avg = res.forward_ms if self._fwd_ms_avg is None else 0.8 * self._fwd_ms_avg + 0.2 * res.forward_ms

        # LIDAR_TOP frame, model metres -> ROS ego, real metres -> map at the capture pose.
        pose = payload["pose_real"]
        route_ego_real = res.route_ros_ego / self._scale
        plan_ego = extend_plan(route_ego_real, self._plan_extend)
        plan_map = ego_to_map(plan_ego, pose[:2], pose[2])
        desired_speed_real = res.desired_speed / self._speed_scale
        stamp = payload["stamp"]

        self._plan_pub.publish(self._to_trajectory(plan_map, desired_speed_real, stamp))
        self._path_pub.publish(self._to_path(plan_map, stamp))
        near_map = model_to_map(payload["near_node"] / self._scale, pose[:2], pose[2])
        self._marker_pub.publish(self._to_markers(plan_map, near_map, stamp))
        if res.text:
            self._cot_pub.publish(String(data=res.text))

        horizon = float(np.linalg.norm(np.diff(route_ego_real, axis=0), axis=1).sum())
        period = f" | period {period_ms:.0f} ms" if period_ms else ""
        self._plan_no += 1
        if self._saver is not None:
            plan_model = plan_ego * self._scale
            nn = payload["near_node"].reshape(-1, 2)         # model ego frame, y right -> ROS ego
            self._saver.submit(
                self._plan_no, payload["bgr"], res.preds, plan_model[:len(route_ego_real) + 1],
                plan_model[len(route_ego_real):], payload["route_ahead"],
                np.column_stack([nn[:, 0], -nn[:, 1]]),
                res.text, [
                    f"plan {self._plan_no}  {payload['scene']} frame {payload['frame_idx']}  {time.strftime('%H:%M:%S')}  "
                    f"cmd {CMD_NAMES.get(payload['command'], payload['command'])}  route idx {payload['route_idx']}  "
                    f"forward {res.forward_ms:.0f} ms{period}  frame age {payload['img_age']:.2f} s",
                    f"speed {payload['speed_real']:.2f} m/s real ({payload['speed_model']:.1f} model)  ->  "
                    f"desired {desired_speed_real:.2f} m/s real ({res.desired_speed:.2f} model)  "
                    f"horizon {horizon:.2f} m real",
                ])
        self.get_logger().info(
            f"[ORION] frame {payload['frame_idx']} cmd {CMD_NAMES.get(payload['command'], payload['command'])} "
            f"(route idx {payload['route_idx']}) | speed in {payload['speed_model']:.1f} model m/s | "
            f"desired {res.desired_speed:.2f} model -> {desired_speed_real:.2f} real m/s | "
            f"horizon {horizon:.2f} m real | prep {payload['prep_ms']:.0f} ms, "
            f"forward {res.forward_ms:.0f} ms{period} | frame age {payload['img_age']:.2f}s")
        if self._log_waypoints:
            wps = " ".join(f"({x:.2f},{y:+.2f})" for x, y in route_ego_real)
            self.get_logger().info(f"[ORION] waypoints ego x,y real m: {wps}")
        if res.text:
            self.get_logger().info(f"[ORION] text: {res.text}")
        return {"total_ms": (time.perf_counter() - t_start) * 1000.0 + payload["prep_ms"],
                "forward_ms": res.forward_ms}

    def _on_done(self, fut: Future) -> None:
        try:
            m = fut.result()
        except Exception as exc:
            import traceback
            self.get_logger().error(f"inference failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return
        finally:
            if self._prepared is not None:
                self._dispatch()
        self._durations.append(m["total_ms"])
        if len(self._durations) % 20 == 0:
            d = sorted(self._durations[-20:])
            self.get_logger().info(f"[PROF] last 20: mean {sum(d)/len(d):.0f} ms, min {d[0]:.0f}, max {d[-1]:.0f}")

    # ── message building ─────────────────────────────────────────────────────

    @staticmethod
    def _headings(pts: np.ndarray) -> np.ndarray:
        d = np.diff(pts, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        return np.append(yaw, yaw[-1]) if len(yaw) else np.zeros(len(pts))

    def _to_trajectory(self, plan_map: np.ndarray, speed: float, stamp) -> Trajectory:
        traj = Trajectory()
        traj.header.stamp = stamp
        traj.header.frame_id = "map"
        for i, (pt, yaw) in enumerate(zip(plan_map, self._headings(plan_map))):
            tp = TrajectoryPoint()
            tp.pose.position.x, tp.pose.position.y = float(pt[0]), float(pt[1])
            tp.pose.orientation.z = float(np.sin(yaw / 2.0))
            tp.pose.orientation.w = float(np.cos(yaw / 2.0))
            tp.longitudinal_velocity_mps = float(speed)
            sec = i * ORION_TRAJ_DT
            tp.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
            traj.points.append(tp)
        return traj

    def _to_path(self, plan_map: np.ndarray, stamp) -> Path:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "map"
        for pt, yaw in zip(plan_map, self._headings(plan_map)):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(pt[0]), float(pt[1])
            ps.pose.orientation.z = float(np.sin(yaw / 2.0))
            ps.pose.orientation.w = float(np.cos(yaw / 2.0))
            path.poses.append(ps)
        return path

    def _to_markers(self, plan_map: np.ndarray, near_map: np.ndarray, stamp) -> MarkerArray:
        arr = MarkerArray()
        line = Marker()
        line.header.stamp, line.header.frame_id = stamp, "map"
        line.ns, line.id, line.type, line.action = "orion_plan", 0, Marker.LINE_STRIP, Marker.ADD
        line.scale.x = 0.03
        line.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        line.pose.orientation.w = 1.0
        line.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in plan_map]
        arr.markers.append(line)
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, "map"
        m.ns, m.id, m.type, m.action = "orion_near_node", 1, Marker.SPHERE, Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(near_map[0][0]), float(near_map[0][1]), 0.05
        m.pose.orientation.w = 1.0
        arr.markers.append(m)
        return arr

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._saver is not None:
            self._saver.close()
            self.get_logger().info(f"saved {self._saver.saved} plan frames")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrionRealWorldNode()
    # Two threads: sensor callbacks in the default group, the inference timer
    # (JPEG pass + ORION pipeline, ~100 ms) in its own.  Not more: every extra
    # executor thread contends for the GIL with the inference thread.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
