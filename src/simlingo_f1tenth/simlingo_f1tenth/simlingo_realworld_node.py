#!/usr/bin/env python3
"""SimLingo inference on a real F1TENTH car (runs on the Jetson AGX Orin).

Subscriptions
  image_topic   sensor_msgs/CompressedImage | Image   front camera on the Orin
  pose_topic    nav_msgs/Odometry                      particle filter, map frame (/pf/pose/odom)
  speed_topic   nav_msgs/Odometry                      VESC odometry, twist.linear.x (/odom)
  route_topic   nav_msgs/Path                          global route, map frame (optional; see route_csv)

Publications
  plan_topic       autoware_planning_msgs/Trajectory  the prediction, MAP frame, real metres and
                                                      real m/s, anchored at the pose it was made from
  path_topic       nav_msgs/Path                      the same plan for RViz
  language_topic   std_msgs/String                    the model's text output
  markers          visualization_msgs/MarkerArray     plan line + the two target points

No control is produced here: trajectory_controller_node tracks the plan and talks
to the VESC.  Structure follows simlingo_ros/simlingo_node.py (payload built on
the executor thread, inference on a single worker thread that owns the CUDA
handles); the CARLA-specific parts -- bridge frame mirroring, CarlaRoute,
CarlaEgoVehicleControl, the in-node PID -- are gone.

World scale
  The lab track is a 1:10 replica of the CARLA scene.  Positions, route
  waypoints and speed are multiplied by ``world_scale`` before they reach the
  model, and the predicted waypoints and speed are divided by it on the way
  out.  With world_scale=10 the model sees the geometry it was trained on and
  the 7.5 m / 50 m target-point window becomes 0.75 m / 5 m on the floor.
"""

from __future__ import annotations

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
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from simlingo_f1tenth.frame_saver import FrameSaver
from simlingo_f1tenth.route_planner import (
    DEFAULT_MAX_DISTANCE, DEFAULT_MIN_DISTANCE, RoutePlanner, load_route_csv,
    model_to_map, thin_route, yaw_from_quaternion,
)
from simlingo_f1tenth.simlingo_model import (
    TRAIN_BOTTOM_CROP_FRAC, TRAIN_CAM_FOV_DEG, TRAIN_IMAGE_H, TRAIN_IMAGE_W, WAYPOINT_DT,
    SimLingoRunner, format_camera_frame, jpeg_roundtrip,
)


class _RosLogAdapter:
    """Lets SimLingoRunner log through the node."""

    def __init__(self, logger) -> None:
        self._l = logger

    def info(self, m):  self._l.info(str(m))
    def warn(self, m):  self._l.warn(str(m))
    def warning(self, m):  self._l.warn(str(m))
    def error(self, m): self._l.error(str(m))


class SimLingoRealWorldNode(Node):

    def __init__(self) -> None:
        super().__init__("simlingo_realworld_node")

        # ── model ────────────────────────────────────────────────────────────
        self.declare_parameter("simlingo_path",   "/benchmarking/simlingo")
        self.declare_parameter("checkpoint_path", "/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt")
        self.declare_parameter("fast_inference",  True)
        self.declare_parameter("use_cot",         False)   # thinking mode: commentary before the waypoints
        self.declare_parameter("inference_period_sec", 0.25)   # timer; effectively "as fast as the model goes"

        # ── topics ───────────────────────────────────────────────────────────
        self.declare_parameter("image_topic",    "/camera/front/image/compressed")
        self.declare_parameter("image_compressed", False)      # auto if topic ends in /compressed
        self.declare_parameter("pose_topic",     "/pf/pose/odom")
        self.declare_parameter("speed_topic",    "/odom")
        self.declare_parameter("route_topic",    "/global_path")
        self.declare_parameter("plan_topic",     "/simlingo/plan")
        self.declare_parameter("path_topic",     "/simlingo/plan_path")
        self.declare_parameter("language_topic", "/simlingo/language_output")
        self.declare_parameter("marker_topic",   "/simlingo/markers")

        # ── geometry / scale ─────────────────────────────────────────────────
        self.declare_parameter("world_scale",    10.0)
        # Model m/s per real m/s, for the speed fed in and the speed predicted.  0 = world_scale.
        # Smaller than world_scale makes the car faster than the 1:10 geometry implies
        # (5: a predicted 2 m/s is commanded as 0.4 m/s); distances stay on world_scale.
        self.declare_parameter("speed_world_scale", 0.0)
        # /pf/pose/odom is the laser pose; base_link is 0.27 m behind it (bringup static TF).
        self.declare_parameter("pose_offset_x",  -0.27)
        self.declare_parameter("route_csv",      "")           # x,y[,v] rows, map frame, real metres
        self.declare_parameter("route_loop",     False)        # closed circuit vs point-to-point
        self.declare_parameter("route_min_spacing_m", 0.0)     # REAL metres; 0 keeps every waypoint
        self.declare_parameter("target_min_distance", DEFAULT_MIN_DISTANCE)   # model metres
        self.declare_parameter("target_max_distance", DEFAULT_MAX_DISTANCE)   # model metres

        # ── image formatting ─────────────────────────────────────────────────
        self.declare_parameter("image_width",      TRAIN_IMAGE_W)
        self.declare_parameter("image_height",     TRAIN_IMAGE_H)
        self.declare_parameter("bottom_crop_frac", TRAIN_BOTTOM_CROP_FRAC)
        self.declare_parameter("camera_fov_deg",   TRAIN_CAM_FOV_DEG)   # intrinsics handed to the model
        self.declare_parameter("jpeg_roundtrip_raw", True)   # raw Image input: reproduce training JPEG

        # ── misc ─────────────────────────────────────────────────────────────
        self.declare_parameter("stale_image_sec", 1.0)      # skip inference on frames older than this
        self.declare_parameter("log_every",       1)
        self.declare_parameter("log_waypoints",   True)     # print the 20 predicted waypoints per plan
        # Directory for one JPEG per plan: input frame + predicted trajectory + language. "" = off.
        self.declare_parameter("save_frames_dir", "")
        self.declare_parameter("save_frames_every", 1)      # save every Nth plan

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        simlingo_path = str(p("simlingo_path"))
        for extra in (simlingo_path, simlingo_path + "/team_code"):
            if extra and extra not in sys.path:
                sys.path.insert(0, extra)

        self._scale         = float(p("world_scale"))
        self._speed_scale   = float(p("speed_world_scale")) or self._scale
        self._pose_offset_x = float(p("pose_offset_x"))
        self._img_w         = int(p("image_width"))
        self._img_h         = int(p("image_height"))
        self._crop_frac     = float(p("bottom_crop_frac"))
        self._jpeg_raw      = bool(p("jpeg_roundtrip_raw"))
        self._stale_image   = float(p("stale_image_sec"))
        self._route_spacing = float(p("route_min_spacing_m"))
        self._log_waypoints = bool(p("log_waypoints"))
        if self._scale <= 0.0 or self._speed_scale <= 0.0:
            raise ValueError("world_scale must be > 0 and speed_world_scale >= 0")

        # ── state ────────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None        # formatted, cropped, RGB
        self._latest_img_wall: Optional[float] = None
        self._img_count = 0
        self._pose: Optional[np.ndarray] = None               # [x, y, yaw] base_link, map, REAL metres
        self._speed = 0.0                                     # REAL m/s
        self._planner = RoutePlanner(
            min_distance=float(p("target_min_distance")),
            max_distance=float(p("target_max_distance")),
            loop=bool(p("route_loop")),
        )
        self._durations: List[float] = []
        self._plan_no = 0
        self._saver: Optional[FrameSaver] = None
        if str(p("save_frames_dir")):
            self._saver = FrameSaver(str(p("save_frames_dir")), int(p("save_frames_every")),
                                     logger=self.get_logger(), world_scale=self._scale)
            self.get_logger().info(f"saving plan frames to {p('save_frames_dir')}")

        # ── publishers ───────────────────────────────────────────────────────
        self._plan_pub   = self.create_publisher(Trajectory,  str(p("plan_topic")), 10)
        self._path_pub   = self.create_publisher(Path,        str(p("path_topic")), 10)
        self._lang_pub   = self.create_publisher(String,      str(p("language_topic")), 10)
        self._marker_pub = self.create_publisher(MarkerArray, str(p("marker_topic")), 10)

        # ── subscribers ──────────────────────────────────────────────────────
        img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=2)
        image_topic = str(p("image_topic"))
        compressed = bool(p("image_compressed")) or image_topic.endswith("/compressed")
        if compressed:
            self.create_subscription(CompressedImage, image_topic, self._image_cb_compressed, img_qos)
        else:
            self.create_subscription(Image, image_topic, self._image_cb_raw, img_qos)
        self.get_logger().info(f"image ({'jpeg' if compressed else 'raw'}): {image_topic}")

        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, str(p("pose_topic")),  self._pose_cb,  be)
        self.create_subscription(Odometry, str(p("speed_topic")), self._speed_cb, be)

        route_csv = str(p("route_csv"))
        if route_csv:
            self._set_route(load_route_csv(route_csv), f"csv {route_csv}")
        # A Path topic replaces the CSV route if/when it arrives (latched publisher on the car).
        route_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                               depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, str(p("route_topic")), self._route_cb, route_qos)
        self.get_logger().info(f"route: topic {p('route_topic')}" + (f" (initial: {route_csv})" if route_csv else ""))

        # ── model, on its own thread ─────────────────────────────────────────
        self._runner = SimLingoRunner(
            checkpoint_path=str(p("checkpoint_path")), simlingo_path=simlingo_path,
            fast_inference=bool(p("fast_inference")), use_cot=bool(p("use_cot")),
            camera_fov_deg=float(p("camera_fov_deg")),
            logger=_RosLogAdapter(self.get_logger()),
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active: Optional[Future] = None
        self.get_logger().info("Loading SimLingo (60-90 s on Orin)...")
        self._executor.submit(self._runner.load).result()

        self._infer_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(float(p("inference_period_sec")), self._timer_cb, callback_group=self._infer_group)
        self.get_logger().info(
            f"ready: world_scale={self._scale:g}, speed scale {self._speed_scale:g}, target window "
            f"{self._planner.min_distance:g}-{self._planner.max_distance:g} model m "
            f"({self._planner.min_distance / self._scale:.2f}-{self._planner.max_distance / self._scale:.2f} real m)")

    # ── sensor callbacks ─────────────────────────────────────────────────────

    def _store_frame(self, rgb: np.ndarray) -> None:
        formatted = format_camera_frame(rgb, self._img_w, self._img_h, self._crop_frac)
        with self._lock:
            self._latest_rgb = formatted
            self._latest_img_wall = time.time()
            self._img_count += 1

    def _image_cb_compressed(self, msg: CompressedImage) -> None:
        bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn("undecodable compressed frame", throttle_duration_sec=2.0)
            return
        self._store_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _image_cb_raw(self, msg: Image) -> None:
        enc = msg.encoding.lower()
        ch = 4 if enc in ("rgba8", "bgra8") else 3
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, ch)[:, :, :3]
        rgb = img if enc.startswith("rgb") else cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)
        if self._jpeg_raw:
            rgb = jpeg_roundtrip(rgb)
        self._store_frame(rgb)

    def _pose_cb(self, msg: Odometry) -> None:
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        x = msg.pose.pose.position.x + self._pose_offset_x * np.cos(yaw)
        y = msg.pose.pose.position.y + self._pose_offset_x * np.sin(yaw)
        with self._lock:
            self._pose = np.array([x, y, yaw], dtype=np.float64)

    def _speed_cb(self, msg: Odometry) -> None:
        self._speed = float(msg.twist.twist.linear.x)

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
        length = float(np.linalg.norm(np.diff(wps_real, axis=0), axis=1).sum())
        self.get_logger().info(f"route set from {source}: {len(wps_real)} waypoints, {length:.1f} m real "
                               f"({length * self._scale:.0f} m model), loop={self._planner.loop}")

    # ── inference pipeline ───────────────────────────────────────────────────

    def _timer_cb(self) -> None:
        try:
            if self._active is not None and not self._active.done():
                return
            payload = self._build_payload()
            if payload is None:
                return
            self._active = self._executor.submit(self._run_inference, payload)
            self._active.add_done_callback(self._on_done)
        except Exception as exc:
            self.get_logger().error(f"_timer_cb failed: {type(exc).__name__}: {exc}", throttle_duration_sec=5.0)

    def _build_payload(self) -> Optional[dict]:
        with self._lock:
            rgb, img_wall = self._latest_rgb, self._latest_img_wall
            pose = None if self._pose is None else self._pose.copy()
            has_route = self._planner.has_route
        if rgb is None:
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

        ego_pos_model = pose[:2] * self._scale
        ego_yaw = float(pose[2])
        speed_model = self._speed * self._speed_scale
        with self._lock:
            tp0, tp1 = self._planner.target_points(ego_pos_model, ego_yaw)
            route_idx, route_done = self._planner.index, self._planner.finished()
        if route_done:
            self.get_logger().info("route finished", throttle_duration_sec=5.0)

        processed, num_patches = self._runner.preprocess(rgb)
        return {
            "rgb": rgb, "processed": processed, "num_patches": num_patches,
            "speed_model": speed_model, "speed_real": self._speed,
            "target_points": np.array([tp0, tp1], dtype=np.float32),
            "pose_real": pose, "ego_pos_model": ego_pos_model, "ego_yaw": ego_yaw,
            "route_idx": route_idx, "img_age": age, "stamp": self.get_clock().now().to_msg(),
            "t_built": time.perf_counter(),
        }

    def _run_inference(self, payload: dict) -> dict:
        t0 = time.perf_counter()
        res = self._runner.infer(payload["processed"], payload["num_patches"],
                                 payload["speed_model"], payload["target_points"])

        # Model ego frame (model metres) -> map frame (real metres), anchored at
        # the pose the frame was captured from.
        pose = payload["pose_real"]
        route_map = model_to_map(res.route / self._scale, pose[:2], pose[2])
        desired_speed_real = res.desired_speed / self._speed_scale
        stamp = payload["stamp"]
        self._plan_no += 1

        if self._saver is not None:
            tp = payload["target_points"]
            self._saver.submit(self._plan_no, payload["rgb"], res.route, res.speed_wps, tp, res.language, [
                f"plan {self._plan_no}  {time.strftime('%H:%M:%S')}  route idx {payload['route_idx']}  "
                f"model {res.model_ms:.0f} ms  frame age {payload['img_age']:.2f} s",
                f"speed {payload['speed_real']:.2f} m/s real ({payload['speed_model']:.1f} model)  ->  "
                f"desired {desired_speed_real:.2f} m/s real ({res.desired_speed:.2f} model)  "
                f"tp0 [{tp[0][0]:+.1f},{tp[0][1]:+.1f}] tp1 [{tp[1][0]:+.1f},{tp[1][1]:+.1f}] model m",
            ])

        self._plan_pub.publish(self._to_trajectory(route_map, desired_speed_real, stamp))
        self._path_pub.publish(self._to_path(route_map, stamp))
        tp_map = model_to_map(payload["target_points"].astype(np.float64) / self._scale, pose[:2], pose[2])
        self._marker_pub.publish(self._to_markers(route_map, tp_map, stamp))
        if res.language:
            self._lang_pub.publish(String(data=res.language))

        tp = payload["target_points"]
        horizon = float(np.linalg.norm(np.diff(route_map, axis=0), axis=1).sum())
        self.get_logger().info(
            f"[SIMLINGO] tp0=[{tp[0][0]:+.1f},{tp[0][1]:+.1f}] tp1=[{tp[1][0]:+.1f},{tp[1][1]:+.1f}] model m "
            f"(route idx {payload['route_idx']}) | speed in {payload['speed_model']:.1f} model m/s | "
            f"desired {res.desired_speed:.2f} model -> {desired_speed_real:.2f} real m/s | "
            f"horizon {horizon:.2f} m real | model {res.model_ms:.0f} ms | frame age {payload['img_age']:.2f}s")
        if self._log_waypoints:
            # ROS ego frame (x forward, y LEFT), real metres: what the controller tracks.
            wps = " ".join(f"({x:.2f},{-y:+.2f})" for x, y in res.route / self._scale)
            self.get_logger().info(f"[SIMLINGO] waypoints ego x,y real m: {wps}")
        if res.language:
            self.get_logger().info(f"[SIMLINGO] language: {res.language}")
        return {"total_ms": (time.perf_counter() - t0) * 1000.0, "model_ms": res.model_ms}

    def _on_done(self, fut: Future) -> None:
        try:
            m = fut.result()
        except Exception as exc:
            self.get_logger().error(f"inference failed: {type(exc).__name__}: {exc}")
            return
        self._durations.append(m["total_ms"])
        if len(self._durations) % 20 == 0:
            d = sorted(self._durations[-20:])
            self.get_logger().info(f"[PROF] last 20: mean {sum(d)/len(d):.0f} ms, min {d[0]:.0f}, max {d[-1]:.0f}")

    # ── message building ─────────────────────────────────────────────────────

    @staticmethod
    def _headings(route_map: np.ndarray) -> np.ndarray:
        d = np.diff(route_map, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        return np.append(yaw, yaw[-1]) if len(yaw) else np.zeros(len(route_map))

    def _to_trajectory(self, route_map: np.ndarray, speed: float, stamp) -> Trajectory:
        traj = Trajectory()
        traj.header.stamp = stamp
        traj.header.frame_id = "map"
        for i, (pt, yaw) in enumerate(zip(route_map, self._headings(route_map))):
            tp = TrajectoryPoint()
            tp.pose.position.x, tp.pose.position.y = float(pt[0]), float(pt[1])
            tp.pose.orientation.z = float(np.sin(yaw / 2.0))
            tp.pose.orientation.w = float(np.cos(yaw / 2.0))
            tp.longitudinal_velocity_mps = float(speed)
            sec = i * WAYPOINT_DT
            tp.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
            traj.points.append(tp)
        return traj

    def _to_path(self, route_map: np.ndarray, stamp) -> Path:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "map"
        for pt, yaw in zip(route_map, self._headings(route_map)):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(pt[0]), float(pt[1])
            ps.pose.orientation.z = float(np.sin(yaw / 2.0))
            ps.pose.orientation.w = float(np.cos(yaw / 2.0))
            path.poses.append(ps)
        return path

    def _to_markers(self, route_map: np.ndarray, tp_map: np.ndarray, stamp) -> MarkerArray:
        arr = MarkerArray()
        line = Marker()
        line.header.stamp, line.header.frame_id = stamp, "map"
        line.ns, line.id, line.type, line.action = "simlingo_plan", 0, Marker.LINE_STRIP, Marker.ADD
        line.scale.x = 0.03
        line.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)
        line.pose.orientation.w = 1.0
        line.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in route_map]
        arr.markers.append(line)
        for i, (x, y) in enumerate(tp_map):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, "map"
            m.ns, m.id, m.type, m.action = "simlingo_target", i + 1, Marker.SPHERE, Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.12
            m.color = ColorRGBA(r=1.0, g=0.3 if i else 0.0, b=0.0, a=1.0)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.05
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
    node = SimLingoRealWorldNode()
    # Two threads: sensor callbacks in the default group, the inference timer
    # (JPEG decode + tiling, tens of ms) in its own.  Not more: rclpy's executor
    # polls in Python and every extra thread contends for the GIL with the
    # inference thread (measured 3x slower with the default thread count).
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
