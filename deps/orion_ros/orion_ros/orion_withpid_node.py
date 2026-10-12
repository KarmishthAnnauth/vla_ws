#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that streams CARLA sensor topics into the ORION VLA model and
drives a self-contained closed loop: the model's predicted trajectory is fed to
a Bench2Drive PID controller in-node, which publishes carla_msgs/CarlaEgoVehicleControl
(steer/throttle/brake) directly on /carla/hero/vehicle_control_cmd — exactly like
the leaderboard agent's carla.VehicleControl, so no carla_ackermann_control node is
needed. This is the PID variant of orion_node.py (which instead publishes an
Autoware Trajectory for a downstream Stanley controller).

Design goal: the payload handed to the ORION model must be prepared in EXACTLY
the same way as the reference closed-loop agent
(``Orion/team_code/orion_b2d_agent.py::OrionAgent.run_step``). To guarantee that,
this node reuses ORION's own code rather than reimplementing the preprocessing:

  raw 6-cam images + odometry
      -> build the same ``results`` dict the agent builds
      -> ``Compose(cfg.inference_only_pipeline)``   (resize/crop, normalize, pad,
                                                      VQA tokenization)
      -> ``mmcv.parallel.collate``  (a.k.a. mm_collate_to_batch_form)
      -> ``model(batch, return_loss=False)``
      -> output[0]['pts_bbox']['ego_fut_preds']  -> (6, 2) ego trajectory

Only the *raw input sources* differ from the CARLA agent. They are the same
carla-simulator/ros-bridge topics consumed by the SimLingo reference node:
  * 6x ``sensor_msgs/Image`` (raw rgba8/bgra8), one per ORION camera in
    ORION_CAMERA_ORDER -> each decoded to BGR into its OWN input slot, matching
    OrionAgent.tick(). See orion_ros.camera_input.
  * ``std_msgs/Float32`` speed                          -> can_bus[7]
  * ``nav_msgs/Odometry``                               -> ego_pose / heading
  * ``sensor_msgs/Imu``                                 -> can_bus accel + angular
  * ``carla_msgs/CarlaRoute`` (latched)                 -> driving command

NOTE (deferred, to reconcile together later):
  * Topic names are all parameters with placeholder defaults.
  * The odometry->can_bus coordinate conventions replicate the agent's *formulas*
    (sign flips, ego_theta = -yaw + pi/2) under the assumption that the odometry
    frame matches CARLA's world frame and odom-yaw matches the CARLA compass.
    These sign/frame choices are isolated in ``_odom_to_canbus_ego_pose`` and
    marked with TODOs so we can adjust them once the real CARLA bridge topics and
    frames are known.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaRoute
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from std_msgs.msg import ColorRGBA, Float32, String

from orion_ros.camera_input import (
    DEFAULT_CAMERA_TOPICS,
    ORION_CAMERA_ORDER,
    REFERENCE_CAMERA,
    MultiCameraBuffer,
    decode_image,
)
from orion_ros import orion_speedups
from visualization_msgs.msg import Marker, MarkerArray

try:
    from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint

    _HAS_AUTOWARE_TRAJ = True
except ImportError:
    _HAS_AUTOWARE_TRAJ = False


# ORION_CAMERA_ORDER and the raw-Image decode live in camera_input so this node
# and orion_withpid_node share one implementation. The hardcoded calibration
# matrices below are indexed by those names and assume the CARLA mounting poses
# declared in OrionAgent.sensors().

# ORION predicts 6 future waypoints at 2 Hz -> 0.5 s spacing, 3 s horizon.
ORION_TRAJ_DT = 0.5

# OrionHead.pre_update_memory keeps its rolling temporal memory only while
#     abs(sample_time) < 2.0  AND  scene_token is unchanged
# where sample_time is the delta between consecutive `timestamp` values. On a
# failure it calls memory_refresh(mem, 0), i.e. mem * 0 — the entire temporal
# memory is zeroed. Nothing in the model logs this, so the node does.
ORION_MEMORY_MAX_DT = 2.0
# OrionAgent uses `results['timestamp'] = self.step / 20`: a synthetic 20 Hz
# clock counted in frames consumed, NOT wall time. Under CARLA's synchronous
# mode those coincide; in async real-time they do not.
ORION_AGENT_HZ = 20.0
ORION_FUT_TS = 6

# OrionAgent.run_step: `if tick_data['speed'] > 5: throttle_traj = 0`.
ORION_SPEED_CAP_MPS = 5.0

# Where the reference agent thinks the ego IS.
#
# orion_b2d_agent.py localises from its GNSS sensor, which sensors() mounts at
# x=-1.4 in the vehicle frame, and CARLA reports the geolocation OF THE SENSOR --
# so the agent's `pos` is 1.4 m BEHIND the vehicle origin. Every ego-referenced
# quantity is built from it: can_bus[0:2], ego2world -> ego_pose/lidar2global,
# the RoutePlanner step that picks the driving command, and command_near_xy ->
# local_command_xy.
#
# /carla/hero/odometry is the vehicle origin, so feeding it straight in put all
# four of those 1.4 m ahead of the reference. That is not a wash: ORION's
# lidar2ego is fixed, so shifting the ego frame moves where the model believes
# its own cameras are, and 1.4 m against control_pid's aim_dist of 3.5 m is 40 %
# of the lookahead.
#
# Set the `gnss_mount_offset_x` parameter to 0.0 to get the old (vehicle-origin)
# behaviour back.
AGENT_GNSS_MOUNT_X = -1.4

# CARLA max steering angle (rad); converts the PID's normalised [-1,1] steer to a
# physical angle for AckermannDrive (same as orion_lite_rosnode).
MAX_STEER_ANGLE = 0.7

# Bench2Drive RoutePlanner thresholds (agent: RoutePlanner(4.0, 50.0)).
ROUTE_MIN_DIST = 4.0
ROUTE_MAX_DIST = 50.0


# ---------------------------------------------------------------------------
# Verbatim copies from Orion/team_code/orion_b2d_agent.py so the payload is
# built identically. DO NOT edit these values — they are part of the model's
# expected input contract.
# ---------------------------------------------------------------------------
def command2hot(command: int, max_dim: int = 6) -> np.ndarray:
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot


def command2nohot(command: int, max_dim: int = 6) -> int:
    if command < 0:
        command = 4
    command -= 1
    return command


def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


# Mirrors orion_b2d_agent.custom_wrap_fp16_model: enable fp16 on all fp16-capable
# modules but force the perception heads back to fp32. Called every inference to
# match the agent exactly.
_CUSTOM_FP16 = dict(map_head=False, pts_bbox_head=False)


def custom_wrap_fp16_model(model) -> None:
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True
    for module_name, v in _CUSTOM_FP16.items():
        if module_name in model._modules:
            model._modules[module_name].fp16_enabled = v


# Hardcoded LiDAR/camera calibration, copied verbatim from the agent's setup().
LIDAR2IMG = {
    "CAM_FRONT": np.array([[1.14251841e03, 8.00000000e02, 0.00000000e00, -9.52000000e02],
                           [0.00000000e00, 4.50000000e02, -1.14251841e03, -8.09704417e02],
                           [0.00000000e00, 1.00000000e00, 0.00000000e00, -1.19000000e00],
                           [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_FRONT_LEFT": np.array([[6.03961325e-14, 1.39475744e03, 0.00000000e00, -9.20539908e02],
                                [-3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
                                [-8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_FRONT_RIGHT": np.array([[1.31064327e03, -4.77035138e02, 0.00000000e00, -4.06010608e02],
                                 [3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
                                 [8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
                                 [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_BACK": np.array([[-5.60166031e02, -8.00000000e02, 0.00000000e00, -1.28800000e03],
                          [5.51091060e-14, -4.50000000e02, -5.60166031e02, -8.58939847e02],
                          [1.22464680e-16, -1.00000000e00, 0.00000000e00, -1.61000000e00],
                          [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_BACK_LEFT": np.array([[-1.14251841e03, 8.00000000e02, 0.00000000e00, -6.84385123e02],
                               [-4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
                               [-9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
                               [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_BACK_RIGHT": np.array([[3.60989788e02, -1.34723223e03, 0.00000000e00, -1.04238127e02],
                                [4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
                                [9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
}
LIDAR2CAM = {
    "CAM_FRONT": np.array([[1., 0., 0., 0.], [0., 0., -1., -0.24],
                           [0., 1., 0., -1.19], [0., 0., 0., 1.]]),
    "CAM_FRONT_LEFT": np.array([[0.57357644, 0.81915204, 0., -0.22517331],
                                [0., 0., -1., -0.24],
                                [-0.81915204, 0.57357644, 0., -0.82909407],
                                [0., 0., 0., 1.]]),
    "CAM_FRONT_RIGHT": np.array([[0.57357644, -0.81915204, 0., 0.22517331],
                                 [0., 0., -1., -0.24],
                                 [0.81915204, 0.57357644, 0., -0.82909407],
                                 [0., 0., 0., 1.]]),
    "CAM_BACK": np.array([[-1., 0., 0., 0.], [0., 0., -1., -0.24],
                          [0., -1., 0., -1.61], [0., 0., 0., 1.]]),
    "CAM_BACK_LEFT": np.array([[-0.34202014, 0.93969262, 0., -0.25388956],
                               [0., 0., -1., -0.24],
                               [-0.93969262, -0.34202014, 0., -0.49288953],
                               [0., 0., 0., 1.]]),
    "CAM_BACK_RIGHT": np.array([[-0.34202014, -0.93969262, 0., 0.25388956],
                                [0., 0., -1., -0.24],
                                [0.93969262, -0.34202014, 0., -0.49288953],
                                [0., 0., 0., 1.]]),
}
LIDAR2EGO = np.array([[0., 1., 0., -0.39],
                      [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84],
                      [0., 0., 0., 1.]])


class OrionWithPidRosNode(Node):
    """ORION inference on live CARLA streams, with an in-node PID controller that
    publishes CarlaEgoVehicleControl directly (self-contained closed loop)."""

    def __init__(self) -> None:
        super().__init__("orion_withpid_node")

        # --- ORION repo / model parameters ---
        # Path to the cloned Orion repository (so its custom `mmcv` package and the
        # model/pipeline registries can be imported). Empty -> assume already on PYTHONPATH.
        self.declare_parameter("orion_repo_path", "")
        # ORION config that defines `model` + `inference_only_pipeline`.
        self.declare_parameter(
            "orion_config_path",
            "adzoo/orion/configs/orion_stage3_agent.py",
        )
        self.declare_parameter("orion_checkpoint_path", "ckpts/Orion.pth")
        # Inference precision: "fp16" (faster on Orin, ~near-identical accuracy) or
        # "fp32". This flips the precision flags on the loaded config so the agent
        # config's inference_only_pipeline is preserved and the GT-dependent metric
        # branch stays disabled. "" leaves the config's own flags untouched.
        self.declare_parameter("precision", "fp16")
        # What to feed ORION as `timestamp`, which drives its temporal-memory
        # retention (see ORION_MEMORY_MAX_DT):
        #   "sensor" (default) — the real sensor stamp. Honest. If frames are
        #       more than 2 s apart, as in async real-time with slow inference,
        #       ORION's memory is wiped every frame and it degenerates to
        #       single-frame inference. Latency benchmarks stay valid; driving
        #       quality is NOT comparable to the paper.
        #   "agent" — frame_idx / 20, exactly the formula OrionAgent uses. Keeps
        #       the memory alive so the model runs in its designed streaming
        #       regime, but tells it 0.05 s passed when seconds really did, so
        #       its velocity/motion reasoning is fed a false dt.
        # Neither is "correct" when inference is slower than the scene: use the
        # pair to measure what the temporal memory is actually worth on Orin.
        self.declare_parameter("timestamp_mode", "sensor")

        # --- Topic parameters (CARLA ros-bridge, ego role_name = "hero") ---
        # Six raw sensor_msgs/Image topics in ORION_CAMERA_ORDER — the surround
        # view OrionAgent.tick() actually feeds the model. Each camera's pixels go
        # into its own slot alongside its own calibration (see _build_batch).
        self.declare_parameter("camera_topics", DEFAULT_CAMERA_TOPICS)
        # Deprecated single-camera parameter, kept so older launch files and
        # command lines keep working. When set it wins and forces replicate mode,
        # which is NOT faithful to the agent — see camera_input.MultiCameraBuffer.
        self.declare_parameter("front_camera_topic", "")
        # The agent's six images come from one simulator tick; here they are six
        # independent DDS streams. Warn when they drift further apart than this.
        self.declare_parameter("camera_sync_tolerance_sec", 0.1)
        self.declare_parameter("speed_topic", "/carla/hero/speed")
        self.declare_parameter("odometry_topic", "/carla/hero/odometry")
        self.declare_parameter("imu_topic", "/carla/hero/imu")
        self.declare_parameter("route_topic", "/carla/hero/global_plan")
        self.declare_parameter("trajectory_topic", "/orion/predicted_trajectory")
        self.declare_parameter("cot_topic", "/orion/cot")

        # --- Inference behaviour ---
        # Timer interval at which we CHECK whether to dispatch. The model runs as
        # fast as it can (one in flight at a time); each run always uses the most
        # recent buffered frames. Keep this small so the gap between a finished
        # run and the next dispatch is tiny.
        self.declare_parameter("inference_period_sec", 0.05)
        # Only dispatch when a NEW camera frame has arrived since the last run, so
        # we never reprocess stale data (and never feed the temporal memory a
        # duplicated ego pose). Set False to benchmark raw throughput on whatever
        # is latest regardless of novelty.
        self.declare_parameter("require_new_frame", True)
        # CoT/reasoning text is only produced by reasoning configs (e.g.
        # orion_stage3_cot.py). The default orion_stage3_agent.py is planning-only
        # (fast) and emits no CoT. This flag just controls publishing.
        self.declare_parameter("publish_cot", True)
        # Driving command fed to the model (ORION's command2hot encoding):
        #   1=left, 2=right, 3=straight, 4=follow-lane, 5=lane-change-L, 6=lane-change-R.
        # No route planner here yet -> default to follow-lane. Wire a topic later.
        self.declare_parameter("driving_command", 4)
        # The agent re-encodes each CARLA frame to JPEG quality 20 before the
        # pipeline. Replicate that lossy step for payload fidelity. <=0 disables.
        self.declare_parameter("replicate_jpeg_quality", 20)
        # --- Inference speedups (orion_speedups.py; all post-build, ORION
        # sources untouched). Measured with tools/bench_orion.py.
        # Fold the LoRA adapters into the LLM weights (peft never merges).
        self.declare_parameter("merge_lora", True)
        # LLaMA prefill attention through flash_attn instead of transformers-4.31
        # eager attention (fp16 only; falls back automatically otherwise).
        self.declare_parameter("llm_flash_attn", True)
        # Comma-separated torch.compile targets ("vit", "llm", "heads"); "" = eager.
        # The warm-up forward compiles them (~3.5 min on Orin; inductor caches
        # to TORCHINDUCTOR_CACHE_DIR). Measured with tools/bench_orion.py:
        # forward 1730 -> 1010 ms, ViT 880 -> 430 ms, trajectory delta < 1 cm.
        self.declare_parameter("compile_targets", "heads,llm,vit")
        # "default" (inductor fusion only). "reduce-overhead" adds CUDA graphs:
        # no steady-state gain (the forward is GPU-bound, not launch-bound) and
        # it re-records on the first real frame, costing minutes more startup.
        self.declare_parameter("compile_mode", "default")
        # Rewrite the EVA-ViT block glue (fused qkv / w1w2 GEMMs, native SDPA):
        # same weights, ViT 434 -> 420 ms compiled and half the compile time.
        self.declare_parameter("vit_glue", True)
        # TensorRT plan for the EVA-ViT backbone (tools/export_vit_onnx.py +
        # trtexec). "" = PyTorch. When set, "vit" is dropped from compile_targets.
        self.declare_parameter("backbone_engine", "")
        # Per-stage CUDA-event timing appended to the inference log line.
        self.declare_parameter("profile_stages", False)
        # Threads for the six per-camera JPEG re-encodes and for the per-image
        # loops inside the ORION pipeline transforms (1 = sequential).
        self.declare_parameter("decode_workers", 6)
        # Prepare frame N+1 (decode + pipeline + collate, ~120 ms of CPU) on a
        # second worker while frame N's forward runs on the GPU. The prep is
        # started `pipeline_margin_ms` + <avg prep> before the forward is
        # expected to finish, so the next forward starts with no gap while its
        # input is as fresh as in the sequential path.
        self.declare_parameter("pipeline_prep", True)
        self.declare_parameter("pipeline_margin_ms", 40.0)
        # Exact head speedups (orion_speedups.py): run the map head with its 300
        # one-to-one lane queries only (the 1500 one-to-many queries are a
        # training trick masked off from everything that is used at inference),
        # and run the map head on a side CUDA stream concurrently with the
        # detection head.
        self.declare_parameter("map_head_slice", True)
        # "Lite vision" (launch arg vision_lite:=true sets both; off = base model):
        # ViT input size (640 = trained; 512 -> 1024 instead of 1600 tokens per
        # view; must be a multiple of 256) and rear-camera refresh period
        # (2 = the three rear views go through the ViT every other frame and
        # their previous features are reused in between, ~1 s stale).
        # Both change what the model computes: gate them on a route score.
        self.declare_parameter("vit_input_size", 640)
        self.declare_parameter("rear_view_refresh_every", 1)
        # LLM GEMMs: store the K=11008 down projection transposed (cuBLAS runs
        # it 30 % faster, same numbers), and optionally W8A8 dynamic INT8 for
        # all decoder projections via torch._int_mm (1.6-2.9x faster GEMMs;
        # changes numerics -- validate on a route score before relying on it).
        self.declare_parameter("llm_down_proj_transpose", True)
        self.declare_parameter("llm_int8", False)
        # SmoothQuant recipe for llm_int8 (tools/int8_sweep.py): calibration
        # stats file, alpha, and decoder layers kept in fp16.
        self.declare_parameter("llm_int8_stats", "")
        self.declare_parameter("llm_int8_alpha", 0.8)
        self.declare_parameter("llm_int8_skip_layers", [0, 1, 30, 31])
        # Sweep (tools/int8_sweep.py, 2026-09-14): MLP-only at alpha 0.8 with
        # those 4 layers fp16 = 4.4 % waypoint-feature error / 3 cm trajectory
        # shift for ~95 ms; all projections = 7 % / 5 cm for ~110 ms. fp16
        # changes stay < 1 cm, so this needs a closed-loop route score first.
        self.declare_parameter("llm_int8_targets", "gate_proj,up_proj,down_proj")
        self.declare_parameter("overlap_heads", False)   # measured: no gain, heads are CPU-launch bound
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("min_trajectory_speed_mps", 0.0)
        # --- Closed-loop control cadence -------------------------------------
        # The PID runs on its own timer rather than once per forward pass. A
        # forward pass takes ~2 s on this Orin, and publishing one control per
        # pass means holding a single steer/throttle/brake triple for those 2 s:
        # on 2026-09-14 route 2 was driven by seven commands in total and left
        # the road on the first left turn. The reference agent
        # (OrionAgent.run_step) runs control_pid every simulator tick at 20 Hz,
        # so 20 Hz here is the faithful cadence, not a new behaviour -- it also
        # puts PIDController's 40-sample windows back on the 2 s of history they
        # were tuned for, instead of the 80 s that 0.49 Hz gives them.
        self.declare_parameter("control_hz", 20.0)
        # How long to wait for a NEW plan before stopping the car.
        #
        # Measured from when the last plan was installed, NOT from the age of the
        # plan itself. Those are different quantities and conflating them is a
        # mistake worth not repeating: a plan is already ~2 s old when it is
        # installed (that is how long the forward pass took) and is then tracked
        # for the ~2 s the next pass takes, so its age peaks near 2x the
        # inference time plus one camera period -- ~4.7 s here. A 4 s "max age"
        # therefore fired at the end of every single cycle and stamped
        # brake=1.0 on a perfectly healthy run (2026-09-14).
        #
        # Time since the last arrival has no such coupling: in steady state it
        # never exceeds one inference, so 6 s means roughly three missed passes
        # and only ever fires when the model really has stopped answering.
        self.declare_parameter("plan_stall_timeout_sec", 6.0)
        self.declare_parameter("brake_when_stale", True)
        # --- Per-tick control trace (PID tuning) ------------------------------
        # Off by default, and CSV rather than get_logger(): at control_hz the
        # logger would emit 20 lines/s and bury orin.log.
        #
        # The column that motivates this is `swap`. The aim point is recomputed
        # every tick from the SAME waypoints until a new plan lands, so the error
        # signal steps at each plan swap -- and PID.step differentiates that step
        # directly (`derivative = window[-1] - window[-2]`, a raw sample
        # difference with no dt). A D-term kick therefore shows up as a steer
        # discontinuity on exactly the rows where swap=1. Faster inference does
        # not change the gains, but it does change how OFTEN those steps arrive
        # (~16 ticks/plan at 0.8 s/pass, ~40 at 2 s), which is the one place
        # inference time couples into closed-loop steering.
        #
        # Note the braking path (_stop_the_car) returns before control_pid runs,
        # so a stalled plan leaves a gap in `tick` rather than a row.
        # Longitudinal offset, in the vehicle frame, of the point the reference
        # agent localises from (its GNSS mount). 0.0 = the vehicle origin.
        self.declare_parameter("gnss_mount_offset_x", AGENT_GNSS_MOUNT_X)
        self.declare_parameter("control_trace", False)
        # "" = auto: /benchmarking/log/control_trace_<YYYYmmdd_HHMMSS>.csv, one
        # file per launch so successive tuning runs never overwrite each other.
        self.declare_parameter("control_trace_path", "")

        self._device = torch.device("cuda")

        front_topic = str(self.get_parameter("front_camera_topic").value or "")
        if front_topic:
            self._camera_topics = [front_topic]
        else:
            self._camera_topics = [
                str(t) for t in (self.get_parameter("camera_topics").value or []) if str(t)
            ]
        if not self._camera_topics:
            raise ValueError(
                "camera_topics must list 6 raw sensor_msgs/Image topics in ORION order"
            )
        self._camera_sync_tol = float(
            self.get_parameter("camera_sync_tolerance_sec").value
        )
        self._timestamp_mode = str(self.get_parameter("timestamp_mode").value or "sensor")
        if self._timestamp_mode not in ("sensor", "agent"):
            raise ValueError("timestamp_mode must be 'sensor' or 'agent'")

        self._inference_period = float(self.get_parameter("inference_period_sec").value)
        self._require_new_frame = bool(self.get_parameter("require_new_frame").value)
        self._publish_cot = bool(self.get_parameter("publish_cot").value)
        self._driving_command = int(self.get_parameter("driving_command").value)
        self._jpeg_quality = int(self.get_parameter("replicate_jpeg_quality").value)
        self._merge_lora = bool(self.get_parameter("merge_lora").value)
        self._llm_flash_attn = bool(self.get_parameter("llm_flash_attn").value)
        self._compile_targets = [t.strip() for t in
                                 str(self.get_parameter("compile_targets").value or "").split(",")
                                 if t.strip()]
        self._compile_mode = str(self.get_parameter("compile_mode").value or "default")
        self._profile_stages = bool(self.get_parameter("profile_stages").value)
        self._vit_glue = bool(self.get_parameter("vit_glue").value)
        self._pipeline_prep = bool(self.get_parameter("pipeline_prep").value)
        self._pipeline_margin_ms = float(self.get_parameter("pipeline_margin_ms").value)
        self._map_head_slice = bool(self.get_parameter("map_head_slice").value)
        self._vit_input_size = int(self.get_parameter("vit_input_size").value)
        self._rear_refresh = int(self.get_parameter("rear_view_refresh_every").value)
        self._stagger: Optional[orion_speedups.StaggeredViews] = None
        self._llm_down_t = bool(self.get_parameter("llm_down_proj_transpose").value)
        self._llm_int8 = bool(self.get_parameter("llm_int8").value)
        self._llm_int8_stats = str(self.get_parameter("llm_int8_stats").value or "")
        self._llm_int8_alpha = float(self.get_parameter("llm_int8_alpha").value)
        self._llm_int8_skip = [int(i) for i in (self.get_parameter("llm_int8_skip_layers").value or [])]
        self._llm_int8_targets = [t.strip() for t in
                                  str(self.get_parameter("llm_int8_targets").value or "").split(",") if t.strip()]
        self._overlap_heads = bool(self.get_parameter("overlap_heads").value)
        self._backbone_engine = str(self.get_parameter("backbone_engine").value or "")
        if self._backbone_engine:
            self._compile_targets = [t for t in self._compile_targets if t != "vit"]
        self._decoder = orion_speedups.ParallelDecoder(
            decode_image, workers=int(self.get_parameter("decode_workers").value))
        self._stage_timer: Optional[orion_speedups.StageTimer] = None
        self._base_frame = str(self.get_parameter("base_frame_id").value)
        self._map_frame = str(self.get_parameter("map_frame_id").value)
        self._min_traj_speed = float(self.get_parameter("min_trajectory_speed_mps").value)
        self._control_hz = float(self.get_parameter("control_hz").value)
        self._plan_stall_timeout = float(
            self.get_parameter("plan_stall_timeout_sec").value)
        self._brake_when_stale = bool(self.get_parameter("brake_when_stale").value)
        self._gnss_offset_x = float(self.get_parameter("gnss_mount_offset_x").value)
        if self._gnss_offset_x:
            self.get_logger().info(
                f"ego localised {self._gnss_offset_x:+.2f} m (vehicle frame) from the "
                f"odometry origin, matching the reference agent's GNSS mount")
        self._control_trace = bool(self.get_parameter("control_trace").value)
        self._trace_fh = None
        self._trace_tick = 0
        self._trace_plan_seq = 0
        self._trace_last_arrived = None
        self._trace_ticks_on_plan = 0
        self._last_pid_meta = None
        if self._control_trace:
            import os as _os
            _p = str(self.get_parameter("control_trace_path").value or "")
            if not _p:
                _p = "/benchmarking/log/control_trace_%s.csv" % (
                    time.strftime("%Y%m%d_%H%M%S"))
            try:
                _os.makedirs(_os.path.dirname(_p), exist_ok=True)
                self._trace_fh = open(_p, "w", buffering=1)
                self._trace_fh.write(
                    "t,tick,plan_seq,swap,ticks_on_plan,plan_age,stalled,"
                    "speed,desired_speed,delta,angle,angle_final,aim_x,aim_y,"
                    "steer,throttle,brake\n")
                self.get_logger().info("control trace -> %s" % _p)
            except Exception as _exc:
                self.get_logger().warn(
                    "control_trace disabled: cannot write %s (%s)" % (_p, _exc))
                self._control_trace = False

        # --- Publishers ---
        qos = 10
        # Self-contained closed loop (like orion_lite): the model's predicted
        # trajectory is consumed in-node by a PID controller that emits
        # CarlaEgoVehicleControl directly — no Autoware Trajectory / Stanley hop.
        self.declare_parameter("control_topic", "/carla/hero/vehicle_control_cmd")
        control_topic = self.get_parameter("control_topic").value
        self._control_pub = self.create_publisher(
            CarlaEgoVehicleControl, control_topic,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=1))
        self.get_logger().info(f"Publishing CarlaEgoVehicleControl on {control_topic}")

        cot_topic = self.get_parameter("cot_topic").value
        self._cot_pub = self.create_publisher(String, cot_topic, qos)
        self.get_logger().info(f"CoT/reasoning topic: {cot_topic} (publish_cot={self._publish_cot})")

        # --- Buffers: latest-only, so inference always runs on the freshest data ---
        # One latest-frame buffer per ORION camera. Created together with the
        # subscriptions below, i.e. only after the model has finished loading.
        self._cams: Optional[MultiCameraBuffer] = None
        self._latest_speed: float = 0.0
        self._latest_odom: Optional[Odometry] = None
        # The latest plan and the pose/time it was made at:
        #   (traj (N,2), ego_xy (2,), ego_theta, t_dispatch, t_arrived)
        # Written as ONE tuple by the inference thread and read as one by the
        # control timer, so the reader can never see half of an update.
        self._traj_state: Optional[tuple] = None
        self._last_control: Optional[tuple] = None
        self._ctrl_count: int = 0
        self._latest_imu: Optional[Imu] = None
        # Latched global plan: deque of (np.array([x, y]) map-frame, road_option:int).
        # Mutated (popped) as the ego advances, exactly like Bench2Drive RoutePlanner.
        self._route: deque = deque()
        self._last_ref_stamp: Optional[tuple] = None  # (sec, nanosec) of last-used CAM_FRONT
        self._frame_idx: int = 0  # per-route, like the agent's self.step
        # Per-route context. ORION is a streaming model: OrionHead keeps a rolling
        # memory across frames and only drops it when `scene_token` changes or
        # more than 2 s of stamp time pass. Both this node and the agent used a
        # constant token, which is fine upstream (fresh agent per route) but here
        # would let one route's memory bleed into the next. See
        # _reset_for_new_route.
        self._scene_token: str = "route-0000"
        self._route_serial: int = 0
        self._route_fingerprint = None
        self._pending_route_reset: bool = False
        # Temporal-memory continuity bookkeeping (see _check_memory_continuity).
        self._prev_sensor_ts = None
        self._memoryless_frames: int = 0
        self._dt_samples: int = 0
        self._last_dt_log_t: float = 0.0
        self._last_skip_log_t: float = 0.0  # throttle for dispatch-state logging

        # Cameras: RELIABLE. This is NOT a free choice — it was settled empirically:
        # each frame is 900x1600 bgra8 = 5.76 MB, which fragments into thousands of
        # UDP packets, and over this transport ~every sample loses at least one
        # fragment. RELIABLE retransmits and recovers it (echo/RViz/RELIABLE-orion
        # all receive frames); BEST_EFFORT discards the whole sample on any lost
        # fragment and receives 0/6 frames forever. For multi-MB samples on a lossy
        # link, best-effort is wrong. The reader does NOT wedge during the inference
        # blackout (a RELIABLE run picked up the latest frame fine after a run); any
        # observed stall is the CARLA synchronous control loop freezing the sim, not
        # this subscription.
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5
        )
        # carla_ros_bridge publishes odometry AND imu as RELIABLE (verified with
        # `ros2 topic info <topic> --verbose`). Match it so these subscriptions
        # don't show as a QoS mismatch; a BEST_EFFORT subscriber would still be
        # compatible, but these are small/low-rate messages so RELIABLE is free.
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10
        )
        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # --- Model (loaded in the worker thread for per-thread CUDA handles) ---
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None
        # Prep pipelining (see pipeline_prep): one prep worker, one prepared
        # batch at most, dispatch guarded by a lock because the forward-done
        # and prep-done callbacks fire on worker threads.
        self._prep_executor = ThreadPoolExecutor(max_workers=1)
        self._prep_future: Optional[Future] = None
        self._prep_lock = threading.Lock()       # serialises _build_batch (frame_idx, memory checks)
        self._dispatch_lock = threading.Lock()
        self._fwd_expected_end: Optional[float] = None
        self._fwd_ms_avg: Optional[float] = None
        self._prep_ms_avg: float = 150.0
        self._last_fwd_start: Optional[float] = None
        self._last_period_mark: Optional[float] = None
        self._model = None
        self._inference_only_pipeline = None
        self._mm_collate = None
        self._get_box_type = None

        self.get_logger().info("Loading ORION model ...")
        self._executor.submit(self._setup_model).result()

        # Create subscriptions ONLY AFTER the model has loaded. Model loading
        # blocks __init__ for ~60-90 s, and rclpy.spin() does not start until
        # __init__ returns. Any subscription created before this point has its DDS
        # reader buffering samples for that whole window with nothing draining them
        # (the executor isn't spinning yet). For a RELIABLE 5.76 MB camera stream
        # that unserviced pileup exhausts DDS reader resources and wedges delivery
        # — you get a couple of frames once spin starts, then silence. Creating the
        # subs here means they go live just as spin begins and are drained
        # immediately. (route is TRANSIENT_LOCAL, so it still latches the last
        # published global plan even though it subscribes late.)
        self._cams = MultiCameraBuffer(
            self, self._camera_topics, img_qos, self._camera_sync_tol
        )

        speed_topic = self.get_parameter("speed_topic").value
        self.create_subscription(Float32, speed_topic, self._speed_callback, 10)
        self.get_logger().info(f"Subscribed to speed topic: {speed_topic}")

        odom_topic = self.get_parameter("odometry_topic").value
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")

        imu_topic = self.get_parameter("imu_topic").value
        self.create_subscription(Imu, imu_topic, self._imu_callback, odom_qos)
        self.get_logger().info(f"Subscribed to IMU topic: {imu_topic}")

        route_topic = self.get_parameter("route_topic").value
        self.create_subscription(CarlaRoute, route_topic, self._route_callback, route_qos)
        self.get_logger().info(f"Subscribed to route topic: {route_topic}")

        self._timer = self.create_timer(self._inference_period, self._timer_callback)
        # Both timers run on the executor thread; inference itself is off on the
        # ThreadPoolExecutor, so this one keeps ticking during a forward pass --
        # which is the entire point of it.
        self._control_timer = self.create_timer(
            1.0 / self._control_hz, self._control_callback)
        self.get_logger().info(
            f"Control loop at {self._control_hz:g} Hz "
            f"(car stopped if no new plan arrives for "
            f"{self._plan_stall_timeout:g} s)")

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    def _setup_model(self) -> None:
        """Build the ORION model + the exact inference pipeline used by the agent."""
        repo_path = str(self.get_parameter("orion_repo_path").value or "")
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        # The ORION configs use RELATIVE paths (e.g. llm_path='ckpts/pretrain_qformer/')
        # for both the model weights AND the pipeline tokenizer, resolved against the
        # process CWD. Run from the repo root so they resolve, exactly like the agent.
        if repo_path:
            os.chdir(repo_path)

        # Bench2Drive PID controller (the same one orion_lite_rosnode uses);
        # pure-python, importable now that repo_path is on sys.path / the CWD.
        from team_code.pid_controller import PIDController
        self._pid = PIDController()

        # Import ORION's own machinery. Importing these packages also triggers the
        # registry registration of the `Orion` detector and all pipeline transforms.
        from mmcv import Config
        from mmcv.models import build_model
        from mmcv.utils import load_checkpoint
        from mmcv.datasets.pipelines import Compose
        from mmcv.parallel.collate import collate as mm_collate_to_batch_form
        from mmcv.core.bbox import get_box_type

        self._mm_collate = mm_collate_to_batch_form
        self._get_box_type = get_box_type

        cfg_path = str(self.get_parameter("orion_config_path").value)
        ckpt_path = str(self.get_parameter("orion_checkpoint_path").value)
        cfg = Config.fromfile(cfg_path)

        # Select inference precision by flipping the model flags (mutually
        # exclusive in Orion.__init__). fp16 => half-precision LLM + ViT backbone
        # (heads stay fp32); fp16_eval stays False so the GT-dependent metric
        # branch is skipped — we have no ground truth in ROS.
        precision = str(self.get_parameter("precision").value or "").lower()
        if precision == "fp16":
            cfg.model["fp16_infer"] = True
            cfg.model["fp16_eval"] = False
            cfg.model["fp32_infer"] = False
            self.get_logger().info("Inference precision: FP16 (fp16_infer)")
        elif precision == "fp32":
            cfg.model["fp16_infer"] = False
            cfg.model["fp16_eval"] = False
            cfg.model["fp32_infer"] = True
            self.get_logger().info("Inference precision: FP32 (fp32_infer)")
        else:
            self.get_logger().info("Inference precision: using config defaults")

        # Optional plugin import (matches the agent). The orion configs don't set
        # `plugin`, so this is usually a no-op.
        if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
            import importlib
            module_path = cfg.plugin_dir.rstrip("/").replace("/", ".")
            importlib.import_module(module_path)

        self._model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                                  test_cfg=cfg.get("test_cfg"))
        load_checkpoint(self._model, ckpt_path, map_location="cpu")
        self._model.cuda()
        self._model.eval()

        # Post-build speedups. Order matters: merge LoRA before compiling so
        # peft is not in the graph; wrap the stage timer last so it times the
        # final objects. fp16_enabled flags are static, so set them once here
        # instead of walking every module before every forward.
        info = self.get_logger().info
        if self._merge_lora:
            orion_speedups.merge_lora(self._model, info)
        if self._llm_flash_attn:
            orion_speedups.patch_llm_flash_attention(self._model, info)
        if self._vit_glue and not self._backbone_engine:
            orion_speedups.patch_vit_blocks(self._model, info)
        if self._vit_input_size != 640:
            orion_speedups.set_vit_input_size(self._model, self._vit_input_size, info)
        if self._llm_int8:
            orion_speedups.apply_llm_int8(self._model, self._llm_int8_stats, self._llm_int8_alpha,
                                          self._llm_int8_skip, info, self._llm_int8_targets)
            if self._llm_down_t and "down_proj" not in self._llm_int8_targets:
                orion_speedups.transpose_llm_down_proj(self._model, info)
        elif self._llm_down_t:
            orion_speedups.transpose_llm_down_proj(self._model, info)
        if self._compile_targets:
            orion_speedups.compile_submodules(
                self._model, self._compile_targets, self._compile_mode, info)
        if self._backbone_engine:
            orion_speedups.install_trt_backbone(self._model, self._backbone_engine, info)
        if self._map_head_slice:
            orion_speedups.slice_map_head_one2one(self._model, info)
        if self._rear_refresh > 1:
            self._stagger = orion_speedups.install_staggered_views(self._model, self._rear_refresh, info)
        if self._overlap_heads:
            orion_speedups.overlap_heads(self._model, info)
        custom_wrap_fp16_model(self._model)
        if self._profile_stages:
            self._stage_timer = orion_speedups.StageTimer()
            self._stage_timer.wrap_orion(self._model)

        # Build the inference-only pipeline, dropping the disk image loader exactly
        # like the agent does (images are already in memory).
        pipeline_cfg = [
            t for t in cfg.inference_only_pipeline
            if t["type"] not in ("LoadMultiViewImageFromFilesInCeph",)
        ]
        if self._vit_input_size != 640:
            orion_speedups.set_pipeline_input_size(pipeline_cfg, self._vit_input_size)
            self.get_logger().info(f"pipeline: ViT input {self._vit_input_size}x{self._vit_input_size}")
        self._inference_only_pipeline = Compose(pipeline_cfg)
        # Same transforms, per-image loops on a thread pool (bit-exact output).
        orion_speedups.parallelize_pipeline(
            self._inference_only_pipeline,
            workers=int(self.get_parameter("decode_workers").value),
            log=self.get_logger().info)

        self._warmup()
        self.get_logger().info("ORION model loaded and ready.")

    def _warmup(self) -> None:
        """Run ONE full forward on a black dummy frame so the first real
        inference isn't a cold-start outlier (CUDA kernel JIT, cuDNN autotune,
        allocator). Goes through the real pipeline so shapes are guaranteed
        correct. Runs in the model worker thread (same CUDA context as
        inference). Best-effort: logged, never fatal."""
        try:
            t0 = time.time()
            dummy = np.zeros((900, 1600, 3), dtype=np.uint8)   # matches 1600x900 cam
            lidar2global = np.eye(4) @ LIDAR2EGO
            results = {
                "lidar2img":     np.stack([LIDAR2IMG[c] for c in ORION_CAMERA_ORDER]),
                "lidar2cam":     np.stack([LIDAR2CAM[c] for c in ORION_CAMERA_ORDER]),
                "cam_intrinsic": [np.matmul(LIDAR2IMG[c], np.linalg.inv(LIDAR2CAM[c]))
                                  for c in ORION_CAMERA_ORDER],
                "img":         [dummy for _ in ORION_CAMERA_ORDER],
                "folder": " ", "scene_token": " ", "frame_idx": 0, "timestamp": 0.0,
                "box_type_3d": self._get_box_type("LiDAR")[0],
                "can_bus":     np.zeros(18),
                "command":     command2nohot(4),
                "ego_fut_cmd": command2hot(4),
                "ego_pose":    lidar2global,
                "ego_pose_inv": invert_matrix_egopose_numpy(lidar2global),
                "lidar2ego":   LIDAR2EGO,
                "l2g_r_mat":   lidar2global[0:3, 0:3],
                "l2g_t":       lidar2global[0:3, 3],
            }
            stacked = np.stack(results["img"], axis=-1)
            results["img_shape"] = results["ori_shape"] = results["pad_shape"] = stacked.shape
            results = self._inference_only_pipeline(results)
            batch = self._mm_collate([results], samples_per_gpu=1)
            for key, data in batch.items():
                if key != "img_metas" and torch.is_tensor(data[0]):
                    data[0] = data[0].to(self._device)
                if key == "input_ids":
                    for i in range(len(data[0])):
                        for k in range(len(data[0][i])):
                            data[0][i][k] = data[0][i][k].to(self._device)
            with torch.inference_mode():
                self._model(batch, return_loss=False)
                if self._stagger is not None:
                    # second forward = the reduced (front-only) ViT shape
                    self._model(batch, return_loss=False)
            torch.cuda.synchronize()
            if self._stagger is not None:
                self._stagger.reset()
            # The warmup forward is a REAL forward: it left the black dummy
            # frame in OrionHead's temporal memory. Clear test_flag so the
            # model's own forward_test drops both heads' memory before the
            # first real frame, instead of fusing it with a black image.
            if hasattr(self._model, "test_flag"):
                self._model.test_flag = False
            self.get_logger().info(f"Model warmup OK in {(time.time() - t0) * 1e3:.0f} ms")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Model warmup skipped ({type(e).__name__}: {e})")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def _speed_callback(self, msg: Float32) -> None:
        self._latest_speed = float(msg.data)

    def _odometry_callback(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _imu_callback(self, msg: Imu) -> None:
        self._latest_imu = msg

    def _route_callback(self, msg: CarlaRoute) -> None:
        # poses[] and road_options[] run in parallel (see Bench2Drive ros2_agent.py).
        n = min(len(msg.poses), len(msg.road_options))
        route = deque()
        for i in range(n):
            p = msg.poses[i]
            route.append(
                (np.array([p.position.x, p.position.y], dtype=np.float64),
                 int(msg.road_options[i]))
            )
        self._route = route

        # A CarlaRoute publication means the CARLA-side agent was rebuilt, i.e. a
        # new route started. The topic is TRANSIENT_LOCAL, so a late subscriber
        # also gets the latched message for the CURRENT route -- fingerprint the
        # content so a redelivery of the same plan does not wipe memory mid-route.
        fp = (len(route),
              tuple(route[0][0]) if route else (),
              tuple(route[-1][0]) if route else ())
        if fp != self._route_fingerprint:
            self._route_fingerprint = fp
            self._route_serial += 1
            self._scene_token = f"route-{self._route_serial:04d}"
            # Applied on the inference thread, never here: resetting the head
            # mid-forward would corrupt the run in flight.
            self._pending_route_reset = True
            self.get_logger().info(
                f"Route received: {len(route)} waypoints — NEW route "
                f"({self._scene_token}); context will be reset before the next "
                f"inference."
            )
        else:
            self.get_logger().info(
                f"Route received: {len(route)} waypoints — same plan as "
                f"{self._scene_token}, keeping context."
            )

    def _reset_for_new_route(self) -> None:
        """Drop everything scoped to the route that just ended.

        Upstream gets this for free: leaderboard_evaluator.py builds a fresh
        agent object for every route and destroy()s it afterwards, so no state
        can cross a route boundary. Here only the thin gateway agent on the CARLA
        box is rebuilt -- this node is one long-lived process started once by
        start_orion.sh for the whole run -- so the boundary has to be recreated
        by hand. Same problem, and the same fix, as
        simlingo_node._reset_for_new_route.

        What actually carries over, worst first:

          * OrionHead / map-head temporal memory. `pre_update_memory` keeps the
            rolling memory whenever `scene_token` matches AND the stamp delta is
            under 2 s; with a constant token the first frame of a new route would
            be fused with the last frames of the previous one -- a different map,
            kilometres away. Clearing `test_flag` makes the model's own
            `forward_test` call `reset_memory()` on both heads on the next pass,
            which is the same path it uses for the very first frame.
          * The PID's 40-sample integral windows. OrionAgent.setup() builds a new
            PIDController per route. Routes tend to END with the ego blocked, so
            the windows are at their most biased exactly when the next route
            starts -- and at ORION's frame rate 40 samples is many seconds.
          * frame_idx, which the agent restarts from 0 each route.

        Sensor buffers (images, odometry, speed, IMU) are deliberately left
        alone: they are overwritten continuously and describe the world, not the
        route.
        """
        if self._stagger is not None:
            self._stagger.reset()
        model = self._model
        if model is not None and hasattr(model, "test_flag"):
            # Let the model reset its own heads on the next forward_test.
            model.test_flag = False
        self._frame_idx = 0
        self._last_ref_stamp = None
        self._prev_sensor_ts = None
        self._reset_controller_state()
        self._pending_route_reset = False
        self.get_logger().info(
            f"Context reset for {self._scene_token}: temporal memory, "
            f"frame counter and controller state cleared."
        )

    def _reset_controller_state(self) -> None:
        # OrionAgent.setup() constructs PIDController() fresh per route; do the
        # same rather than reaching into its deques.
        # Imported locally, as in _setup_model: the Orion repo only joins
        # sys.path once the model has been set up, so there is no module-level
        # PIDController to reference here.
        from team_code.pid_controller import PIDController
        self._pid = PIDController()
        # Drop the previous route's plan with it: the control timer must not go
        # on steering the new route towards the old one's waypoints.
        self._traj_state = None
        self._last_control = None

    def _check_memory_continuity(self, sensor_ts: float) -> None:
        """Report when ORION is running without its temporal memory.

        `pre_update_memory` zeroes the whole memory when consecutive timestamps
        are >= 2 s apart. In CARLA's synchronous mode that never happens (the sim
        waits for the agent, so frames are 0.05 s apart in sim time no matter how
        slow inference is). In async real-time it happens on EVERY frame as soon
        as inference exceeds ~2 s — the model silently becomes a single-frame
        detector, and no log line anywhere says so. Hence this one.
        """
        prev, self._prev_sensor_ts = self._prev_sensor_ts, sensor_ts
        if prev is None:
            return
        dt = abs(sensor_ts - prev)
        self._dt_samples += 1
        if dt < ORION_MEMORY_MAX_DT:
            return
        self._memoryless_frames += 1

        now = time.monotonic()
        if now - self._last_dt_log_t < 10.0:
            return
        self._last_dt_log_t = now
        pct = 100.0 * self._memoryless_frames / max(self._dt_samples, 1)
        extra = ""
        if self._timestamp_mode == "agent":
            extra = (" — timestamp_mode=agent is masking this: the model is told "
                     f"{1.0 / ORION_AGENT_HZ:.2f} s elapsed, so memory is kept but "
                     "its motion reasoning is given a false dt")
        self.get_logger().warn(
            f"temporal memory: {dt:.2f} s between frames exceeds ORION's "
            f"{ORION_MEMORY_MAX_DT:.1f} s retention window, so the rolling memory "
            f"is zeroed and ORION runs single-frame "
            f"({self._memoryless_frames}/{self._dt_samples} frames, {pct:.0f}%)"
            f"{extra}. Latency benchmarks are unaffected; driving quality is not "
            f"comparable to the paper's synchronous-mode results."
        )

    # ------------------------------------------------------------------
    # Timer -> dispatch inference (non-blocking, single worker, always-latest)
    # ------------------------------------------------------------------
    def _log_dispatch_state(self, reason: str, interval_sec: float = 2.0) -> None:
        """Throttled log explaining why the timer is NOT dispatching inference.
        The timer fires at ~20 Hz, so without throttling these would flood the log.
        Use it to tell apart the silent-skip branches: a stalled camera stream
        ("no new frame") vs. a hung forward pass ("busy") vs. missing inputs."""
        now = time.monotonic()
        if now - self._last_skip_log_t < interval_sec:
            return
        self._last_skip_log_t = now
        self.get_logger().info(f"[dispatch] {reason}")

    def _timer_callback(self) -> None:
        # One inference in flight at a time -> no backlog of stale data.
        if self._active_future is not None and not self._active_future.done():
            self._maybe_pipeline_prep()
            # Seeing this persist for far longer than one inference => the forward
            # pass is hung (vs. "no new frame" below, which means the stream stalled).
            self._log_dispatch_state("busy: inference still in flight")
            return
        if self._prep_future is not None:
            # A pipelined prep is pending or done: its completion callback (or
            # this tick) starts the forward from the prepared batch.
            self._try_start_from_prepared()
            return
        snapshot = self._take_snapshot()
        if snapshot is None:
            return
        self._start_forward(self._executor.submit(self._run_inference, snapshot),
                            include_prep=True)

    def _take_snapshot(self) -> Optional[dict]:
        """Gate on complete, fresh inputs and snapshot the LATEST of each."""
        missing = self._cams.missing() if self._cams is not None else ["(no subs)"]
        if missing:
            self._log_dispatch_state(
                f"waiting: no frame yet from {', '.join(missing)}"
            )
            return None
        if self._latest_odom is None:
            self._log_dispatch_state("waiting: no odometry received yet")
            return None

        # Gate on a fresh front frame so we never reprocess the same data.
        ref_stamp = self._cams.reference_stamp()
        if self._require_new_frame and ref_stamp == self._last_ref_stamp:
            self._log_dispatch_state(
                f"skipped: no new front frame since last run "
                f"(stamp held at {ref_stamp[0]}.{ref_stamp[1]:09d}) — camera stream stalled?"
            )
            return None
        self._last_ref_stamp = ref_stamp

        # Snapshot the LATEST of every input at dispatch time (cheap copy).
        images = self._cams.snapshot()
        if images is None:
            self._log_dispatch_state("waiting: incomplete camera set")
            return None
        return {
            "images": images,
            "speed": self._latest_speed,
            "odom": self._latest_odom,
            "imu": self._latest_imu,
            # Wall clock, because the control timer below has no access to the
            # simulator's. It is the DISPATCH instant, not the capture instant:
            # the transport latency from shutter to here (link + decompress) is
            # not counted, so the plan is re-based slightly conservatively --
            # aiming a little nearer than the truth rather than past it.
            "t_dispatch": time.time(),
        }

    def _start_forward(self, future: Future, include_prep: bool) -> None:
        now = time.time()
        self._last_fwd_start = now
        expected = (self._fwd_ms_avg or 0.0) + (self._prep_ms_avg if include_prep else 0.0)
        self._fwd_expected_end = now + expected / 1e3 if self._fwd_ms_avg else None
        self._active_future = future
        future.add_done_callback(self._on_future_done)

    def _maybe_pipeline_prep(self) -> None:
        """While a forward is in flight: start preparing the next frame so it
        is ready right when the GPU frees up. Started late on purpose (see the
        pipeline_prep parameter) so the input is no older than it would be in
        the sequential path."""
        if (not self._pipeline_prep or self._prep_future is not None
                or self._fwd_expected_end is None or self._fwd_ms_avg is None
                or self._fwd_ms_avg < 2.0 * self._prep_ms_avg + self._pipeline_margin_ms):
            return
        if time.time() < self._fwd_expected_end - (self._prep_ms_avg + self._pipeline_margin_ms) / 1e3:
            return
        snapshot = self._take_snapshot()
        if snapshot is None:
            return
        self._prep_future = self._prep_executor.submit(self._prepare, snapshot)
        self._prep_future.add_done_callback(self._on_prep_done)

    def _on_prep_done(self, future: Future) -> None:
        self._try_start_from_prepared()

    def _try_start_from_prepared(self) -> None:
        """Start the forward on the prepared batch if the GPU worker is free.
        Called from the timer, the prep-done callback and the forward-done
        callback; the lock makes exactly one of them dispatch."""
        with self._dispatch_lock:
            fut = self._prep_future
            if fut is None or not fut.done():
                return
            if self._active_future is not None and not self._active_future.done():
                return
            self._prep_future = None
            try:
                prepared = fut.result()
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.get_logger().error(
                    f"ORION prep failed: {exc}\n{traceback.format_exc()}")
                return
            self._start_forward(self._executor.submit(self._forward, prepared),
                                include_prep=False)

    # ------------------------------------------------------------------
    # Payload construction — mirrors OrionAgent.run_step exactly
    # ------------------------------------------------------------------
    def _compute_driving_command(self, ego_xy: np.ndarray) -> int:
        """Time-varying navigation command fed to ORION (command2hot encoding):
          1=left, 2=right, 3=straight, 4=follow-lane, 5=lane-change-L, 6=lane-change-R.

        Faithful port of Bench2Drive RoutePlanner.run_step (team_code/planner.py):
        advance along the latched CarlaRoute using the ego position, popping
        waypoints already passed (within ROUTE_MIN_DIST, scanning up to
        ROUTE_MAX_DIST ahead), and return the road_option of the new front
        waypoint. Route poses and ego position are both in the ROS map frame, so
        the distance math matches the source (consistent frames). Falls back to
        the static `driving_command` parameter until the route arrives.
        """
        route = self._route
        if not route:
            return self._driving_command
        if len(route) == 1:
            return self._sanitize_cmd(route[0][1])

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(route)):
            if cumulative_distance > ROUTE_MAX_DIST:
                break
            cumulative_distance += np.linalg.norm(route[i][0] - route[i - 1][0])
            distance = np.linalg.norm(route[i][0] - ego_xy)
            if distance <= ROUTE_MIN_DIST and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i
        for _ in range(to_pop):
            if len(route) > 2:
                route.popleft()
        return self._sanitize_cmd(route[0][1])

    def _sanitize_cmd(self, cmd: int) -> int:
        """road_options is a uint8[] field, so CARLA's VOID (-1) arrives as 255.
        Anything outside the valid 1..6 RoadOption range -> static fallback."""
        cmd = int(cmd)
        return cmd if 1 <= cmd <= 6 else self._driving_command

    def _agent_ego_xy(self, xy, ego_theta: float) -> np.ndarray:
        """Vehicle-origin ENU position -> the position the reference agent uses.

        The agent's GNSS sits at (gnss_mount_offset_x, 0) in the vehicle frame;
        rotate that into ENU by the vehicle->ENU yaw and add it. Yaw only, like
        the agent, which works in a flat 2-D world.
        """
        xy = np.asarray(xy, dtype=np.float64)[:2]
        d = self._gnss_offset_x
        if not d:
            return xy.copy()
        return xy + np.array([d * math.cos(ego_theta), d * math.sin(ego_theta)],
                             dtype=np.float64)

    def _odom_to_canbus_ego_pose(self, odom: Odometry, imu: Optional[Imu], speed: float):
        """Build the 18-dim can_bus vector and the ego_pose/lidar2global matrices
        from a carla-simulator/ros-bridge ``nav_msgs/Odometry`` (+ ``sensor_msgs/Imu``).

        IMPORTANT — no manual sign flips here. The CARLA agent
        (orion_b2d_agent.py) reads RAW CARLA sensors (left-handed) and manually
        applies the CARLA->ROS conversion: ``can_bus[1] = -pos.y``,
        ``ego_theta = -compass + pi/2``, ``can_bus[11] *= -1``,
        ``can_bus[13:16] = -angular_velocity``. The ros-bridge does MOST of that
        conversion already (transforms.py negates position y, negates yaw, and
        rotates linear velocity into the body frame), so those need no flip here.

        Angular velocity is the exception, and it was wrong until 2026-09-16.
        ``carla_ros_bridge/imu.py`` publishes ``(-gx, +gy, -gz)`` -- it negates
        gyro **x and z**, not y, because angular velocity is a pseudovector and
        flipping the y axis inverts the other two components. The agent instead
        negates all three (``can_bus[13:16] = -angular_velocity`` over the RAW
        CARLA gyro), giving ``(-gx, -gy, -gz)``. Feeding the bridge values
        straight through therefore put ``can_bus[14]`` at ``+gy`` where the
        reference has ``-gy``, so the y component is flipped below. The
        ego_pose is the vehicle pose in the ROS "map" (ENU) frame == the agent's
        ego2world frame.
        """
        pose = odom.pose.pose
        # Position is already ROS/ENU: (carla_x, -carla_y, carla_z).
        pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)

        # Heading in the ROS map frame == agent's ego_theta (vehicle->ENU yaw).
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        ego_theta = float(Rotation.from_quat(quat).as_euler("zyx")[0])

        # Speed from the dedicated /carla/hero/speed topic (== CARLA speedometer,
        # the same source the agent used). Fall back to the odom twist magnitude.
        if speed is None or speed <= 0.0:
            lin = odom.twist.twist.linear
            speed = math.hypot(lin.x, lin.y)

        # Acceleration + angular velocity come from the IMU (same source the
        # agent used). The ros-bridge already publishes these in ROS body-frame
        # axes (x-fwd, y-left, z-up) with the CARLA->ROS sign conversion applied,
        # so we use them directly — no manual flips. Falls back to odom twist /
        # zeros if no IMU message has arrived yet.
        if imu is not None:
            acceleration = np.array(
                [imu.linear_acceleration.x, imu.linear_acceleration.y,
                 imu.linear_acceleration.z], dtype=np.float64
            )
            angular_velocity = np.array(
                [imu.angular_velocity.x, imu.angular_velocity.y,
                 imu.angular_velocity.z], dtype=np.float64
            )
        else:
            acceleration = np.zeros(3)
            ang = odom.twist.twist.angular
            angular_velocity = np.array([ang.x, ang.y, ang.z], dtype=np.float64)

        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        # The agent's GNSS-derived position, not the vehicle origin (see
        # AGENT_GNSS_MOUNT_X). ego2world below inherits it, so ego_pose and
        # lidar2global describe the same point the reference does.
        gnss_xy = self._agent_ego_xy(pos[:2], ego_theta)

        can_bus = np.zeros(18)
        can_bus[0] = gnss_xy[0]
        can_bus[1] = gnss_xy[1]      # already negated by the bridge — do NOT flip again
        can_bus[3:7] = rotation
        can_bus[7] = speed
        can_bus[10:13] = acceleration
        # y flipped: the bridge negated gyro x and z, the agent negates all three
        # (see the frame note above). Accelerometer needs no such flip -- the
        # bridge's (ax, -ay, az) already matches the agent's can_bus[11] *= -1.
        can_bus[13:16] = np.array(
            [angular_velocity[0], -angular_velocity[1], angular_velocity[2]],
            dtype=np.float64)
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180

        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = can_bus[0:2]

        lidar2global = ego2world @ LIDAR2EGO
        ego_pose = lidar2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        return can_bus, ego_pose, ego_pose_inv, lidar2global

    def _build_batch(self, snapshot: dict):
        """Reproduce the agent's `results` dict, run the ORION inference pipeline,
        and collate into a model-ready batch. Returns (batch, meta)."""
        odom = snapshot["odom"]

        results: dict = {}
        results["lidar2img"] = []
        results["lidar2cam"] = []
        results["cam_intrinsic"] = []
        results["img"] = []
        results["folder"] = " "
        results["scene_token"] = self._scene_token
        frame_idx = self._frame_idx
        results["frame_idx"] = frame_idx
        self._frame_idx += 1

        sensor_ts = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        if self._timestamp_mode == "agent":
            results["timestamp"] = frame_idx / ORION_AGENT_HZ
        else:
            results["timestamp"] = sensor_ts
        self._check_memory_continuity(sensor_ts)
        results["box_type_3d"], _ = self._get_box_type("LiDAR")

        # One decode per camera, each into its own slot — matching
        # OrionAgent.tick(), which hands the pipeline six distinct views. The
        # calibration appended alongside is per-camera, so the pixels and the
        # extrinsics must describe the same mount or the model is being lied to.
        images = snapshot["images"]
        # The six decodes run on a thread pool (OpenCV drops the GIL); in
        # replicate mode all six keys hold the SAME record object and the
        # decoder collapses them to a single decode.
        results["img"] = self._decoder(
            [images[cam] for cam in ORION_CAMERA_ORDER], self._jpeg_quality)
        for cam in ORION_CAMERA_ORDER:
            results["lidar2img"].append(LIDAR2IMG[cam])
            results["lidar2cam"].append(LIDAR2CAM[cam])
            results["cam_intrinsic"].append(
                np.matmul(LIDAR2IMG[cam], np.linalg.inv(LIDAR2CAM[cam]))
            )
        results["lidar2img"] = np.stack(results["lidar2img"], axis=0)
        results["lidar2cam"] = np.stack(results["lidar2cam"], axis=0)

        can_bus, ego_pose, ego_pose_inv, lidar2global = self._odom_to_canbus_ego_pose(
            odom, snapshot.get("imu"), float(snapshot.get("speed", 0.0))
        )
        results["can_bus"] = can_bus
        results["ego_pose"] = ego_pose
        results["ego_pose_inv"] = ego_pose_inv
        results["lidar2ego"] = LIDAR2EGO
        results["l2g_r_mat"] = lidar2global[0:3, 0:3]
        results["l2g_t"] = lidar2global[0:3, 3]

        # can_bus[0:2] IS the agent's position; reuse it so the route step, the
        # PID target and the pose handed to the model cannot drift apart.
        ego_xy = can_bus[0:2].copy()
        command = self._compute_driving_command(ego_xy)
        results["command"] = command2nohot(command)
        results["ego_fut_cmd"] = command2hot(command)

        stacked_imgs = np.stack(results["img"], axis=-1)
        results["img_shape"] = stacked_imgs.shape
        results["ori_shape"] = stacked_imgs.shape
        results["pad_shape"] = stacked_imgs.shape

        # Identical preprocessing + tokenization + collation as the agent.
        results = self._inference_only_pipeline(results)
        batch = self._mm_collate([results], samples_per_gpu=1)

        # Move tensors to the GPU exactly as the agent does.
        for key, data in batch.items():
            if key != "img_metas":
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self._device)
            if key == "input_ids":
                for i in range(len(data[0])):
                    for k in range(len(data[0][i])):
                        data[0][i][k] = data[0][i][k].to(self._device)

        # near_node: the PID target in ROS map/ENU frame. RoutePlanner.run_step
        # returns `self.route[0], self.route[1]` and OrionAgent.tick unpacks the
        # SECOND of those as command_near_xy:
        #     (_, curr_command), (near_node, near_command) = run_step(pos)
        # so this is route[1], not route[0]. Under the stock controller this only
        # reaches control_pid's `target`, which feeds `angle_target` — metadata
        # only, because use_target_to_aim is hardcoded False. Keeping it exact
        # means re-enabling that flag does not silently steer to the wrong point.
        near_node = (
            np.array(self._route[1][0], dtype=np.float64) if len(self._route) > 1
            else (np.array(self._route[0][0], dtype=np.float64) if self._route else None)
        )

        meta = {
            # CARLA's synchronous agent matches the control timestamp against the
            # FRONT-CAMERA frame's sim time (Bench2Drive requirement, per the
            # simlingo reference) — NOT the odometry stamp. The record is
            # (image.header.stamp, height, width, encoding, bytes).
            "stamp": snapshot["images"][REFERENCE_CAMERA][0],
            "ego_pos": np.array(
                [odom.pose.pose.position.x, odom.pose.pose.position.y,
                 odom.pose.pose.position.z]
            ),
            "ego_rot": Rotation.from_quat(
                [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                 odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
            ).as_matrix(),
            "ego_xy": ego_xy,                  # ROS ENU [x, y]
            "ego_theta": float(can_bus[16]),   # vehicle->ENU yaw (== agent ego_theta)
            "near_node": near_node,            # ROS ENU [x, y] or None
            "speed": float(snapshot.get("speed", 0.0)),
        }
        return batch, meta

    # ------------------------------------------------------------------
    # Inference (worker thread)
    # ------------------------------------------------------------------
    def _run_inference(self, snapshot: dict) -> dict:
        """Sequential path: prep + forward on the GPU worker."""
        return self._forward(self._prepare(snapshot))

    def _prepare(self, snapshot: dict) -> dict:
        """CPU half of a frame (decode, pipeline, collate, H2D). Runs on the GPU
        worker in the sequential path or on the prep worker when pipelined;
        the lock keeps _build_batch's bookkeeping (frame_idx, memory checks,
        route planner) strictly ordered."""
        with self._prep_lock:
            t0 = time.time()
            batch, meta = self._build_batch(snapshot)
            prep_ms = (time.time() - t0) * 1e3
        self._prep_ms_avg = 0.8 * self._prep_ms_avg + 0.2 * prep_ms
        return {"batch": batch, "meta": meta, "snapshot": snapshot, "prep_ms": prep_ms}

    @torch.no_grad()
    def _forward(self, prepared: dict) -> dict:
        t_start = time.time()
        # Applied here, on the inference thread, so it can never land in the
        # middle of a forward pass. A batch prepared before the reset carries
        # the old scene token / command, so it is rebuilt from its snapshot.
        if self._pending_route_reset:
            self._reset_for_new_route()
            prepared = self._prepare(prepared["snapshot"])
        batch, meta, snapshot = prepared["batch"], prepared["meta"], prepared["snapshot"]
        t_prep = time.time()

        with torch.inference_mode():
            output = self._model(batch, return_loss=False)
        torch.cuda.synchronize()
        t_forward = time.time()
        stages = self._stage_timer.report() if self._stage_timer is not None else None
        forward_ms = (t_forward - t_prep) * 1e3
        self._fwd_ms_avg = forward_ms if self._fwd_ms_avg is None else 0.8 * self._fwd_ms_avg + 0.2 * forward_ms
        period_ms = (t_start - self._last_period_mark) * 1e3 if self._last_period_mark else None
        self._last_period_mark = t_start

        out = output[0]
        # (6, 2) future trajectory (cumulative for the VAE head), in ORION's native
        # LIDAR_TOP frame. The Bench2Drive PID was tuned in THIS frame, so feed it
        # the RAW preds — do NOT apply the base_link axis-swap that the Autoware
        # trajectory path (orion_node.py) used.
        ego_fut_preds = out["pts_bbox"]["ego_fut_preds"].cpu().numpy()

        # The plan is handed to the control loop rather than collapsed into a
        # single control here. _compute_control still does exactly what
        # OrionAgent.run_step does -- it is just called at control_hz against a
        # plan re-cut for where the ego is by then (see _control_callback).
        self._traj_state = (
            ego_fut_preds.copy(),
            np.asarray(meta["ego_xy"], dtype=np.float64).copy(),
            float(meta["ego_theta"]),
            float(snapshot.get("t_dispatch", time.time())),   # when it was SEEN
            time.time(),                                      # when it ARRIVED
        )

        # CoT / VQA reasoning text — only present on reasoning configs.
        cot = self._extract_reasoning(out.get("text_out")) if self._publish_cot else None
        if cot:
            self._cot_pub.publish(String(data=cot))

        return {
            "prep_ms": prepared["prep_ms"],
            "forward_ms": forward_ms,
            "total_ms": prepared["prep_ms"] + forward_ms,
            "period_ms": period_ms,
            "num_pts": len(ego_fut_preds),
            "stages": stages,
        }

    def _extract_reasoning(self, text_out) -> Optional[str]:
        """text_out is a list like [{'Q': str, 'A': [str]}, ...] from simple_test_pts."""
        if not text_out:
            return None
        parts = []
        for qa in text_out:
            try:
                q = qa.get("Q")
                a = qa.get("A")
                a = a[0] if isinstance(a, (list, tuple)) and a else a
                if a:
                    parts.append(f"Q: {q}\nA: {a}" if q else str(a))
            except AttributeError:
                continue
        return "\n\n".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Closed-loop control, decoupled from inference
    # ------------------------------------------------------------------
    @staticmethod
    def _rot(angle: float) -> np.ndarray:
        c, sn = math.cos(angle), math.sin(angle)
        return np.array([[c, -sn], [sn, c]])

    def _rebase_plan(self, traj: np.ndarray, age: float,
                     ego_xy0: np.ndarray, theta0: float,
                     ego_xy_now: np.ndarray, theta_now: float) -> np.ndarray:
        """The plan, re-cut from where the ego is NOW and in its current frame.

        Two corrections, both required, for different reasons:

        **Time.** ORION's waypoints are 0.5 s apart measured from the frame it
        looked at, and that frame is ~2 s old by the time the forward pass
        returns. Handing control_pid `waypoints[0]` unmodified aims the car at a
        point it drove past a second ago -- and worse, control_pid reads
        ``desired_speed`` from ``||waypoints[0]||``, so a point the car has
        nearly reached reads as "slow down". The plan is therefore resampled
        from `age` onwards: element j is where the model expected to be at
        ``age + 0.5*(j+1)``.

        **Frame.** The waypoints are in the ego frame of the capture instant and
        the ego has moved since, so they are rotated and translated into the
        current ego frame using odometry. This is also what makes the scheme
        self-correcting: if the car did not go where the model predicted, the
        geometry is still relative to where it actually is.

        Beyond the 3 s horizon the last segment's velocity is extended. That is
        not decoration -- a 2 s forward pass eats 2 s of a 3 s plan, so without
        it the controller runs dry for the second half of every inference cycle
        and the car would brake every other second.

        The plan is treated as rooted at the ego, which is how control_pid reads
        it (``||waypoints[0]||`` as distance travelled in 0.5 s). The LIDAR2EGO
        offset is not applied for that reason; it would also almost cancel,
        being constant in the vehicle frame.
        """
        knots = np.vstack((np.zeros((1, 2)), np.asarray(traj, dtype=np.float64)))
        horizon = ORION_TRAJ_DT * (len(knots) - 1)
        v_end = (knots[-1] - knots[-2]) / ORION_TRAJ_DT

        def sample(t: float) -> np.ndarray:
            if t >= horizon:
                return knots[-1] + (t - horizon) * v_end
            u = max(t, 0.0) / ORION_TRAJ_DT
            i = min(int(math.floor(u)), len(knots) - 2)
            return knots[i] + (u - i) * (knots[i + 1] - knots[i])

        pts = np.array([sample(age + ORION_TRAJ_DT * (j + 1))
                        for j in range(len(traj))])

        # world -> plan frame is rot(+raw); see _compute_control for why raw is
        # pi/2 - ego_theta. Composing the two poses gives p' = R(dtheta)(p - o),
        # with o the current ego position expressed in the plan's own frame.
        raw0 = np.pi / 2.0 - theta0
        raw_now = np.pi / 2.0 - theta_now
        o = self._rot(raw0) @ (np.asarray(ego_xy_now) - np.asarray(ego_xy0))
        return (self._rot(raw_now - raw0) @ (pts - o).T).T

    def _control_callback(self) -> None:
        """Track the latest plan. Runs at control_hz, including during a forward pass."""
        state = self._traj_state          # one atomic read of the whole tuple
        if state is None or self._latest_odom is None:
            return
        traj, ego_xy0, theta0, t0, t_arrived = state

        now = time.time()
        stalled = now - t_arrived
        if stalled > self._plan_stall_timeout:
            self._stop_the_car(
                f"no new plan for {stalled:.1f}s "
                f"(limit {self._plan_stall_timeout:g}s) -- the model has stopped "
                f"answering")
            return
        # The plan's own age is NOT a reason to stop: it is ~2 s old the moment
        # it arrives and grows to ~4 s before the next one lands, which is
        # simply what a 2 s forward pass costs. _rebase_plan extrapolates past
        # the 3 s horizon precisely so this window stays driveable.
        age = now - t0

        odom = self._latest_odom
        ego_xy_now = np.array([odom.pose.pose.position.x,
                               odom.pose.pose.position.y], dtype=np.float64)
        theta_now = float(Rotation.from_quat([
            odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w,
        ]).as_euler("zyx")[0])
        # ego_xy0 came from meta["ego_xy"], which is the GNSS-mount point. Use the
        # same convention here or _rebase_plan re-cuts the plan against a
        # displacement that is wrong by the mount offset whenever the ego turns.
        ego_xy_now = self._agent_ego_xy(ego_xy_now, theta_now)

        plan = self._rebase_plan(traj, age, ego_xy0, theta0, ego_xy_now, theta_now)

        # Read, never popped: the route deque is advanced on the inference
        # thread by _compute_driving_command, and popping from here would race
        # it. near_node only reaches control_pid's `target` -> `angle_target`,
        # which is metadata while use_target_to_aim is False, so a slightly
        # stale one changes nothing today -- keeping it exact means re-enabling
        # that flag does not silently steer to the wrong point.
        near_node = (
            np.array(self._route[1][0], dtype=np.float64) if len(self._route) > 1
            else (np.array(self._route[0][0], dtype=np.float64) if self._route else None)
        )
        meta = {
            "stamp": self.get_clock().now().to_msg(),
            "ego_xy": ego_xy_now,
            "ego_theta": theta_now,
            "near_node": near_node,
            "speed": float(self._latest_speed),
        }
        control = self._compute_control(plan, meta)
        if control is None:
            return
        self._control_pub.publish(control)
        self._last_control = (control.steer, control.throttle, control.brake)
        self._ctrl_count += 1
        if self._control_trace:
            self._write_control_trace(t_arrived, age, stalled, control)

    def _write_control_trace(self, t_arrived, age, stalled, control) -> None:
        """One CSV row per control tick. See the control_trace parameter."""
        if self._trace_fh is None:
            return
        m = self._last_pid_meta or {}
        swap = 0
        if t_arrived != self._trace_last_arrived:
            self._trace_last_arrived = t_arrived
            self._trace_plan_seq += 1
            self._trace_ticks_on_plan = 0
            swap = 1
        self._trace_ticks_on_plan += 1
        self._trace_tick += 1
        nan = float("nan")
        aim = m.get("aim") or (nan, nan)
        def g(k):
            v = m.get(k, nan)
            try:
                return float(v)
            except (TypeError, ValueError):
                return nan
        try:
            self._trace_fh.write(
                "%.6f,%d,%d,%d,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.5f,%.5f,"
                "%.4f,%.4f,%.5f,%.4f,%.4f\n" % (
                    time.time(), self._trace_tick, self._trace_plan_seq, swap,
                    self._trace_ticks_on_plan, age, stalled,
                    g("speed"), g("desired_speed"), g("delta"),
                    g("angle"), g("angle_final"),
                    float(aim[0]), float(aim[1]),
                    control.steer, control.throttle, control.brake))
        except Exception:
            # A tuning aid must never be able to take the control loop down.
            pass

    def _stop_the_car(self, why: str) -> None:
        if not self._brake_when_stale:
            return
        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.steer = 0.0
        cmd.throttle = 0.0
        cmd.brake = 1.0
        self._control_pub.publish(cmd)
        self._last_control = (0.0, 0.0, 1.0)
        self.get_logger().warn(f"braking: {why}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # PID control (Bench2Drive parity)
    # ------------------------------------------------------------------
    def _compute_control(self, ego_fut_preds: np.ndarray, meta: dict) -> Optional[CarlaEgoVehicleControl]:
        """Bench2Drive PID over the predicted trajectory -> CarlaEgoVehicleControl.
        Faithful port of orion_b2d_agent: build the near-waypoint target in the
        vehicle-local frame, run control_pid, and publish the resulting
        steer/throttle/brake DIRECTLY as carla.VehicleControl (no AckermannDrive
        round-trip, so no carla_ackermann_control node is needed).

        Frame note: this node works in the ROS map/ENU frame, where ego_theta is
        the vehicle->ENU yaw (== the agent's ego_theta). The agent's raw compass
        heading is therefore raw_theta = pi/2 - ego_theta. Route and ego position
        are already ENU (the ros-bridge applied the CARLA->ROS conversion), so the
        world->local rotation needs no Y-flip — unlike orion_lite which flips Y on
        the raw CARLA coordinates.
        """
        near_node = meta.get("near_node")
        if near_node is None:
            self.get_logger().warn("No global plan yet; not publishing control.",
                                   throttle_duration_sec=5.0)
            return None

        ego_xy = meta["ego_xy"]
        raw_theta = np.pi / 2.0 - meta["ego_theta"]
        near_xy_world = np.array([near_node[0] - ego_xy[0], near_node[1] - ego_xy[1]])
        rot = np.array([[math.cos(raw_theta), -math.sin(raw_theta)],
                        [math.sin(raw_theta),  math.cos(raw_theta)]])
        local_command_xy = rot @ near_xy_world

        # speed must be a numpy SCALAR (0-dim), not a (1,) array and not a Python
        # float: control_pid does `desired_speed - speed` -> the speed PID's
        # integral window (np.mean), and also `speed.astype(np.float64)` in its
        # metadata. A (1,) array corrupts the scalar-seeded window (inhomogeneous
        # np.mean) and breaks `(speed/desired_speed) > ratio`; a Python float has
        # no .astype. np.float64 satisfies all three.
        steer, throttle, brake, metadata = self._pid.control_pid(
            ego_fut_preds, np.float64(meta["speed"]), local_command_xy)
        self._last_pid_meta = metadata

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 0.75))
        brake = float(brake)
        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0
        # OrionAgent.run_step caps the vehicle at 5 m/s (18 km/h) by cutting
        # throttle outright — applied AFTER the brake rules above, and not part of
        # control_pid, so it is easy to miss when porting. Without it the car runs
        # well past the speed the reference eval ever reaches, which changes
        # closed-loop behaviour (and the Bench2Drive score) on its own.
        if float(meta["speed"]) > ORION_SPEED_CAP_MPS:
            throttle = 0.0

        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = meta["stamp"]
        cmd.steer = steer
        cmd.throttle = throttle
        cmd.brake = brake
        return cmd

    # ------------------------------------------------------------------
    # Output conversion (Autoware parity)
    # ------------------------------------------------------------------
    def _to_autoware_trajectory(self, traj_np: np.ndarray, stamp) -> "Trajectory":
        traj_msg = Trajectory()
        traj_msg.header.stamp = stamp
        traj_msg.header.frame_id = self._base_frame  # ORION waypoints are ego-frame

        last_xy: Optional[tuple] = None
        for idx, point in enumerate(traj_np):
            tp = TrajectoryPoint()
            tp.pose.position.x = float(point[0])
            tp.pose.position.y = float(point[1])
            tp.pose.position.z = 0.0

            # Heading from consecutive waypoints.
            if idx + 1 < len(traj_np):
                dx = float(traj_np[idx + 1][0]) - float(point[0])
                dy = float(traj_np[idx + 1][1]) - float(point[1])
            elif last_xy is not None:
                dx = float(point[0]) - last_xy[0]
                dy = float(point[1]) - last_xy[1]
            else:
                dx, dy = 1.0, 0.0
            yaw = math.atan2(dy, dx)
            q = Rotation.from_euler("z", yaw).as_quat()
            tp.pose.orientation.x = float(q[0])
            tp.pose.orientation.y = float(q[1])
            tp.pose.orientation.z = float(q[2])
            tp.pose.orientation.w = float(q[3])

            if last_xy is None:
                speed = 0.0
            else:
                dist = math.hypot(float(point[0]) - last_xy[0], float(point[1]) - last_xy[1])
                speed = dist / ORION_TRAJ_DT
            tp.longitudinal_velocity_mps = float(max(speed, self._min_traj_speed))
            tp.lateral_velocity_mps = 0.0
            tp.acceleration_mps2 = 0.0
            tp.heading_rate_rps = 0.0

            t = idx * ORION_TRAJ_DT
            tp.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))

            traj_msg.points.append(tp)
            last_xy = (float(point[0]), float(point[1]))

        # Back-fill first point velocity so the controller doesn't stall at v=0.
        if len(traj_msg.points) >= 2:
            traj_msg.points[0].longitudinal_velocity_mps = (
                traj_msg.points[1].longitudinal_velocity_mps
            )
        return traj_msg

    def _trajectory_to_markers(self, traj_np: np.ndarray, ego_pos: np.ndarray,
                               ego_rot: np.ndarray, stamp) -> MarkerArray:
        marker_array = MarkerArray()
        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = self._map_frame
        line.ns = "orion_trajectory"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.3
        line.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        line.pose.orientation.w = 1.0
        for point in traj_np:
            local = np.array([float(point[0]), float(point[1]), 0.0])
            world = ego_rot @ local + ego_pos
            line.points.append(Point(x=float(world[0]), y=float(world[1]), z=float(world[2])))
        marker_array.markers.append(line)
        return marker_array

    # ------------------------------------------------------------------
    def _on_future_done(self, future: Future) -> None:
        # Runs on the GPU worker right after the forward: if a pipelined prep
        # is already done, the next forward starts here, before the log below.
        self._try_start_from_prepared()
        try:
            m = future.result()
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.get_logger().error(
                f"ORION inference failed: {exc}\n{traceback.format_exc()}")
            return
        if not m:
            return
        ctrl = ""
        if self._last_control is not None:
            steer, throttle, brake = self._last_control
            ctrl = (f" -> latest control steer={steer:+.3f} "
                    f"throttle={throttle:.3f} brake={brake:.3f}")
        # The control count is the whole point of the split: it should be
        # ~control_hz x the inference time (about 40 at 20 Hz and 2 s/pass).
        # If it reads 1, the control timer is not running and the car is being
        # driven one command per forward pass again.
        published, self._ctrl_count = self._ctrl_count, 0
        period = f", period={m['period_ms']:.0f} ms" if m.get("period_ms") else ""
        self.get_logger().info(
            f"ORION inference: total={m['total_ms']:.1f} ms "
            f"(prep={m['prep_ms']:.1f}, forward={m['forward_ms']:.1f}{period}), "
            f"{m['num_pts']} waypoints, ~{1000.0 / max(m['total_ms'], 1e-3):.2f} FPS, "
            f"{published} controls published since the last one"
            f"{ctrl}"
        )
        if m.get("stages"):
            self.get_logger().info(
                "ORION stages (ms): "
                + orion_speedups.StageTimer.format(m["stages"], m["forward_ms"]))

    def destroy_node(self) -> None:
        if self._trace_fh is not None:
            try:
                self._trace_fh.close()
            except Exception:
                pass
            self._trace_fh = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._prep_executor.shutdown(wait=False, cancel_futures=True)
        self._decoder.shutdown()
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    # Drop the "[<sim-time>]" timestamp from the console log prefix
    # (default format is "[{severity}] [{time}] [{name}]: {message}").
    os.environ.setdefault(
        "RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}")
    rclpy.init(args=args)
    node = OrionWithPidRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
