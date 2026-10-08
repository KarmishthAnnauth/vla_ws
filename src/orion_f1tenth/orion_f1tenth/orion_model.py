#!/usr/bin/env python3
"""ORION model runner, free of ROS and of CARLA.

Everything model-specific from ``orion_ros/orion_withpid_node.py`` lives here:
config + checkpoint loading, the post-build speedups, the agent's ``results``
dict, ORION's own ``inference_only_pipeline`` + ``collate``, and the forward
pass.  The ROS node only gathers sensor data and converts frames.

The payload is prepared EXACTLY as the reference closed-loop agent does it
(``Orion/team_code/orion_b2d_agent.py::OrionAgent.run_step``): the same
``results`` dict field for field, the same calibration constants, then ORION's
``Compose(cfg.inference_only_pipeline)`` (resize/crop, normalise, pad, VQA
tokenisation) and ``mmcv.parallel.collate``.  Nothing of the preprocessing is
reimplemented.  Only the raw inputs differ:

  * one real camera instead of six CARLA cameras.  The front frame is made to
    look like ORION's ``CAM_FRONT`` (1600x900, 70 deg horizontal FOV) and is
    also handed to the two front-side slots (``side_views="copy"``); the three
    rear slots are black.  The six calibration matrices stay the agent's.
  * pose/speed from the F1TENTH stack instead of GNSS/IMU/speedometer, already
    scaled into the model's world (see ``world_scale`` in the node).

Threading contract: ``load()`` and every ``infer()`` must run in the *same* OS
thread (Jetson iGPU cuBLAS/cuDNN handles are per thread).  ``build_batch()``
is CPU work plus a host-to-device copy and may run on another thread, as the
reference node's prep worker does.

Speed profiles (``profile``) bundle the speedups of ``orion_speedups.py``,
all post-build transforms on the model object (ORION sources untouched):

  baseline  what the agent runs: no speedup at all                (~1.75 s forward)
  eager     exact fp16 speedups, no torch.compile                  (~1.5 s)
  fast      + torch.compile of ViT / LLM / heads: the 2026-09-14
            "1.0 s per frame" configuration, trajectory within 1 cm
            of the baseline                                        (~0.93 s forward)
  int8      fast + W8A8 INT8 for the LLM MLP projections (SmoothQuant,
            layers 0,1,30,31 kept fp16): 3-5 cm trajectory shift    (~0.85 s)
  lite      fast + 512 px ViT input, rear views every other frame:
            0.47 m trajectory shift, a different model in effect   (~0.7 s)
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants of the checkpoint (OrionAgent.sensors() / setup()).  Fixed
# properties of the model, not of the physical camera.
# ---------------------------------------------------------------------------
ORION_CAMERA_ORDER: List[str] = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
FRONT_SLOTS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
REAR_SLOTS = ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

TRAIN_CAM_W = 1600
TRAIN_CAM_H = 900
TRAIN_FRONT_FOV_DEG = 70.0          # CAM_FRONT / sides; CAM_BACK is 110
# The agent re-encodes every camera frame at JPEG quality 20 (OrionAgent.tick).
AGENT_JPEG_QUALITY = 20
# ORION predicts 6 waypoints at 2 Hz: 0.5 s spacing, 3 s horizon.
ORION_TRAJ_DT = 0.5
ORION_NUM_WAYPOINTS = 6
# OrionHead keeps its temporal memory only while consecutive timestamps are
# < 2 s apart and the scene token is unchanged.
ORION_MEMORY_MAX_DT = 2.0
# OrionAgent: results['timestamp'] = step / 20 (synchronous 20 Hz CARLA).
ORION_AGENT_HZ = 20.0
# Bench2Drive RoutePlanner(4.0, 50.0) in the agent -- model metres.
ROUTE_MIN_DIST = 4.0
ROUTE_MAX_DIST = 50.0
# RoadOption codes fed as ego_fut_cmd (command2hot): 1=left 2=right 3=straight
# 4=follow-lane 5=change-left 6=change-right.
CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT, CMD_FOLLOW = 1, 2, 3, 4
CMD_NAMES = {1: "LEFT", 2: "RIGHT", 3: "STRAIGHT", 4: "LANEFOLLOW", 5: "CHANGELEFT", 6: "CHANGERIGHT"}

# Verbatim copies from Orion/team_code/orion_b2d_agent.py (OrionAgent.setup).
# Part of the model's input contract: DO NOT edit.
LIDAR2IMG: Dict[str, np.ndarray] = {
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
LIDAR2CAM: Dict[str, np.ndarray] = {
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


_CUSTOM_FP16 = dict(map_head=False, pts_bbox_head=False)


def custom_wrap_fp16_model(model) -> None:
    """orion_b2d_agent.custom_wrap_fp16_model: fp16 everywhere except the heads."""
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True
    for module_name, v in _CUSTOM_FP16.items():
        if module_name in model._modules:
            model._modules[module_name].fp16_enabled = v


# ---------------------------------------------------------------------------
# Speed profiles: the orion_ros node parameters each one sets.  Everything
# not listed keeps the orion_ros default (decode on a thread pool, ...).
# ---------------------------------------------------------------------------
_EXACT = dict(merge_lora=True, llm_flash_attn=True, vit_glue=True, map_head_slice=True,
              llm_down_proj_transpose=True, llm_int8=False, vit_input_size=640,
              rear_view_refresh_every=1, compile_targets="", compile_mode="default")
SPEED_PROFILES: Dict[str, Dict] = {
    "baseline": dict(merge_lora=False, llm_flash_attn=False, vit_glue=False, map_head_slice=False,
                     llm_down_proj_transpose=False, llm_int8=False, vit_input_size=640,
                     rear_view_refresh_every=1, compile_targets="", compile_mode="default"),
    "eager": dict(_EXACT),
    "fast": dict(_EXACT, compile_targets="heads,llm,vit"),
    "int8": dict(_EXACT, compile_targets="heads,llm,vit", llm_int8=True),
    "lite": dict(_EXACT, compile_targets="heads,llm,vit", vit_input_size=512, rear_view_refresh_every=2),
}
# SmoothQuant recipe of the int8 profile (tools/int8_sweep.py, 2026-09-14).
LLM_INT8_ALPHA = 0.8
LLM_INT8_SKIP_LAYERS = (0, 1, 30, 31)
LLM_INT8_TARGETS = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# Camera frame formatting (any thread)
# ---------------------------------------------------------------------------
def crop_to_hfov(bgr: np.ndarray, camera_hfov_deg: float, target_hfov_deg: float = TRAIN_FRONT_FOV_DEG) -> np.ndarray:
    """Keep the central ``target_hfov_deg`` of a pinhole frame with ``camera_hfov_deg``.

    A 110 deg webcam frame fed whole into a 70 deg camera slot shows the model a
    scene whose objects are 1.9x too small and far; cropping the central sector
    makes the pixels match the intrinsics ORION was trained with (what the
    70 deg CARLA camera would have seen).  0 = unknown FOV, no crop.
    """
    if camera_hfov_deg <= target_hfov_deg:
        return bgr
    h, w = bgr.shape[:2]
    keep = w * math.tan(math.radians(target_hfov_deg) / 2.0) / math.tan(math.radians(camera_hfov_deg) / 2.0)
    new_w = max(2, int(round(keep)))
    x0 = (w - new_w) // 2
    return bgr[:, x0:x0 + new_w]


def format_camera_frame(bgr: np.ndarray, out_w: int = TRAIN_CAM_W, out_h: int = TRAIN_CAM_H,
                        camera_hfov_deg: float = 0.0) -> np.ndarray:
    """Make a real camera frame look like ORION's CAM_FRONT.

    1. optional crop to the 70 deg horizontal FOV (``camera_hfov_deg`` > 70);
    2. centre-crop to the training aspect ratio (16:9) rather than squashing;
    3. resize to 1600x900, the resolution the hard-coded calibration assumes.
    Returns BGR uint8, the channel order the agent hands the pipeline
    (NormalizeMultiviewImage does the BGR->RGB conversion itself).
    """
    bgr = crop_to_hfov(bgr, camera_hfov_deg)
    h, w = bgr.shape[:2]
    target_aspect = out_w / out_h
    if w / h > target_aspect:                       # too wide: crop the sides
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        bgr = bgr[:, x0:x0 + new_w]
    elif w / h < target_aspect:                     # too tall: crop top/bottom
        new_h = int(round(w / target_aspect))
        y0 = (h - new_h) // 2
        bgr = bgr[y0:y0 + new_h, :]
    if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
        interp = cv2.INTER_AREA if bgr.shape[1] > out_w else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=interp)
    return np.ascontiguousarray(bgr)


def jpeg_roundtrip(bgr: np.ndarray, quality: int = AGENT_JPEG_QUALITY) -> np.ndarray:
    """OrionAgent.tick: ``imencode('.jpg', img, quality 20)`` then ``imdecode``."""
    if quality <= 0:
        return bgr
    _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_views(front_bgr: np.ndarray, side_views: str = "copy") -> List[np.ndarray]:
    """The six camera slots, ORION order, from one front frame.

    side_views="copy":  front frame in CAM_FRONT, CAM_FRONT_LEFT and
                        CAM_FRONT_RIGHT (the model is told the same pixels sit
                        at yaw 0 / -55 / +55 deg), rear slots black.
    side_views="black": front frame in CAM_FRONT only, five black slots.
    """
    if front_bgr.shape[:2] != (TRAIN_CAM_H, TRAIN_CAM_W):
        raise ValueError(f"front frame must be {TRAIN_CAM_W}x{TRAIN_CAM_H}, got {front_bgr.shape[1]}x{front_bgr.shape[0]}")
    black = np.zeros_like(front_bgr)
    if side_views == "copy":
        return [front_bgr, front_bgr, front_bgr, black, black, black]
    if side_views == "black":
        return [front_bgr, black, black, black, black, black]
    raise ValueError(f"side_views must be 'copy' or 'black', got {side_views!r}")


# ---------------------------------------------------------------------------
# Ego state -> can_bus / ego_pose, as OrionAgent.run_step builds them
# ---------------------------------------------------------------------------
def build_can_bus(pos_xy: Sequence[float], ego_theta: float, speed: float,
                  acceleration: Sequence[float] = (0.0, 0.0, 0.0),
                  angular_velocity: Sequence[float] = (0.0, 0.0, 0.0)):
    """The agent's 18-dim can_bus and ego_pose / ego_pose_inv / lidar2global.

    Inputs are in the agent's FINAL conventions, i.e. after its CARLA->ROS
    conversion: a right-handed world (x, y) with ``ego_theta`` the
    counter-clockwise yaw from +x (agent: ``-compass + pi/2``), body-frame
    acceleration with y to the LEFT (agent: ``can_bus[11] *= -1``) and
    angular velocity with z counter-clockwise (agent: ``-angular_velocity``
    over CARLA's left-handed gyro).  A ROS map pose and ROS body twists go in
    unchanged; the position is the point the agent localises from (its GNSS
    at x = -1.4 m in the vehicle frame, i.e. about the rear axle -- the
    F1TENTH's base_link).
    """
    from pyquaternion import Quaternion

    can_bus = np.zeros(18)
    can_bus[0] = float(pos_xy[0])
    can_bus[1] = float(pos_xy[1])
    can_bus[3:7] = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
    can_bus[7] = float(speed)
    can_bus[10:13] = np.asarray(acceleration, dtype=np.float64)
    can_bus[13:16] = np.asarray(angular_velocity, dtype=np.float64)
    can_bus[16] = ego_theta
    can_bus[17] = ego_theta / np.pi * 180
    ego2world = np.eye(4)
    ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
    ego2world[0:2, 3] = can_bus[0:2]
    lidar2global = ego2world @ LIDAR2EGO
    ego_pose = lidar2global
    ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
    return can_bus, ego_pose, ego_pose_inv, lidar2global


def desired_speed_from_waypoints(wps: np.ndarray) -> float:
    """Scalar target speed exactly as Bench2Drive PIDController.control_pid derives it:
    0.75 * |wp0| * 2 + 0.25 * |wp1 - wp0| * 2  (0.5 s per waypoint -> m/s)."""
    wps = np.asarray(wps, dtype=np.float64)
    return float(0.75 * np.linalg.norm(wps[0]) * 2.0 + 0.25 * np.linalg.norm(wps[1] - wps[0]) * 2.0)


def lidar_to_ros_ego(preds: np.ndarray) -> np.ndarray:
    """ORION's (6, 2) ego_fut_preds live in the LIDAR_TOP frame: index 1 is
    forward, index 0 is RIGHT (LIDAR2EGO maps lidar +y -> ego +x).  ROS ego is
    x forward, y LEFT: (x, y) = (p[1], -p[0])."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1, 2)
    return np.stack([p[:, 1], -p[:, 0]], axis=1)


def extend_plan(pts_ego: np.ndarray, extend_sec: float, dt: float = ORION_TRAJ_DT) -> np.ndarray:
    """Prepend the ego origin and extrapolate the last segment's velocity for
    ``extend_sec`` beyond the horizon.  A ~1 s forward pass eats a third of the
    3 s plan before it arrives and another third before the next one lands;
    without the extension the controller runs dry at the end of every cycle
    (the reference node does the same in ``_rebase_plan``)."""
    pts = np.vstack([np.zeros((1, 2)), np.asarray(pts_ego, dtype=np.float64).reshape(-1, 2)])
    n_extra = int(round(max(extend_sec, 0.0) / dt))
    if n_extra > 0 and len(pts) >= 2:
        v = pts[-1] - pts[-2]
        extra = pts[-1] + np.arange(1, n_extra + 1)[:, None] * v[None, :]
        pts = np.vstack([pts, extra])
    return pts


def command_from_route(route: np.ndarray, idx: int, lookahead_m: float, turn_deg: float,
                       loop: bool = False) -> int:
    """A RoadOption from the route geometry ahead (no road options in a CSV route).

    Heading change between the route direction at ``idx`` and the direction
    ``lookahead_m`` further along (model metres): beyond ``turn_deg`` to the
    left -> LEFT (1), to the right -> RIGHT (2), otherwise LANEFOLLOW (4).
    In CARLA only junction turns carry LEFT/RIGHT; bends in a road are
    LANEFOLLOW, so keep ``turn_deg`` generous.
    """
    route = np.asarray(route, dtype=np.float64).reshape(-1, 2)
    n = len(route)
    if n < 3:
        return CMD_FOLLOW

    def wp(i):
        return route[i % n] if loop else route[min(i, n - 1)]

    d0 = wp(idx + 1) - wp(idx)
    if np.linalg.norm(d0) < 1e-6:
        return CMD_FOLLOW
    acc, i = 0.0, idx + 1
    while acc < lookahead_m and (loop or i < n - 1) and i - idx < n:
        acc += float(np.linalg.norm(wp(i + 1) - wp(i)))
        i += 1
    d1 = wp(i) - wp(i - 1)
    if np.linalg.norm(d1) < 1e-6:
        return CMD_FOLLOW
    a = math.degrees(math.atan2(d0[0] * d1[1] - d0[1] * d1[0], d0[0] * d1[0] + d0[1] * d1[1]))
    if a > turn_deg:
        return CMD_LEFT
    if a < -turn_deg:
        return CMD_RIGHT
    return CMD_FOLLOW


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    preds: np.ndarray          # [6, 2] LIDAR_TOP frame (index 0 right, index 1 forward), model metres
    text: str                  # VQA / CoT output (empty on the planning-only agent config)
    forward_ms: float

    @property
    def route_ros_ego(self) -> np.ndarray:
        """[6, 2] ROS ego frame (x forward, y left), model metres."""
        return lidar_to_ros_ego(self.preds)

    @property
    def desired_speed(self) -> float:
        return desired_speed_from_waypoints(self.preds)


class OrionRunner:
    """Loads ORION once and turns (six views, ego state, command) into waypoints."""

    def __init__(self, repo_path: str, config_path: str, checkpoint_path: str,
                 precision: str = "fp16", profile: str = "fast",
                 overrides: Optional[Dict] = None, llm_int8_stats: str = "",
                 decode_workers: int = 6, profile_stages: bool = False,
                 logger: Optional[logging.Logger] = None) -> None:
        if profile not in SPEED_PROFILES:
            raise ValueError(f"profile must be one of {sorted(SPEED_PROFILES)}, got {profile!r}")
        self.repo_path = repo_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.precision = precision
        self.profile = profile
        self.opts = dict(SPEED_PROFILES[profile])
        self.opts.update({k: v for k, v in (overrides or {}).items() if v is not None})
        self.llm_int8_stats = llm_int8_stats
        self.decode_workers = int(decode_workers)
        self.profile_stages = bool(profile_stages)
        self.log = logger or logging.getLogger("orion_runner")

        self._model = None
        self._pipeline = None
        self._collate = None
        self._get_box_type = None
        self._device = None
        self._stagger = None
        self._stage_timer = None
        self.vit_input_size = int(self.opts["vit_input_size"])

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ── loading (worker thread) ─────────────────────────────────────────────

    def load(self) -> None:
        """OrionWithPidRosNode._setup_model, minus ROS."""
        import torch

        if self.repo_path and self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        # The configs use RELATIVE paths ('ckpts/pretrain_qformer/') for the LLM
        # weights AND the pipeline tokenizer, resolved against the CWD.
        if self.repo_path:
            os.chdir(self.repo_path)

        from mmcv import Config
        from mmcv.core.bbox import get_box_type
        from mmcv.datasets.pipelines import Compose
        from mmcv.models import build_model
        from mmcv.parallel.collate import collate as mm_collate_to_batch_form
        from mmcv.utils import load_checkpoint
        from orion_ros import orion_speedups

        self._collate = mm_collate_to_batch_form
        self._get_box_type = get_box_type
        self._device = torch.device("cuda")

        cfg = Config.fromfile(self.config_path)
        precision = (self.precision or "").lower()
        if precision == "fp16":
            cfg.model["fp16_infer"] = True
            cfg.model["fp16_eval"] = False
            cfg.model["fp32_infer"] = False
        elif precision == "fp32":
            cfg.model["fp16_infer"] = False
            cfg.model["fp16_eval"] = False
            cfg.model["fp32_infer"] = True
        self.log.info(f"ORION {self.config_path}, precision {precision or 'config default'}, profile {self.profile}")
        if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
            import importlib
            importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))

        t0 = time.time()
        model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, self.checkpoint_path, map_location="cpu")
        model.cuda()
        model.eval()
        self.log.info(f"model built and weights loaded in {time.time() - t0:.0f} s")

        # Post-build speedups, in the reference node's order (LoRA before
        # compile; stage timer last).
        o, info = self.opts, self.log.info
        compile_targets = [t.strip() for t in str(o["compile_targets"]).split(",") if t.strip()]
        if o["merge_lora"]:
            orion_speedups.merge_lora(model, info)
        if o["llm_flash_attn"]:
            orion_speedups.patch_llm_flash_attention(model, info)
        if o["vit_glue"]:
            orion_speedups.patch_vit_blocks(model, info)
        if self.vit_input_size != 640:
            orion_speedups.set_vit_input_size(model, self.vit_input_size, info)
        if o["llm_int8"]:
            orion_speedups.apply_llm_int8(model, self.llm_int8_stats, LLM_INT8_ALPHA,
                                          LLM_INT8_SKIP_LAYERS, info, LLM_INT8_TARGETS)
            if o["llm_down_proj_transpose"] and "down_proj" not in LLM_INT8_TARGETS:
                orion_speedups.transpose_llm_down_proj(model, info)
        elif o["llm_down_proj_transpose"]:
            orion_speedups.transpose_llm_down_proj(model, info)
        if compile_targets:
            orion_speedups.compile_submodules(model, compile_targets, str(o["compile_mode"]), info)
        if o["map_head_slice"]:
            orion_speedups.slice_map_head_one2one(model, info)
        if int(o["rear_view_refresh_every"]) > 1:
            self._stagger = orion_speedups.install_staggered_views(model, int(o["rear_view_refresh_every"]), info)
        custom_wrap_fp16_model(model)
        if self.profile_stages:
            self._stage_timer = orion_speedups.StageTimer()
            self._stage_timer.wrap_orion(model)
        self._model = model

        # The agent's pipeline minus the disk loader (images are in memory).
        pipeline_cfg = [t for t in cfg.inference_only_pipeline
                        if t["type"] not in ("LoadMultiViewImageFromFilesInCeph",)]
        if self.vit_input_size != 640:
            orion_speedups.set_pipeline_input_size(pipeline_cfg, self.vit_input_size)
        self._pipeline = Compose(pipeline_cfg)
        orion_speedups.parallelize_pipeline(self._pipeline, workers=self.decode_workers, log=info)

        self._warmup()
        self.log.info("ORION model loaded and ready.")

    def _warmup(self) -> None:
        """One real forward on black frames: CUDA/cuDNN warm-up and, for the
        compiled profiles, the torch.compile of ViT/LLM/heads (minutes)."""
        import torch

        t0 = time.time()
        front = np.zeros((TRAIN_CAM_H, TRAIN_CAM_W, 3), dtype=np.uint8)
        can_bus, ego_pose, ego_pose_inv, l2g = build_can_bus((0.0, 0.0), 0.0, 0.0)
        batch = self.build_batch(make_views(front, "copy"), can_bus, ego_pose, ego_pose_inv, l2g,
                                 CMD_FOLLOW, scene_token=" ", frame_idx=0, timestamp=0.0)
        with torch.inference_mode():
            self._model(batch, return_loss=False)
            if self._stagger is not None:          # second forward = the front-only ViT shape
                self._model(batch, return_loss=False)
        torch.cuda.synchronize()
        if self._stagger is not None:
            self._stagger.reset()
        # The warm-up left black frames in the temporal memory: let the model's
        # own forward_test reset both heads before the first real frame.
        self.reset_memory()
        self.log.info(f"warm-up forward done in {time.time() - t0:.0f} s")

    # ── per-frame (any thread) ──────────────────────────────────────────────

    def build_batch(self, views: List[np.ndarray], can_bus: np.ndarray, ego_pose: np.ndarray,
                    ego_pose_inv: np.ndarray, lidar2global: np.ndarray, command: int,
                    scene_token: str, frame_idx: int, timestamp: float):
        """OrionAgent.run_step's ``results`` dict -> pipeline -> collate -> GPU."""
        import torch

        results: dict = {
            "lidar2img": [], "lidar2cam": [], "cam_intrinsic": [], "img": list(views),
            "folder": " ", "scene_token": scene_token, "frame_idx": int(frame_idx),
            "timestamp": float(timestamp),
        }
        results["box_type_3d"], _ = self._get_box_type("LiDAR")
        for cam in ORION_CAMERA_ORDER:
            results["lidar2img"].append(LIDAR2IMG[cam])
            results["lidar2cam"].append(LIDAR2CAM[cam])
            results["cam_intrinsic"].append(np.matmul(LIDAR2IMG[cam], np.linalg.inv(LIDAR2CAM[cam])))
        results["lidar2img"] = np.stack(results["lidar2img"], axis=0)
        results["lidar2cam"] = np.stack(results["lidar2cam"], axis=0)
        results["can_bus"] = can_bus
        results["command"] = command2nohot(int(command))
        results["ego_fut_cmd"] = command2hot(int(command))
        results["ego_pose"] = ego_pose
        results["ego_pose_inv"] = ego_pose_inv
        results["lidar2ego"] = LIDAR2EGO
        results["l2g_r_mat"] = lidar2global[0:3, 0:3]
        results["l2g_t"] = lidar2global[0:3, 3]
        stacked = np.stack(results["img"], axis=-1)
        results["img_shape"] = results["ori_shape"] = results["pad_shape"] = stacked.shape

        results = self._pipeline(results)
        batch = self._collate([results], samples_per_gpu=1)
        for key, data in batch.items():
            if key != "img_metas":
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self._device)
            if key == "input_ids":
                for i in range(len(data[0])):
                    for k in range(len(data[0][i])):
                        data[0][i][k] = data[0][i][k].to(self._device)
        return batch

    # ── inference (loader thread only) ──────────────────────────────────────

    def infer(self, batch) -> InferenceResult:
        import torch

        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self._model(batch, return_loss=False)
        torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - t0) * 1000.0
        out = output[0]
        preds = out["pts_bbox"]["ego_fut_preds"].cpu().numpy().astype(np.float32)
        if preds.shape != (ORION_NUM_WAYPOINTS, 2):
            raise RuntimeError(f"ego_fut_preds has shape {preds.shape}, expected (6, 2)")
        text = self._extract_text(out.get("text_out"))
        if self._stage_timer is not None:
            self.log.info("ORION stages (ms): " + self._stage_timer.format(self._stage_timer.report(), forward_ms))
        return InferenceResult(preds=preds, text=text, forward_ms=forward_ms)

    def reset_memory(self) -> None:
        """Make the next forward_test drop both heads' temporal memory (new route)."""
        if self._stagger is not None:
            self._stagger.reset()
        if self._model is not None and hasattr(self._model, "test_flag"):
            self._model.test_flag = False

    @staticmethod
    def _extract_text(text_out) -> str:
        if not text_out:
            return ""
        parts = []
        for qa in text_out:
            try:
                q, a = qa.get("Q"), qa.get("A")
                a = a[0] if isinstance(a, (list, tuple)) and a else a
                if a:
                    parts.append(f"Q: {q}\nA: {a}" if q else str(a))
            except AttributeError:
                continue
        return "\n\n".join(parts)


class FakeOrionRunner(OrionRunner):
    """No ORION at all: the same interface, a straight-ahead plan at a fixed speed.

    For exercising camera, route, scaling, publishing and the controller without
    the 52 GB of weights (node parameter ``fake_model``).  The six views still
    go through ``make_views`` so the image path is covered; the batch is a dict.
    """

    FAKE_SPEED_MPS = 5.0          # model m/s (0.5 m/s real at world_scale 10)
    FAKE_FORWARD_SEC = 0.3

    def load(self) -> None:
        self._model = object()
        self.log.info("FAKE ORION: straight plans, no model loaded")

    def build_batch(self, views, can_bus, ego_pose, ego_pose_inv, lidar2global, command,
                    scene_token, frame_idx, timestamp):
        if len(views) != 6:
            raise ValueError("six views expected")
        return {"can_bus": np.asarray(can_bus), "command": int(command), "frame_idx": int(frame_idx),
                "timestamp": float(timestamp), "img_shape": np.stack(views, axis=-1).shape}

    def infer(self, batch) -> InferenceResult:
        time.sleep(self.FAKE_FORWARD_SEC)
        step = self.FAKE_SPEED_MPS * ORION_TRAJ_DT
        # LIDAR_TOP frame: index 1 forward, index 0 right; a gentle right bend
        preds = np.array([[0.02 * (i + 1) ** 2, step * (i + 1)] for i in range(ORION_NUM_WAYPOINTS)], np.float32)
        return InferenceResult(preds=preds, text="", forward_ms=self.FAKE_FORWARD_SEC * 1000.0)

    def reset_memory(self) -> None:
        pass
