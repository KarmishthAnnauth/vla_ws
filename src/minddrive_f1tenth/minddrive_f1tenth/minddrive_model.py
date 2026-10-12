#!/usr/bin/env python3
"""MindDrive model runner, free of ROS and of CARLA.

Everything model-specific from ``minddrive_ros/minddrive_node.py`` lives here:
config + checkpoint loading, the post-build speedups, the agent's ``results``
dict, MindDrive's own ``inference_only_pipeline`` + ``collate``, and the forward
pass.  The ROS node only gathers sensor data and converts frames.  The MindDrive
counterpart of ``orion_f1tenth/orion_model.py``; the frame helpers are the same
code, because MindDrive's sensor rig IS ORION's (``MinddriveAgent.sensors()``).

The payload is prepared EXACTLY as the reference closed-loop agent does it
(``MindDrive/team_code/minddrive_b2d_agent.py::MinddriveAgent.run_step``):
``minddrive_ros.agent_constants.build_agent_results`` (the ``results`` dict
field for field, the hard-coded calibration), then MindDrive's
``Compose(cfg.inference_only_pipeline)`` (resize/crop, normalise, pad, VQA
prompt + tokenisation), ``mmcv.parallel.collate`` and the agent's H2D loop.
Nothing of the preprocessing is reimplemented.  Only the raw inputs differ:

  * one real camera instead of six CARLA cameras.  The front frame is made to
    look like ``CAM_FRONT`` (1600x900, 70 deg horizontal FOV) and is also
    handed to the two front-side slots (``side_views="copy"``); the three rear
    slots are black.  The six calibration matrices stay the agent's.
  * pose/speed from the F1TENTH stack instead of GNSS/IMU/speedometer, already
    scaled into the model's world (see ``world_scale`` in the node).

Outputs.  MindDrive predicts TWO trajectories per frame, both in its LIDAR_TOP
frame (index 0 right, index 1 forward), plus the decision expert's two discrete
meta-actions:

  ego_fut_preds      (6, 2)   the SPEED trajectory: 0.5 s steps, 3 s horizon.
                              The agent's PID reads its desired speed from it.
  pw_ego_fut_pred    (20, 2)  the PATH trajectory: a polyline along the intended
                              path with no time axis.  The agent's PID steers on it.
  speed_value / path_value    the chosen meta-actions (SPEED_ACTIONS / PATH_ACTIONS).

Threading contract: ``load()`` and every ``infer()`` must run in the *same* OS
thread (Jetson iGPU cuBLAS/cuDNN handles are per thread; the CUDA graphs too).
``build_batch()`` is CPU work plus a host-to-device copy and may run on another
thread, as the reference node's prep worker does.

Speed profiles (``profile``) bundle the speedups of ``minddrive_speedups.py``,
all post-build transforms on the model object (MindDrive sources untouched),
with the numbers of MINDDRIVE_ROS_NODE.md section 6 (3B, six real views):

  baseline  the reference config as is: fp32, no speedup            (~2.3 s forward)
  eager     fp16_infer + the exact set (merged LoRA, ViT glue, weight
            layouts, unpadded window attention, map-head slice, one-row
            decision logits), no torch.compile                       (~1.4 s)
  fast      eager + torch.compile of ViT and LLM + CUDA-graphed ViT:
            start_minddrive.sh --fast, trajectories within 0.1 model m
            (1 cm real at scale 10) of the fp32 reference             (~0.87 s)
  int8      fast + W8A8 INT8 LLM projections (SmoothQuant alpha 0.5,
            stats in /benchmarking/minddrive_int8_calib.pt): 0.4-0.5
            model m trajectory shift -- a precision change            (~0.82 s)
  lite      fast + 512 px ViT input, rear views every other frame:
            1.8-2.7 model m shift, a different model in effect        (~0.65 s)
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

from minddrive_ros.agent_constants import (   # the model's input contract, verbatim upstream copies
    AGENT_HZ, CAMERA_ORDER, FUT_PS, FUT_TS, LIDAR2EGO, LIDAR2IMG, MEMORY_MAX_DT, ROUTE_MAX_DIST,
    ROUTE_MIN_DIST, TRAJ_DT, batch_to_device, build_agent_results, custom_wrap_fp16_model,
)

# ---------------------------------------------------------------------------
# Constants of the checkpoint (MinddriveAgent.sensors() / tick()).  Fixed
# properties of the model, not of the physical camera.
# ---------------------------------------------------------------------------
FRONT_SLOTS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
REAR_SLOTS = ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

TRAIN_CAM_W = 1600
TRAIN_CAM_H = 900
TRAIN_FRONT_FOV_DEG = 70.0          # CAM_FRONT / sides; CAM_BACK is 110
# The agent re-encodes every camera frame at JPEG quality 20 (MinddriveAgent.tick).
AGENT_JPEG_QUALITY = 20
# Speed trajectory: 6 waypoints at 2 Hz (TRAJ_DT, FUT_TS); path trajectory: 20 points (FUT_PS).
MINDDRIVE_TRAJ_DT = TRAJ_DT
MINDDRIVE_NUM_SPEED_WPS = FUT_TS
MINDDRIVE_NUM_PATH_PTS = FUT_PS
# MinddriveHead keeps its temporal memory only while consecutive timestamps are
# < 2 s apart and the scene token is unchanged (same rule as ORION).
MINDDRIVE_MEMORY_MAX_DT = MEMORY_MAX_DT
# MinddriveAgent: results['timestamp'] = step / 20 (synchronous 20 Hz CARLA).
MINDDRIVE_AGENT_HZ = AGENT_HZ
# RoadOption codes fed as ego_fut_cmd (command2hot): 1=left 2=right 3=straight
# 4=follow-lane 5=change-left 6=change-right.
CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT, CMD_FOLLOW = 1, 2, 3, 4
CMD_NAMES = {1: "LEFT", 2: "RIGHT", 3: "STRAIGHT", 4: "LANEFOLLOW", 5: "CHANGELEFT", 6: "CHANGERIGHT"}

# The decision expert's discrete meta-actions, indexed by the `speed_value` /
# `path_value` ints the detector returns (mmcv/models/detectors/minddrive.py
# SPEED_MAPPING / PATH_MAPPING).  Diagnostics only.
SPEED_ACTIONS = ["maintain moderate speed", "stop", "maintain slow speed", "speed up",
                 "slow down", "maintain fast speed", "slow down rapidly"]
PATH_ACTIONS = ["lanefollow", "straight", "turn left", "change lane left",
                "turn right", "change lane right"]


def action_name(table: List[str], idx) -> str:
    try:
        return table[int(idx)]
    except (TypeError, ValueError, IndexError):
        return f"?{idx}"


# Which weights (container paths).  The 3B model is the paper's headline number
# (80.59 DS); the 0.5B is the smaller variant (78.04 DS).  Config paths are
# RELATIVE to the checkout, which the runner makes the CWD, exactly like the
# agent: the configs resolve './Bench2DriveZoo/ckpts/<llm dir>' against it.
VARIANTS: Dict[str, Dict[str, str]] = {
    "3b": dict(config="adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py",
               checkpoint="/models/MindDrive/minddrive_3b_rltrain.pth",
               llm_dir="llava-qwen2.5-3b"),
    "05b": dict(config="adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py",
                checkpoint="/models/MindDrive/minddrive_rltrain.pth",
                llm_dir="llava-qwen2-0.5b"),
}
DEFAULT_INT8_CALIB = "/benchmarking/minddrive_int8_calib.pt"


# ---------------------------------------------------------------------------
# Speed profiles: the minddrive_ros node parameters each one sets, plus the
# precision.  Everything not listed keeps the minddrive_ros default.
# ---------------------------------------------------------------------------
_OFF: Dict[str, object] = dict(
    precision="config", merge_lora=False, llm_attn="keep", vit_glue=False, down_proj_t=False,
    vit_weight_t=False, vit_window_nopad=False, int8_targets="", int8_skip="", int8_calib_path="",
    int8_alpha=0.5, vit_input_size=640, rear_view_refresh_every=1, map_head_slice=False,
    compile_targets="", compile_mode="default", cuda_graph_vit=False, cuda_graph_heads=False,
    logits_slice=False, overlap_experts=False,
)
# The bench-validated exact set (MINDDRIVE_ROS_NODE.md section 6): same weights,
# same math, laid out or fused differently.
_EXACT: Dict[str, object] = dict(
    _OFF, precision="fp16", merge_lora=True, vit_glue=True, down_proj_t=True, vit_weight_t=True,
    vit_window_nopad=True, map_head_slice=True, logits_slice=True,
)
SPEED_PROFILES: Dict[str, Dict[str, object]] = {
    "baseline": dict(_OFF),
    "eager": dict(_EXACT),
    "fast": dict(_EXACT, compile_targets="llm,vit", cuda_graph_vit=True),
    "int8": dict(_EXACT, compile_targets="llm,vit", cuda_graph_vit=True,
                 int8_targets="llm", int8_calib_path=DEFAULT_INT8_CALIB, int8_alpha=0.5),
    "lite": dict(_EXACT, compile_targets="llm,vit", cuda_graph_vit=True,
                 vit_input_size=512, rear_view_refresh_every=2),
}


def parse_overrides(spec: str) -> Dict[str, object]:
    """``"key=value,key=value"`` -> typed speedup options (types from the profile template).

    The same knobs start_minddrive.sh forwards as ``key:=value`` to the CARLA node,
    e.g. ``compile_targets=,cuda_graph_vit=false`` turns the fast profile eager.
    """
    out: Dict[str, object] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"speedup override {item!r} is not key=value")
        key, val = (s.strip() for s in item.split("=", 1))
        if key not in _OFF:
            raise ValueError(f"unknown speedup option {key!r}; known: {sorted(_OFF)}")
        template = _OFF[key]
        if isinstance(template, bool):
            if val.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{key} needs true/false, got {val!r}")
            out[key] = val.lower() in ("true", "1")
        elif isinstance(template, int):
            out[key] = int(val)
        elif isinstance(template, float):
            out[key] = float(val)
        else:
            out[key] = val
    return out


def _csv_list(v) -> List[str]:
    if isinstance(v, (list, tuple)):
        return [str(t).strip() for t in v if str(t).strip()]
    return [t.strip() for t in str(v or "").split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Camera frame formatting (any thread) -- same as orion_f1tenth.orion_model
# ---------------------------------------------------------------------------
def crop_to_hfov(bgr: np.ndarray, camera_hfov_deg: float, target_hfov_deg: float = TRAIN_FRONT_FOV_DEG) -> np.ndarray:
    """Keep the central ``target_hfov_deg`` of a pinhole frame with ``camera_hfov_deg``.

    A 110 deg webcam frame fed whole into a 70 deg camera slot shows the model a
    scene whose objects are 1.9x too small and far; cropping the central sector
    makes the pixels match the intrinsics the model was trained with (what the
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
    """Make a real camera frame look like MindDrive's CAM_FRONT.

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
    """MinddriveAgent.tick: ``imencode('.jpg', img, quality 20)`` then ``imdecode``."""
    if quality <= 0:
        return bgr
    _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_views(front_bgr: np.ndarray, side_views: str = "copy") -> Dict[str, np.ndarray]:
    """The six camera slots, keyed by CAMERA_ORDER name, from one front frame.

    side_views="copy":  front frame in CAM_FRONT, CAM_FRONT_LEFT and
                        CAM_FRONT_RIGHT (the model is told the same pixels sit
                        at yaw 0 / -55 / +55 deg), rear slots black.
    side_views="black": front frame in CAM_FRONT only, five black slots.
    """
    if front_bgr.shape[:2] != (TRAIN_CAM_H, TRAIN_CAM_W):
        raise ValueError(f"front frame must be {TRAIN_CAM_W}x{TRAIN_CAM_H}, got {front_bgr.shape[1]}x{front_bgr.shape[0]}")
    black = np.zeros_like(front_bgr)
    if side_views == "copy":
        return {cam: (front_bgr if cam in FRONT_SLOTS else black) for cam in CAMERA_ORDER}
    if side_views == "black":
        return {cam: (front_bgr if cam == "CAM_FRONT" else black) for cam in CAMERA_ORDER}
    raise ValueError(f"side_views must be 'copy' or 'black', got {side_views!r}")


# ---------------------------------------------------------------------------
# Ego state -> can_bus, as MinddriveAgent.run_step builds it
# ---------------------------------------------------------------------------
def build_can_bus(pos_xy: Sequence[float], ego_theta: float, speed: float,
                  acceleration: Sequence[float] = (0.0, 0.0, 0.0),
                  angular_velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """The agent's 18-dim can_bus (ego_pose and friends come from build_agent_results).

    Inputs are in the agent's FINAL conventions, i.e. after its CARLA->ROS
    conversion: a right-handed world (x, y) with ``ego_theta`` the
    counter-clockwise yaw from +x (agent: ``-compass + pi/2``), body-frame
    acceleration with y to the LEFT (agent: ``can_bus[11] *= -1``) and angular
    velocity with z counter-clockwise (agent: ``-angular_velocity`` over CARLA's
    left-handed gyro).  A ROS map pose and ROS body twists go in unchanged; the
    position is the point the agent localises from (its GNSS at x = -1.4 m in
    the vehicle frame, i.e. about the rear axle -- the F1TENTH's base_link).
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
    return can_bus


# ---------------------------------------------------------------------------
# Model output -> plan geometry
# ---------------------------------------------------------------------------
def desired_speed_from_waypoints(wps: np.ndarray) -> float:
    """Scalar target speed exactly as pid_controller_de.control_pid derives it from
    the SPEED trajectory: 0.75 * |wp0| * 2 + 0.25 * |wp1 - wp0| * 2  (0.5 s per waypoint -> m/s)."""
    wps = np.asarray(wps, dtype=np.float64).reshape(-1, 2)
    return float(0.75 * np.linalg.norm(wps[0]) * 2.0 + 0.25 * np.linalg.norm(wps[1] - wps[0]) * 2.0)


def lidar_to_ros_ego(preds: np.ndarray) -> np.ndarray:
    """MindDrive's (N, 2) trajectories live in the LIDAR_TOP frame: index 1 is
    forward, index 0 is RIGHT (LIDAR2EGO maps lidar +y -> ego +x; control_pid's
    ``angle`` is positive for aim[0] > 0, a right turn).  ROS ego is x forward,
    y LEFT: (x, y) = (p[1], -p[0])."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1, 2)
    return np.stack([p[:, 1], -p[:, 0]], axis=1)


def extend_plan(pts_ego: np.ndarray, extend_sec: float, dt: float = TRAJ_DT) -> np.ndarray:
    """SPEED trajectory: prepend the ego origin and extrapolate the last segment's
    velocity for ``extend_sec`` beyond the 3 s horizon.  A ~1 s forward pass eats a
    third of the plan before it arrives and another third before the next one
    lands; without the extension the controller runs dry at the end of every
    cycle (the reference node does the same in ``_rebase_speed_traj``)."""
    pts = np.vstack([np.zeros((1, 2)), np.asarray(pts_ego, dtype=np.float64).reshape(-1, 2)])
    n_extra = int(round(max(extend_sec, 0.0) / dt))
    if n_extra > 0 and len(pts) >= 2:
        v = pts[-1] - pts[-2]
        extra = pts[-1] + np.arange(1, n_extra + 1)[:, None] * v[None, :]
        pts = np.vstack([pts, extra])
    return pts


def extend_path(pts_ego: np.ndarray, extend_m: float, max_extra: int = 40) -> np.ndarray:
    """PATH trajectory: prepend the ego origin and continue the last segment's
    direction for ``extend_m`` metres at the polyline's own spacing (the spatial
    analogue of ``extend_plan``; the reference node's ``_rebase_path`` extends the
    same way).  A zero-length path (the model wants to stop) is returned as is:
    every point at the ego, which the controller reads as "plan exhausted" = stop."""
    path = np.asarray(pts_ego, dtype=np.float64).reshape(-1, 2)
    pts = np.vstack([np.zeros((1, 2)), path])
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    total = float(seglen.sum())
    if extend_m <= 0.0 or total <= 1e-6 or len(path) < 1:
        return pts
    nz = np.nonzero(seglen > 1e-9)[0]
    u_end = seg[nz[-1]] / seglen[nz[-1]]
    ds = total / len(path)
    n_extra = min(max_extra, int(math.ceil(extend_m / ds)))
    extra = pts[-1] + (np.arange(1, n_extra + 1)[:, None] * ds) * u_end[None, :]
    return np.vstack([pts, extra])


def path_times(pts_ego: np.ndarray, speed_mps: float, dt_fallback: float = TRAJ_DT) -> np.ndarray:
    """time_from_start for a path with no time axis: arc length / speed, or
    ``dt_fallback`` per point when the speed is ~0 (the controller ignores it)."""
    pts = np.asarray(pts_ego, dtype=np.float64).reshape(-1, 2)
    if speed_mps <= 0.05:
        return np.arange(len(pts)) * dt_fallback
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return arc / speed_mps


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
    speed_preds: np.ndarray    # [6, 2]  LIDAR_TOP frame (index 0 right, index 1 forward), model metres
    path_preds: np.ndarray     # [20, 2] LIDAR_TOP frame, model metres, no time axis
    speed_value: object        # decision expert's speed meta-action index (or None)
    path_value: object         # decision expert's path meta-action index (or None)
    forward_ms: float

    @property
    def speed_ros_ego(self) -> np.ndarray:
        """[6, 2] ROS ego frame (x forward, y left), model metres."""
        return lidar_to_ros_ego(self.speed_preds)

    @property
    def path_ros_ego(self) -> np.ndarray:
        """[20, 2] ROS ego frame (x forward, y left), model metres."""
        return lidar_to_ros_ego(self.path_preds)

    @property
    def desired_speed(self) -> float:
        return desired_speed_from_waypoints(self.speed_preds)

    @property
    def speed_action(self) -> str:
        return action_name(SPEED_ACTIONS, self.speed_value)

    @property
    def path_action(self) -> str:
        return action_name(PATH_ACTIONS, self.path_value)

    @property
    def text(self) -> str:
        return f"speed={self.speed_action} path={self.path_action}"


class MinddriveRunner:
    """Loads MindDrive once and turns (six views, ego state, command) into the two trajectories."""

    def __init__(self, repo_path: str, config_path: str, checkpoint_path: str,
                 profile: str = "fast", overrides: Optional[Dict[str, object]] = None,
                 decode_workers: int = 6, profile_stages: bool = False,
                 logger: Optional[logging.Logger] = None) -> None:
        if profile not in SPEED_PROFILES:
            raise ValueError(f"profile must be one of {sorted(SPEED_PROFILES)}, got {profile!r}")
        self.repo_path = repo_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.profile = profile
        self.opts: Dict[str, object] = dict(SPEED_PROFILES[profile])
        self.opts.update({k: v for k, v in (overrides or {}).items() if v is not None})
        self.precision = str(self.opts["precision"]).lower()
        if self.precision not in ("config", "fp32", "fp16"):
            raise ValueError("precision must be 'config', 'fp32' or 'fp16'")
        self.decode_workers = max(1, int(decode_workers))
        self.profile_stages = bool(profile_stages)
        self.log = logger or logging.getLogger("minddrive_runner")

        self._model = None
        self._pipeline = None
        self._collate = None
        self._get_box_type = None
        self._device = None
        self._sp = None                    # minddrive_ros.minddrive_speedups, imported in load()
        self._stage_timer = None
        self.vit_input_size = int(self.opts["vit_input_size"])
        self.overlap_experts = bool(self.opts["overlap_experts"])
        self.compiled = bool(_csv_list(self.opts["compile_targets"]) or self.opts["cuda_graph_vit"]
                             or self.opts["cuda_graph_heads"])

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def speedup_opts(self) -> Dict[str, object]:
        """The options dict ``minddrive_speedups.apply_speedups`` takes."""
        o = dict(self.opts)
        o.pop("precision", None)
        o["int8_targets"] = _csv_list(o["int8_targets"])
        o["int8_skip"] = _csv_list(o["int8_skip"])
        o["compile_targets"] = _csv_list(o["compile_targets"])
        return o

    # ── loading (worker thread) ─────────────────────────────────────────────

    def load(self) -> None:
        """MinddriveRosNode._setup_model, minus ROS."""
        import torch

        if self.repo_path and self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        # The configs use RELATIVE paths ('./Bench2DriveZoo/ckpts/...') for the
        # LLM weights AND the pipeline tokenizer, resolved against the CWD.
        if self.repo_path:
            os.chdir(self.repo_path)

        import mmcv as _mmcv
        from mmcv import Config
        from mmcv.core.bbox import get_box_type
        from mmcv.datasets.pipelines import Compose
        from mmcv.models import build_model
        from mmcv.parallel.collate import collate as mm_collate_to_batch_form
        from mmcv.utils import load_checkpoint
        from minddrive_ros import minddrive_speedups as sp

        self._sp = sp
        self._collate = mm_collate_to_batch_form
        self._get_box_type = get_box_type
        self._device = torch.device("cuda")
        mmcv_dir = os.path.dirname(os.path.abspath(_mmcv.__file__))
        if self.repo_path and not mmcv_dir.startswith(os.path.abspath(self.repo_path)):
            raise RuntimeError(f"mmcv resolves to {mmcv_dir}, not MindDrive's fork under {self.repo_path} "
                               "(PYTHONPATH must put the checkout first)")
        self.log.info(f"mmcv fork: {mmcv_dir}")

        cfg = Config.fromfile(self.config_path)
        if self.precision == "fp16":
            cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = True, False, False
        elif self.precision == "fp32":
            cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = False, False, True
        self.log.info(f"MindDrive {self.config_path}: lm_model_type={cfg.model.get('lm_model_type')} "
                      f"fp32_infer={cfg.model.get('fp32_infer')} fp16_infer={cfg.model.get('fp16_infer')}; "
                      f"checkpoint {self.checkpoint_path}; profile {self.profile} "
                      f"(precision {self.precision})")
        if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
            import importlib
            importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))

        t0 = time.time()
        model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
        self.log.info(f"build_model done in {time.time() - t0:.0f} s; loading checkpoint")
        t1 = time.time()
        load_checkpoint(model, self.checkpoint_path, map_location="cpu")
        self.log.info(f"load_checkpoint done in {time.time() - t1:.0f} s")
        model.cuda()
        model.eval()

        # Post-build speedups, in the order the bench validated them.  Applied
        # before the warm-up so the compile / capture cost is paid there.
        applied = sp.apply_speedups(model, self.speedup_opts(), self.log.info)
        self.log.info(f"speedups applied: {applied or 'none (reference model)'}")
        if self.profile_stages:
            self._stage_timer = sp.StageTimer()
            self._stage_timer.wrap_minddrive(model)
        self._model = model

        # The agent's pipeline minus the disk loader (images are in memory).
        pipeline_cfg = [t for t in cfg.inference_only_pipeline
                        if t["type"] not in ("LoadMultiViewImageFromFilesInCeph",)]
        if self.vit_input_size != 640 and sp.set_pipeline_input_size(pipeline_cfg, self.vit_input_size):
            self.log.info(f"pipeline: ViT input {self.vit_input_size}x{self.vit_input_size}")
        self._pipeline = Compose(pipeline_cfg)
        sp.parallelize_pipeline(self._pipeline, self.decode_workers, self.log.info)
        self.log.info(f"model on GPU ({torch.cuda.memory_allocated() / 1e9:.1f} GB allocated) "
                      f"after {time.time() - t0:.0f} s")

        self._warmup()
        self.log.info("MindDrive model loaded and ready.")

    def _warmup(self) -> None:
        """Real forwards on black frames: CUDA/cuDNN warm-up and, for the compiled
        profiles, the torch.compile of ViT/LLM and the CUDA-graph capture (minutes)."""
        import torch

        t0 = time.time()
        front = np.zeros((TRAIN_CAM_H, TRAIN_CAM_W, 3), dtype=np.uint8)
        can_bus = build_can_bus((0.0, 0.0), 0.0, 0.0)
        batch = self.build_batch(make_views(front, "copy"), can_bus, 0.0, CMD_FOLLOW,
                                 scene_token="warmup", frame_idx=0, timestamp=0.0)
        n_warm = 2 if self.compiled else 1
        with torch.no_grad():
            for _ in range(n_warm):
                self._sp.mark_step()
                custom_wrap_fp16_model(self._model)
                out = self._model(batch, return_loss=False)
                torch.cuda.synchronize()
        pb = out[0]["pts_bbox"]
        # The warm-up left black frames in the temporal memory: let the model's
        # own forward_test reset both heads before the first real frame.
        self.reset_memory()
        self.log.info(f"warm-up done in {time.time() - t0:.0f} s: speed traj {tuple(pb['ego_fut_preds'].shape)}, "
                      f"path traj {tuple(pb['pw_ego_fut_pred'].shape)}, "
                      f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # ── per-frame (any thread) ──────────────────────────────────────────────

    def build_batch(self, views: Dict[str, np.ndarray], can_bus: np.ndarray, ego_theta: float,
                    command: int, scene_token: str, frame_idx: int, timestamp: float):
        """MinddriveAgent.run_step's ``results`` dict -> pipeline -> collate -> GPU."""
        results = build_agent_results(views, can_bus, float(ego_theta), int(command), scene_token,
                                      int(frame_idx), float(timestamp), self._get_box_type)
        results = self._pipeline(results)
        batch = self._collate([results], samples_per_gpu=1)
        batch_to_device(batch, self._device)
        if self.overlap_experts:
            self._sp.reorder_rounds_for_overlap(batch, self._model)
        return batch

    # ── inference (loader thread only) ──────────────────────────────────────

    def infer(self, batch) -> InferenceResult:
        import torch

        t0 = time.perf_counter()
        if self._stage_timer is not None:
            self._stage_timer._pending.clear()
        with torch.no_grad():
            self._sp.mark_step()
            custom_wrap_fp16_model(self._model)        # the agent calls it before EVERY forward
            output = self._model(batch, return_loss=False)
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - t0) * 1000.0
        pb = output[0]["pts_bbox"]
        speed_preds = pb["ego_fut_preds"].cpu().numpy().astype(np.float32).reshape(-1, 2)
        path_preds = pb["pw_ego_fut_pred"].cpu().numpy().astype(np.float32).reshape(-1, 2)
        if speed_preds.shape != (MINDDRIVE_NUM_SPEED_WPS, 2):
            raise RuntimeError(f"ego_fut_preds has shape {speed_preds.shape}, expected ({MINDDRIVE_NUM_SPEED_WPS}, 2)")
        if path_preds.shape[0] < 2:
            raise RuntimeError(f"pw_ego_fut_pred has shape {path_preds.shape}, expected ({MINDDRIVE_NUM_PATH_PTS}, 2)")
        if self._stage_timer is not None:
            self.log.info("MindDrive stages (ms): "
                          + self._stage_timer.format(self._stage_timer.report(), forward_ms))
        return InferenceResult(speed_preds=speed_preds, path_preds=path_preds,
                               speed_value=pb.get("speed_value"), path_value=pb.get("path_value"),
                               forward_ms=forward_ms)

    def reset_memory(self) -> None:
        """Make the next forward_test drop both heads' temporal memory (new route)."""
        model = self._model
        if model is None:
            return
        if hasattr(model, "test_flag"):
            model.test_flag = False          # forward_test -> reset_memory() on both heads
        if getattr(model, "planning_memory", None) is not None:
            model.planning_memory = None
        stagger = getattr(getattr(model, "img_backbone", None), "_md_stagger", None)
        if stagger is not None:
            stagger.reset()


class FakeMinddriveRunner(MinddriveRunner):
    """No MindDrive at all: the same interface, a straight-ahead plan at a fixed speed.

    For exercising camera, route, scaling, publishing and the controller without
    the 28 GB of weights (node parameter ``fake_model``).  The six views still go
    through ``make_views`` so the image path is covered; the batch is a dict.
    """

    FAKE_SPEED_MPS = 5.0          # model m/s (0.5 m/s real at world_scale 10)
    FAKE_FORWARD_SEC = 0.3

    def load(self) -> None:
        self._model = object()
        self.log.info("FAKE MINDDRIVE: straight plans, no model loaded")

    def build_batch(self, views, can_bus, ego_theta, command, scene_token, frame_idx, timestamp):
        if set(views) != set(CAMERA_ORDER):
            raise ValueError("six views expected")
        return {"can_bus": np.asarray(can_bus), "command": int(command), "frame_idx": int(frame_idx),
                "timestamp": float(timestamp), "img_shape": np.stack([views[c] for c in CAMERA_ORDER], axis=-1).shape}

    def infer(self, batch) -> InferenceResult:
        time.sleep(self.FAKE_FORWARD_SEC)
        step = self.FAKE_SPEED_MPS * TRAJ_DT
        # LIDAR_TOP frame: index 1 forward, index 0 right; a gentle right bend
        speed = np.array([[0.02 * (i + 1) ** 2, step * (i + 1)] for i in range(MINDDRIVE_NUM_SPEED_WPS)], np.float32)
        reach = step * MINDDRIVE_NUM_SPEED_WPS
        path = np.array([[0.02 * ((i + 1) * MINDDRIVE_NUM_SPEED_WPS / MINDDRIVE_NUM_PATH_PTS) ** 2,
                          reach * (i + 1) / MINDDRIVE_NUM_PATH_PTS] for i in range(MINDDRIVE_NUM_PATH_PTS)], np.float32)
        return InferenceResult(speed_preds=speed, path_preds=path, speed_value=0, path_value=0,
                               forward_ms=self.FAKE_FORWARD_SEC * 1000.0)

    def reset_memory(self) -> None:
        pass
