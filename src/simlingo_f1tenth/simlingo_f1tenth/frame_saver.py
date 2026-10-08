#!/usr/bin/env python3
"""Saves each planned frame as a JPEG: the model's input image with its predicted
trajectory drawn in, and a text panel with the language output (the commentary
in thinking mode) and the plan's numbers.

The drawing follows the debug view of agent_simlingo.py: predicted route in red,
speed waypoints in green, target points in blue, all projected with the camera
geometry the checkpoint was trained with (110 deg FOV on the 1024x512 frame,
camera 1.5 m behind and 2.0 m above the ego origin).  That is the geometry the
real frame is formatted to imitate, so the overlay shows where the model *thinks*
it is pointing in its own image; it is not a calibration of the car's camera.
Upstream's project_points adds the 1.5 m camera offset twice; this does not.
The car's camera points further down than that, so in the image the trajectory
is squeezed into a short stub near the bottom; a top-down view to the right of
the image shows the same points in real metres around the car.

Writing happens on its own thread so the inference thread never waits for it.
If a backlog builds up (slow disk) frames are dropped, not queued.
"""

from __future__ import annotations

import math
import os
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import cv2
import numpy as np

from simlingo_f1tenth.simlingo_model import (
    TRAIN_CAM_EXTRINSIC_XYZ, TRAIN_CAM_FOV_DEG, TRAIN_IMAGE_H, TRAIN_IMAGE_W,
)

PANEL_H = 130
BEV_W = 320            # top-down view: pixels wide
BEV_PX_PER_M = 100.0   # real metres
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY = (255, 255, 255), (170, 170, 170)
# BGR
RED, GREEN, BLUE = (0, 0, 255), (0, 200, 0), (255, 80, 0)


def project_model_points(pts: np.ndarray, fov_deg: float = TRAIN_CAM_FOV_DEG) -> np.ndarray:
    """Model ego frame (x forward, y right, ground plane, model metres) -> pixels
    in the 1024x512 training frame (the bottom crop only removes rows)."""
    f = TRAIN_IMAGE_W / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx, cy = TRAIN_IMAGE_W / 2.0, TRAIN_IMAGE_H / 2.0
    cam_x, _, cam_z = TRAIN_CAM_EXTRINSIC_XYZ
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    depth = pts[:, 0] - cam_x
    out = np.full((len(pts), 2), np.nan)
    ok = depth > 0.1
    out[ok, 0] = cx + f * pts[ok, 1] / depth[ok]
    out[ok, 1] = cy + f * cam_z / depth[ok]
    return out


def render_bev(h: int, route: np.ndarray, speed_wps: np.ndarray, target_points: np.ndarray,
               world_scale: float) -> np.ndarray:
    """Top-down view, car at the bottom centre facing up, REAL metres, 0.5 m grid."""
    bev = np.full((h, BEV_W, 3), 30, np.uint8)
    ox, oy = BEV_W // 2, h - 20

    def px(pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2) / world_scale
        return [(int(round(ox + y * BEV_PX_PER_M)), int(round(oy - x * BEV_PX_PER_M))) for x, y in pts]

    step = int(BEV_PX_PER_M * 0.5)
    for gy in range(oy, -1, -step):
        cv2.line(bev, (0, gy), (BEV_W, gy), (55, 55, 55), 1)
        cv2.putText(bev, f"{(oy - gy) / BEV_PX_PER_M:.1f} m", (4, gy - 3), FONT, 0.35, GREY, 1, cv2.LINE_AA)
    for gx in range(ox % step, BEV_W, step):
        cv2.line(bev, (gx, 0), (gx, h), (55, 55, 55), 1)
    r = px(route)
    if len(r) > 1:
        cv2.polylines(bev, [np.array(r, np.int32)], False, RED, 2, cv2.LINE_AA)
    for p in r:
        cv2.circle(bev, p, 3, RED, -1, cv2.LINE_AA)
    for p in px(speed_wps):
        cv2.circle(bev, p, 3, GREEN, -1, cv2.LINE_AA)
    for p in px(target_points):
        cv2.circle(bev, p, 6, BLUE, -1, cv2.LINE_AA)
    # the car: 0.33 x 0.2 m, origin at base_link
    cv2.rectangle(bev, (ox - 10, oy - 5), (ox + 10, oy + 28), WHITE, 1)
    cv2.putText(bev, "top-down, real m", (BEV_W - 135, 16), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    return bev


def render(rgb: np.ndarray, route: np.ndarray, speed_wps: np.ndarray, target_points: np.ndarray,
           language: str, info_lines: List[str], world_scale: float = 10.0) -> np.ndarray:
    """Returns a BGR image: the frame with the overlay, the text panel below it."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    def dots(pts, color, r, line=False):
        px = project_model_points(pts)
        valid = [(int(round(u)), int(round(v))) for u, v in px
                 if np.isfinite(u) and -w < u < 2 * w and -h < v < 2 * h]
        if line and len(valid) > 1:
            cv2.polylines(img, [np.array(valid, np.int32)], False, color, 2, cv2.LINE_AA)
        for p in valid:
            cv2.circle(img, p, r, color, -1, cv2.LINE_AA)

    dots(route, RED, 3, line=True)
    dots(speed_wps, GREEN, 3)
    dots(target_points, BLUE, 6)
    img = np.hstack([img, render_bev(h, route, speed_wps, target_points, world_scale)])
    w = img.shape[1]

    wrapped = textwrap.wrap(language.strip() or "(no text)", width=125)
    panel_h = max(PANEL_H, 22 * (len(wrapped) + len(info_lines)) + 16)
    panel = np.zeros((panel_h, w, 3), np.uint8)
    y = 24
    for line in info_lines:
        cv2.putText(panel, line, (10, y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        y += 22
    for line in wrapped:
        cv2.putText(panel, line, (10, y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
        y += 22
    legend = [("route", RED), ("speed wps", GREEN), ("target pts", BLUE)]
    x = w - 300
    for name, color in legend:
        cv2.circle(panel, (x, 18), 5, color, -1, cv2.LINE_AA)
        cv2.putText(panel, name, (x + 10, 23), FONT, 0.45, GREY, 1, cv2.LINE_AA)
        x += 100
    return np.vstack([img, panel])


class FrameSaver:
    def __init__(self, out_dir: str, every: int = 1, jpeg_quality: int = 90, logger=None,
                 world_scale: float = 10.0) -> None:
        self.world_scale = float(world_scale)
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # The node runs as root in the container; files take the owner of the directory
        # (created by the route script as the host user) so they can be deleted on the host.
        st = self.dir.stat()
        self._owner = (st.st_uid, st.st_gid)
        self.every = max(1, int(every))
        self.quality = int(jpeg_quality)
        self.log = logger
        self._n = 0
        self._saved = 0
        self._pending = 0
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame_saver")

    def submit(self, plan_no: int, rgb: np.ndarray, route: np.ndarray, speed_wps: np.ndarray,
               target_points: np.ndarray, language: str, info_lines: List[str]) -> None:
        self._n += 1
        if (self._n - 1) % self.every:
            return
        with self._lock:
            if self._pending >= 3:
                return
            self._pending += 1
        self._pool.submit(self._write, plan_no, rgb, route, speed_wps, target_points, language, info_lines)

    def _write(self, plan_no, rgb, route, speed_wps, target_points, language, info_lines) -> None:
        try:
            out = render(rgb, route, speed_wps, target_points, language, info_lines, self.world_scale)
            path = self.dir / f"plan_{plan_no:05d}.jpg"
            cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            try:
                os.chown(path, *self._owner)
            except OSError:
                pass
            self._saved += 1
        except Exception as exc:                               # never take the node down
            if self.log is not None:
                self.log.error(f"frame save failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._pending -= 1

    @property
    def saved(self) -> int:
        return self._saved

    def close(self) -> None:
        self._pool.shutdown(wait=True)
