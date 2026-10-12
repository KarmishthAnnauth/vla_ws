#!/usr/bin/env python3
"""Saves each planned frame as a JPEG: the CAM_FRONT image MindDrive was given with
its two predicted trajectories drawn in, a top-down view, and a text panel with the
command, the decision expert's meta-actions, speeds and timings.

The MindDrive counterpart of orion_f1tenth/frame_saver.py.  Points are projected
with MindDrive's own CAM_FRONT calibration (LIDAR2IMG, 70 deg on 1600x900, camera
1.60 m above the ground), the geometry the real frame is formatted to imitate,
so the overlay shows where the model *thinks* it is pointing in its own image;
it is not a calibration of the car's camera.  The image is the formatted frame
before the agent's JPEG-q20 pass (same pixels geometrically, easier to read).

Colours: PATH trajectory (what the controller steers on) red, SPEED trajectory
(6 timed waypoints, the desired speed comes from it) orange, the extrapolated
tail handed to the controller thin red, global route ahead yellow, the route
node the planner is at blue.

Writing happens on its own thread so the inference thread never waits for it.
If a backlog builds up (slow disk) frames are dropped, not queued.
"""

from __future__ import annotations

import os
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import cv2
import numpy as np

from minddrive_ros.agent_constants import LIDAR2EGO, LIDAR2IMG

OUT_W, OUT_H = 1024, 576    # the 1600x900 frame is scaled to this
PANEL_H = 110
BEV_W = 320                 # top-down view: pixels wide
BEV_PX_PER_M = 100.0        # real metres
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY = (255, 255, 255), (170, 170, 170)
# BGR
RED, ORANGE, YELLOW, BLUE = (0, 0, 255), (0, 140, 255), (0, 220, 255), (255, 80, 0)

_EGO2LIDAR = np.linalg.inv(LIDAR2EGO)


def ego_to_lidar(pts_ego: np.ndarray) -> np.ndarray:
    """ROS ego ground points (x forward, y left, model metres) -> LIDAR_TOP [N, 3]."""
    pts = np.asarray(pts_ego, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([pts, np.zeros(len(pts)), np.ones(len(pts))])
    return (hom @ _EGO2LIDAR.T)[:, :3]


def project_lidar_points(pts_lidar: np.ndarray) -> np.ndarray:
    """LIDAR_TOP [N, 3] (model metres) -> CAM_FRONT pixels in the 1600x900 frame; NaN behind."""
    pts = np.asarray(pts_lidar, dtype=np.float64).reshape(-1, 3)
    hom = np.column_stack([pts, np.ones(len(pts))]) @ LIDAR2IMG["CAM_FRONT"].T
    out = np.full((len(pts), 2), np.nan)
    ok = hom[:, 2] > 0.1
    out[ok] = hom[ok, :2] / hom[ok, 2:3]
    return out


def lidar_plan_on_ground(preds: np.ndarray) -> np.ndarray:
    """MindDrive's (N, 2) trajectory (LIDAR_TOP x right, y forward) on the ground plane."""
    p = np.asarray(preds, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([p, np.full(len(p), -LIDAR2EGO[2, 3])])


def render_bev(h: int, path_ego: np.ndarray, speed_ego: np.ndarray, ext_ego: np.ndarray,
               route_ego: np.ndarray, near_ego: np.ndarray, world_scale: float) -> np.ndarray:
    """Top-down view, car at the bottom centre facing up, REAL metres, 0.5 m grid.
    All inputs ROS ego (x forward, y left), model metres."""
    bev = np.full((h, BEV_W, 3), 30, np.uint8)
    ox, oy = BEV_W // 2, h - 40

    def px(pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2) / world_scale
        return [(int(round(ox - y * BEV_PX_PER_M)), int(round(oy - x * BEV_PX_PER_M))) for x, y in pts]

    step = int(BEV_PX_PER_M * 0.5)
    for gy in range(oy, -1, -step):
        cv2.line(bev, (0, gy), (BEV_W, gy), (55, 55, 55), 1)
        cv2.putText(bev, f"{(oy - gy) / BEV_PX_PER_M:.1f} m", (4, gy - 3), FONT, 0.35, GREY, 1, cv2.LINE_AA)
    for gx in range(ox % step, BEV_W, step):
        cv2.line(bev, (gx, 0), (gx, h), (55, 55, 55), 1)
    r = px(route_ego)
    if len(r) > 1:
        cv2.polylines(bev, [np.array(r, np.int32)], False, YELLOW, 1, cv2.LINE_AA)
    e = px(ext_ego)
    if len(e) > 1:
        cv2.polylines(bev, [np.array(e, np.int32)], False, RED, 1, cv2.LINE_AA)
    s = px(speed_ego)
    if len(s) > 1:
        cv2.polylines(bev, [np.array(s, np.int32)], False, ORANGE, 1, cv2.LINE_AA)
    for q in s[1:]:
        cv2.circle(bev, q, 3, ORANGE, -1, cv2.LINE_AA)
    p = px(path_ego)
    if len(p) > 1:
        cv2.polylines(bev, [np.array(p, np.int32)], False, RED, 2, cv2.LINE_AA)
    for q in p[1:]:
        cv2.circle(bev, q, 2, RED, -1, cv2.LINE_AA)
    for q in px(near_ego):
        cv2.circle(bev, q, 6, BLUE, -1, cv2.LINE_AA)
    # the car: 0.33 x 0.2 m, origin at base_link
    cv2.rectangle(bev, (ox - 10, oy - 5), (ox + 10, oy + 28), WHITE, 1)
    cv2.putText(bev, "top-down, real m", (BEV_W - 135, 16), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    return bev


def render(bgr: np.ndarray, path_preds: np.ndarray, speed_preds: np.ndarray, plan_ego: np.ndarray,
           ext_ego: np.ndarray, route_ego: np.ndarray, near_ego: np.ndarray, text: str,
           info_lines: List[str], world_scale: float = 10.0) -> np.ndarray:
    """Returns a BGR image: the frame with the overlay and the top-down view, the text panel below.

    path_preds / speed_preds: MindDrive's raw LIDAR_TOP trajectories; the rest ROS ego (x fwd,
    y left), model metres: plan_ego the plan the controller gets (ego origin prepended) and
    ext_ego its extrapolated tail (starting at the last plan point), both as the node publishes
    them (MindDrive's frame rotated, not shifted); route_ego the global route ahead and near_ego
    the route node, true ego (base_link) frame."""
    img = bgr.copy()
    h, w = img.shape[:2]

    def draw(pts_lidar, color, r, thick):
        valid = [(int(round(u)), int(round(v))) for u, v in project_lidar_points(pts_lidar)
                 if np.isfinite(u) and -w < u < 2 * w and -h < v < 2 * h]
        if thick and len(valid) > 1:
            cv2.polylines(img, [np.array(valid, np.int32)], False, color, thick, cv2.LINE_AA)
        if r:
            for q in valid:
                cv2.circle(img, q, r, color, -1, cv2.LINE_AA)

    draw(ego_to_lidar(route_ego), YELLOW, 0, 2)
    # the node treats the lidar frame as the ego frame (rotation only), so the tail goes back the same way
    ext = np.asarray(ext_ego, dtype=np.float64).reshape(-1, 2)
    draw(lidar_plan_on_ground(np.column_stack([-ext[:, 1], ext[:, 0]])), RED, 0, 2)
    origin = lidar_plan_on_ground(np.zeros((1, 2)))
    # the speed trajectory (orange, six points) and the path trajectory (red) from the lidar origin
    draw(np.vstack([origin, lidar_plan_on_ground(speed_preds)]), ORANGE, 0, 2)
    draw(lidar_plan_on_ground(speed_preds), ORANGE, 6, 0)
    draw(np.vstack([origin, lidar_plan_on_ground(path_preds)]), RED, 0, 4)
    draw(lidar_plan_on_ground(path_preds), RED, 4, 0)
    draw(ego_to_lidar(near_ego), BLUE, 10, 0)

    img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    speed_ego = np.vstack([np.zeros((1, 2)), np.stack([speed_preds[:, 1], -speed_preds[:, 0]], axis=1)])
    img = np.hstack([img, render_bev(OUT_H, plan_ego, speed_ego, ext_ego, route_ego, near_ego, world_scale)])
    w = img.shape[1]

    wrapped = textwrap.wrap(text.strip(), width=125) if text.strip() else []
    panel_h = max(PANEL_H, 22 * (len(wrapped) + len(info_lines)) + 16)
    panel = np.zeros((panel_h, w, 3), np.uint8)
    y = 24
    for line in info_lines:
        cv2.putText(panel, line, (10, y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        y += 22
    for line in wrapped:
        cv2.putText(panel, line, (10, y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
        y += 22
    legend = [("path", RED), ("speed traj", ORANGE), ("route", YELLOW), ("route node", BLUE)]
    x = w - 420
    for name, color in legend:
        cv2.circle(panel, (x, 18), 5, color, -1, cv2.LINE_AA)
        cv2.putText(panel, name, (x + 10, 23), FONT, 0.45, GREY, 1, cv2.LINE_AA)
        x += 105
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

    def submit(self, plan_no: int, bgr: np.ndarray, path_preds: np.ndarray, speed_preds: np.ndarray,
               plan_ego: np.ndarray, ext_ego: np.ndarray, route_ego: np.ndarray, near_ego: np.ndarray,
               text: str, info_lines: List[str]) -> None:
        self._n += 1
        if (self._n - 1) % self.every:
            return
        with self._lock:
            if self._pending >= 3:
                return
            self._pending += 1
        self._pool.submit(self._write, plan_no, bgr, path_preds, speed_preds, plan_ego, ext_ego,
                          route_ego, near_ego, text, info_lines)

    def _write(self, plan_no, *args) -> None:
        try:
            out = render(*args, world_scale=self.world_scale)
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
