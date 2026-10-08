#!/usr/bin/env python3
"""Frames, world scaling and target-point selection for the real-world node.

Three frames, and mixing them up is the most expensive mistake available:

  ROS map    right-handed, what the particle filter publishes (x east-ish, y left
             of x, yaw counter-clockwise).  No CARLA bridge here, so nothing is
             mirrored on the way in.
  ROS ego    x forward, y LEFT.
  model      x forward, y RIGHT -- CARLA's frame, which SimLingo was trained on
             and predicts in.

ROS ego and model differ only by the sign of y, so ``map_to_model`` rotates into
the ego frame and negates y.  ``model_to_map`` is its exact inverse.  (The
reference node arrives at the same two functions by a different argument: the
CARLA bridge negates both world y and yaw, which composes to the same mirror.)

World scale: the F1TENTH drives a 1:10 replica of the CARLA scene, so every
metre the car sees is a tenth of a metre the model was trained on.  Callers pass
positions and speeds *already multiplied by world_scale* into this module and
divide the model's output by it on the way out; nothing here knows about scale.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# Route-planner discard window from simlingo/team_code/nav_planner.py
# RoutePlanner(min_distance=7.5, max_distance=50.0) -- model metres.
DEFAULT_MIN_DISTANCE = 7.5
DEFAULT_MAX_DISTANCE = 50.0


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def map_to_ego(pts_map: np.ndarray, ego_pos: np.ndarray, ego_yaw: float) -> np.ndarray:
    """ROS map -> ROS ego (x forward, y LEFT). Returns [N, 2]."""
    pts = np.asarray(pts_map, dtype=np.float64).reshape(-1, 2)
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    R = np.array([[c, s], [-s, c]])
    return (pts - np.asarray(ego_pos, dtype=np.float64).reshape(1, 2)) @ R.T


def ego_to_map(pts_ego: np.ndarray, ego_pos: np.ndarray, ego_yaw: float) -> np.ndarray:
    pts = np.asarray(pts_ego, dtype=np.float64).reshape(-1, 2)
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.asarray(ego_pos, dtype=np.float64).reshape(1, 2)


def map_to_model(pts_map: np.ndarray, ego_pos: np.ndarray, ego_yaw: float) -> np.ndarray:
    """ROS map -> model ego frame (x forward, y RIGHT). Returns [N, 2]."""
    local = map_to_ego(pts_map, ego_pos, ego_yaw)
    return np.stack([local[:, 0], -local[:, 1]], axis=1)


def model_to_map(pts_model: np.ndarray, ego_pos: np.ndarray, ego_yaw: float) -> np.ndarray:
    """Model ego frame (x forward, y RIGHT) -> ROS map. Returns [N, 2]."""
    pts = np.asarray(pts_model, dtype=np.float64).reshape(-1, 2)
    local = np.stack([pts[:, 0], -pts[:, 1]], axis=1)
    return ego_to_map(local, ego_pos, ego_yaw)


def load_route_csv(path: str) -> np.ndarray:
    """``x,y[,v,...]`` rows, map frame, real metres. Header lines and '#' comments are skipped."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(";", ",").split(",") if p.strip()]
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                continue                       # header or malformed line
    if len(rows) < 2:
        raise ValueError(f"route CSV {path} holds fewer than two usable x,y rows")
    return np.asarray(rows, dtype=np.float64)


def thin_route(route: np.ndarray, min_spacing: float) -> np.ndarray:
    """Keep only waypoints at least ``min_spacing`` apart along the route (0 = keep all).

    CARLA's global plan -- what SimLingo's target points come from -- is sparse:
    route nodes tens of metres apart.  A dense path (slam/raceline export at a few
    centimetres) fed straight in gives a target that hovers at the discard
    radius forever.  This makes a dense path look like a sparse plan.
    """
    if min_spacing <= 0.0 or len(route) < 3:
        return route
    keep = [0]
    acc = 0.0
    for i in range(1, len(route)):
        acc += float(np.linalg.norm(route[i] - route[i - 1]))
        if acc >= min_spacing:
            keep.append(i)
            acc = 0.0
    if keep[-1] != len(route) - 1:
        keep.append(len(route) - 1)
    return route[keep]


class RoutePlanner:
    """SimLingo's target-point selection on a map-frame route.

    A transcription of ``RoutePlanner.run_step`` (team_code/nav_planner.py) plus
    the lines that consume it (agent_simlingo.py:444-452), with route progress
    tracked by a monotonic index instead of popleft()ing a deque.  Same rule as
    the reference node states plainly: ``min_distance`` / ``max_distance``
    decide which waypoints are *discarded*, never which one becomes the target.
    The target is always the next entry after the discarded ones.

    All coordinates in model metres (already world-scaled).
    """

    def __init__(self, min_distance: float = DEFAULT_MIN_DISTANCE,
                 max_distance: float = DEFAULT_MAX_DISTANCE,
                 loop: bool = False) -> None:
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.loop = bool(loop)
        self._route: Optional[np.ndarray] = None
        self._idx = 0
        self._started = False

    # ── route management ─────────────────────────────────────────────────────

    @property
    def route(self) -> Optional[np.ndarray]:
        return self._route

    @property
    def index(self) -> int:
        return self._idx

    @property
    def has_route(self) -> bool:
        return self._route is not None and len(self._route) >= 2

    def set_route(self, wps_model: np.ndarray) -> None:
        self._route = np.asarray(wps_model, dtype=np.float64).reshape(-1, 2)
        self._idx = 0
        self._started = False

    def remaining(self) -> int:
        if not self.has_route:
            return 0
        return len(self._route) if self.loop else len(self._route) - self._idx

    def finished(self) -> bool:
        """Point-to-point route consumed (never true for a loop).

        Progress is clamped to n-2 so the last two entries always stand (as
        upstream), so "finished" is reaching that clamp.
        """
        return self.has_route and not self.loop and self._idx >= len(self._route) - 2

    # ── target points ────────────────────────────────────────────────────────

    def _wp(self, i: int) -> np.ndarray:
        n = len(self._route)
        return self._route[i % n] if self.loop else self._route[min(i, n - 1)]

    def target_points(self, ego_pos: np.ndarray, ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        """Next two route entries in the model ego frame, the way upstream picks them."""
        if not self.has_route:
            raise RuntimeError("no route")
        route, n = self._route, len(self._route)
        ego_pos = np.asarray(ego_pos, dtype=np.float64)

        # Upstream assumes the ego starts on route[0]; on a real track the car
        # is placed wherever it is placed.  Start progress at the nearest
        # waypoint, once, and let the normal discard logic run from there.
        if not self._started:
            self._idx = int(np.argmin(np.linalg.norm(route - ego_pos, axis=1)))
            self._started = True

        span = n if self.loop else (n - self._idx)

        # Deviation from upstream, for a real car: advance progress to the
        # nearest waypoint within max_distance *along the route* ahead of the
        # current index.  Upstream never needs this because its discard rule
        # assumes the ego creeps along the route; a localisation jump, a wide
        # overshoot on a tiny track or a late start would otherwise leave
        # waypoints behind the car in range and make the target point trail
        # it.  Windowed and monotonic, so a route that loops back past the car
        # cannot pull progress forward to a later stretch.
        if span > 2:
            best_k, best_d, cumulative = 0, np.inf, 0.0
            for k in range(0, span):
                if k > 0:
                    cumulative += float(np.linalg.norm(self._wp(self._idx + k) - self._wp(self._idx + k - 1)))
                    if cumulative > self.max_distance:
                        break
                d = float(np.linalg.norm(self._wp(self._idx + k) - ego_pos))
                if d < best_d:
                    best_k, best_d = k, d
            self._idx = (self._idx + best_k) % n if self.loop else min(self._idx + best_k, max(n - 2, 0))
            span = n if self.loop else (n - self._idx)

        # Discard waypoints already reached (RoutePlanner.run_step).  The scan
        # walks at most max_distance *along the route*, not in a straight line,
        # so a route that loops back past the car does not get skipped through.
        # Upstream leaves two entries standing so route[1] always exists.
        if span > 2:
            to_pop = 0
            farthest_in_range = -np.inf
            cumulative = 0.0
            for k in range(1, span):
                if cumulative > self.max_distance:
                    break
                cumulative += float(np.linalg.norm(self._wp(self._idx + k) - self._wp(self._idx + k - 1)))
                d = float(np.linalg.norm(self._wp(self._idx + k) - ego_pos))
                if farthest_in_range < d <= self.min_distance:
                    farthest_in_range = d
                    to_pop = k
            if self.loop:
                self._idx = (self._idx + to_pop) % n
            else:
                self._idx = min(self._idx + to_pop, max(n - 2, 0))

        if self.loop:
            tp0, tp1 = self._wp(self._idx + 1), self._wp(self._idx + 2)
        else:
            rem = n - self._idx
            if rem > 2:
                tp0, tp1 = route[self._idx + 1], route[self._idx + 2]
            elif rem > 1:
                tp0 = tp1 = route[self._idx + 1]
            else:
                tp0 = tp1 = route[self._idx]

        return (map_to_model(tp0, ego_pos, ego_yaw)[0],
                map_to_model(tp1, ego_pos, ego_yaw)[0])
