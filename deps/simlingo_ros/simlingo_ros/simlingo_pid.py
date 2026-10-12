#!/usr/bin/env python3
"""SimLingo's own longitudinal + lateral PID controllers, usable without CARLA.

This replaces the Stanley controller in the ROS path. SimLingo's route waypoints
were trained and tuned against *this* controller (``agent_simlingo.py::control_pid``),
so running anything else on top of them changes the closed-loop behaviour the model
expects.

The implementation is a verbatim port of three upstream pieces:

  * ``team_code/transfuser_utils.py::PIDController``        (longitudinal, speed)
  * ``team_code/nav_planner.py::LateralPIDController``      (lateral, steering)
  * ``team_code/agent_simlingo.py::control_pid`` + ``interpolate_waypoints``

We import those classes from the simlingo repo when they are importable and fall
back to the copies below otherwise: ``transfuser_utils`` and ``nav_planner`` both
``import carla`` at module scope, which is not available in the inference
container on the Orin. The fallbacks are line-for-line identical to upstream —
if you touch them, port the change from upstream rather than the other way round.

Gains and thresholds come from ``team_code/config_simlingo.py::GlobalConfig``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

# ── Upstream controllers, with carla-free fallbacks ──────────────────────────

try:  # pragma: no cover - depends on the container's simlingo install
    from team_code.transfuser_utils import PIDController  # type: ignore
    from team_code.nav_planner import LateralPIDController  # type: ignore

    UPSTREAM_CONTROLLERS = True
except Exception:  # ImportError, or `import carla` failing inside them
    UPSTREAM_CONTROLLERS = False

    class PIDController(object):  # noqa: D101 - verbatim: transfuser_utils.py
        """PID controller that converts waypoints to steer, brake and throttle commands"""

        def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
            self.k_p = k_p
            self.k_i = k_i
            self.k_d = k_d

            self.window = deque([0 for _ in range(n)], maxlen=n)

        def step(self, error):
            self.window.append(error)

            if len(self.window) >= 2:
                integral = np.mean(self.window)
                derivative = self.window[-1] - self.window[-2]
            else:
                integral = 0.0
                derivative = 0.0

            return self.k_p * error + self.k_i * integral + self.k_d * derivative

    class LateralPIDController(object):  # noqa: D101 - verbatim: nav_planner.py
        """PID controller"""

        def __init__(self, k_p=3.118357247806046, k_d=1.3782508892109167,
                     k_i=0.6406067986034124, speed_scale=0.9755321901954155,
                     speed_offset=1.9152884533402488, default_lookahead=24,
                     speed_threshold=23.150102938235136, n=6, inference_mode=False):
            self.k_p = k_p
            self.k_d = k_d
            self.k_i = k_i
            self.speed_scale = speed_scale
            self.speed_offset = speed_offset
            self.default_lookahead = default_lookahead
            self.speed_threshold = speed_threshold
            self.n = n
            self.inference_mode = inference_mode

            self._saved_window = []
            self._window = []

        def step(self, route_np, current_speed):
            current_speed = current_speed * 3.6
            if self.inference_mode:
                n_lookahead = np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105) / 10
                n_lookahead = n_lookahead - 2
                n_lookahead = int(min(n_lookahead, route_np.shape[0] - 1))
            else:
                n_lookahead = int(min(np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105),
                                      route_np.shape[0] - 1))

            n_lookahead = min(n_lookahead, len(route_np) - 1)
            desired_heading_vec = route_np[n_lookahead]

            yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
            heading_error = (yaw_path) % (2 * np.pi)
            heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi

            # the scaling doesn't deserve any specific purpose but is a leftover from a previous
            # less efficient implementation, on which we optimized the parameters
            heading_error = heading_error * 180. / np.pi / 90.

            self._window.append(heading_error)
            self._window = self._window[-self.n:]

            derivative = 0. if len(self._window) == 1 else self._window[-1] - self._window[-2]
            integral = np.mean(self._window)

            return np.clip(self.k_p * heading_error + self.k_d * derivative + self.k_i * integral,
                           -1., 1.).item()


# ── Configuration (team_code/config_simlingo.py::GlobalConfig) ───────────────

@dataclass
class SimLingoPIDConfig:
    """The subset of GlobalConfig that control_pid reads."""

    # Longitudinal PID gains
    speed_kp: float = 1.75
    speed_ki: float = 1.0
    speed_kd: float = 2.0
    speed_n: int = 20          # buffer size, tuned at carla_fps (20 Hz)

    # Longitudinal thresholds
    brake_speed: float = 0.4   # desired speed below which brake is triggered
    brake_ratio: float = 1.1
    clip_delta: float = 1.0    # max change in speed input to the longitudinal controller
    clip_throttle: float = 1.0

    # Waypoint timing — fixes which speed waypoints span the 0.5 s window
    carla_fps: int = 20
    wp_dilation: int = 1
    data_save_freq: int = 5

    # Stuck / creep recovery. Upstream counts 20 Hz simulator frames; we run the
    # controller once per inference (~4 Hz), so these are expressed in seconds:
    #   stuck_threshold 800 frames / 20 fps = 40 s
    #   creep_duration   15 frames / 20 fps = 0.75 s
    stuck_threshold_sec: float = 40.0
    creep_duration_sec: float = 0.75
    creep_throttle: float = 0.4
    stuck_speed_mps: float = 0.1


# ── Controller ───────────────────────────────────────────────────────────────

class SimLingoPIDController:
    """``agent_simlingo.py::control_pid`` as a standalone, frame-rate aware object.

    Call :meth:`step` once per model prediction with the raw model outputs. The
    upstream agent calls it once per simulator frame (20 Hz) because it also runs
    the model every frame; here inference is slower, so ``control_hz`` is used to
    keep the speed PID's integral window covering the same ~1 s of history it was
    tuned with (``window_rate_compensation``).
    """

    def __init__(self, config: SimLingoPIDConfig | None = None,
                 control_hz: float = 20.0,
                 window_rate_compensation: bool = True) -> None:
        self.config = config or SimLingoPIDConfig()
        cfg = self.config

        speed_n = cfg.speed_n
        if window_rate_compensation and control_hz > 0.0:
            # speed_n samples at carla_fps == speed_n/carla_fps seconds of history.
            # Keep that duration, not the sample count, when stepping slower.
            speed_n = int(round(cfg.speed_n * control_hz / cfg.carla_fps))
            speed_n = max(2, speed_n)

        self.speed_n = speed_n

        self.speed_controller = PIDController(k_p=cfg.speed_kp, k_i=cfg.speed_ki,
                                              k_d=cfg.speed_kd, n=speed_n)

        # The lateral controller needs the same treatment, and upstream never had
        # to give it any: agent_simlingo.py steps it once per simulator frame, so
        # its gains are implicitly tuned for dt = 1/carla_fps.  Here it is stepped
        # once per inference (~1 Hz over the link), and two of its three terms are
        # rate-dependent:
        #
        #   derivative = window[-1] - window[-2]
        #
        # is a *difference*, not a rate, so k_d * derivative = k_d * dt * de/dt.
        # Stepping 20x slower makes the same de/dt produce a 20x larger term --
        # which is how a mild heading error saturated the output.  Holding
        # k_d * dt constant gives k_d' = k_d * control_hz / carla_fps.
        #
        #   integral = np.mean(window)
        #
        # is a mean rather than a sum, so it is already normalised; what changes
        # is the *duration* it averages over.  Scaling n the way speed_n is
        # scaled keeps that duration fixed and leaves k_i alone.
        #
        # k_p is rate-independent and is left exactly as tuned.
        #
        # Measured on complete_scene_3 route 1 (2026-09-11): asynchronous, 40 of
        # 42 frames pinned at steer=-1.000; the same route synchronous, 0 of 63
        # saturated across 61 distinct values.  Uncompensated k_d is the
        # difference between those two runs.
        lat_kwargs = {"inference_mode": False}
        if window_rate_compensation and control_hz > 0.0:
            rate = control_hz / cfg.carla_fps
            probe = LateralPIDController(inference_mode=False)
            lat_kwargs["k_d"] = probe.k_d * rate
            lat_kwargs["n"] = max(2, int(round(probe.n * rate)))
            self.lateral_rate = rate
            self.lateral_k_d = lat_kwargs["k_d"]
            self.lateral_n = lat_kwargs["n"]
        else:
            self.lateral_rate = 1.0
            self.lateral_k_d = None
            self.lateral_n = None

        # inference_mode=False matches agent_simlingo.py: the route is interpolated
        # to 0.1 m spacing below, which is the spacing the expert branch expects.
        self.turn_controller = LateralPIDController(**lat_kwargs)

        # Stuck / creep state, in seconds rather than frames.
        self._stuck_time: float = 0.0
        self._force_move_until: float = 0.0

    # ── Upstream control_pid ─────────────────────────────────────────────────

    def step(self, route_waypoints: np.ndarray, speed: float,
             speed_waypoints: np.ndarray) -> Tuple[float, float, bool, float]:
        """Verbatim ``control_pid``: (steer, throttle, brake, desired_speed).

        route_waypoints: [N, 2] ego-frame spatial waypoints (x forward, y left)
        speed:           current forward speed in m/s
        speed_waypoints: [M, 2] ego-frame speed waypoints
        """
        cfg = self.config

        # m / s required to drive
        one_second = int(cfg.carla_fps // (cfg.wp_dilation * cfg.data_save_freq))
        half_second = one_second // 2
        desired_speed = float(
            np.linalg.norm(speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]) * 2.0
        )

        brake = (desired_speed < cfg.brake_speed) or ((speed / max(desired_speed, 1e-3)) > cfg.brake_ratio)

        delta = np.clip(desired_speed - speed, 0.0, cfg.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = float(np.clip(throttle, 0.0, cfg.clip_throttle))
        throttle = throttle if not brake else 0.0

        route_interp = self.interpolate_waypoints(np.asarray(route_waypoints).squeeze())

        steer = self.turn_controller.step(route_interp, speed)
        steer = float(np.clip(steer, -1.0, 1.0))
        steer = round(steer, 3)

        return steer, throttle, bool(brake), desired_speed

    # In: Waypoints NxD
    # Out: Waypoints NxD equally spaced 0.1 across D
    @staticmethod
    def interpolate_waypoints(waypoints: np.ndarray) -> np.ndarray:
        waypoints = waypoints.copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]

        dists = np.linalg.norm(waypoints - shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists)) * 1e-4  # Prevents dists not being strictly increasing

        interp = PchipInterpolator(dists, waypoints, axis=0)

        x = np.arange(0.1, dists[-1], 0.1)

        interp_points = interp(x)

        # There is a possibility that all points are at 0, meaning there is no point distanced 0.1
        # In this case we output the last (assumed to be furthest) waypoint.
        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]

        return interp_points

    # ── Stuck / creep recovery (agent_simlingo.py::run_step) ─────────────────

    def apply_creep(self, throttle: float, brake: bool, speed: float,
                    now_sec: float, dt_sec: float) -> Tuple[float, bool]:
        """Restart mechanism in case the car got stuck, in seconds rather than frames."""
        cfg = self.config

        # 0.1 is just an arbitrary low number to threshold when the car is stopped
        if speed < cfg.stuck_speed_mps:
            self._stuck_time += dt_sec
        else:
            self._stuck_time = 0.0

        if self._stuck_time > cfg.stuck_threshold_sec:
            self._force_move_until = now_sec + cfg.creep_duration_sec
            self._stuck_time = 0.0

        if now_sec < self._force_move_until:
            throttle = max(cfg.creep_throttle, throttle)
            brake = False

        return throttle, brake

    @property
    def creeping(self) -> bool:
        return self._force_move_until > 0.0
