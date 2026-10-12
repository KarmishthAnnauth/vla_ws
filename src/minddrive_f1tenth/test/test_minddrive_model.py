"""ROS-free checks of minddrive_f1tenth.minddrive_model (no model, no CUDA needed).

Needs minddrive_ros.agent_constants on the path (numpy only); torch is not imported.
"""

import math

import numpy as np
import pytest

from minddrive_f1tenth.minddrive_model import (
    CAMERA_ORDER, CMD_FOLLOW, CMD_LEFT, CMD_RIGHT, SPEED_PROFILES, TRAIN_CAM_H, TRAIN_CAM_W, InferenceResult,
    build_can_bus, command_from_route, crop_to_hfov, desired_speed_from_waypoints, extend_path, extend_plan,
    format_camera_frame, jpeg_roundtrip, lidar_to_ros_ego, make_views, parse_overrides, path_times,
)


def test_format_camera_frame_16_9_to_1600x900():
    out = format_camera_frame(np.zeros((720, 1280, 3), np.uint8))
    assert out.shape == (TRAIN_CAM_H, TRAIN_CAM_W, 3) and out.dtype == np.uint8


def test_format_camera_frame_crops_4_3_to_16_9_then_resizes():
    img = np.zeros((480, 640, 3), np.uint8)
    img[0:60] = 255              # a stripe at the top that the centre crop must remove
    out = format_camera_frame(img)
    assert out.shape == (TRAIN_CAM_H, TRAIN_CAM_W, 3)
    assert out.max() == 0


def test_crop_to_hfov_keeps_central_sector():
    img = np.zeros((720, 1280, 3), np.uint8)
    out = crop_to_hfov(img, 110.0, 70.0)
    expected = 1280 * math.tan(math.radians(35)) / math.tan(math.radians(55))
    assert abs(out.shape[1] - expected) <= 1
    assert crop_to_hfov(img, 0.0).shape == img.shape          # unknown FOV: untouched
    assert crop_to_hfov(img, 60.0).shape == img.shape         # narrower than the model's: untouched


def test_make_views_layout():
    front = np.full((TRAIN_CAM_H, TRAIN_CAM_W, 3), 7, np.uint8)
    v = make_views(front, "copy")
    assert list(v) == CAMERA_ORDER
    assert all(v[c] is front for c in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"))
    assert all(v[c].max() == 0 for c in ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"))
    v = make_views(front, "black")
    assert v["CAM_FRONT"] is front and all(v[c].max() == 0 for c in CAMERA_ORDER[1:])
    with pytest.raises(ValueError):
        make_views(front, "mirror")
    with pytest.raises(ValueError):
        make_views(np.zeros((720, 1280, 3), np.uint8))


def test_jpeg_roundtrip_keeps_shape_and_zero_quality_is_identity():
    img = np.random.default_rng(0).integers(0, 255, (90, 160, 3), np.uint8)
    out = jpeg_roundtrip(img, 20)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert jpeg_roundtrip(img, 0) is img


def test_build_can_bus_layout():
    pytest.importorskip("pyquaternion")       # in the container; not on every host
    cb = build_can_bus((12.0, -3.0), math.pi / 2, 4.0, (0.1, 0.2, 9.8), (0.0, 0.0, 0.3))
    assert cb.shape == (18,)
    assert cb[0] == 12.0 and cb[1] == -3.0 and cb[7] == 4.0
    assert np.allclose(cb[10:13], [0.1, 0.2, 9.8]) and np.allclose(cb[13:16], [0.0, 0.0, 0.3])
    assert cb[16] == math.pi / 2 and abs(cb[17] - 90.0) < 1e-9
    # quaternion (w, x, y, z) for a 90 deg yaw
    assert np.allclose(cb[3:7], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])


def test_lidar_to_ros_ego_rotation():
    # LIDAR_TOP: index 0 right, index 1 forward -> ROS: x forward, y left
    out = lidar_to_ros_ego(np.array([[1.0, 2.0], [-0.5, 3.0]]))
    assert np.allclose(out, [[2.0, -1.0], [3.0, 0.5]])


def test_desired_speed_formula():
    wps = np.array([[0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0], [0.0, 10.0], [0.0, 12.0]])
    # 0.75 * 2 * 2 + 0.25 * 2 * 2 = 4 m/s
    assert abs(desired_speed_from_waypoints(wps) - 4.0) < 1e-9


def test_extend_plan_prepends_origin_and_extrapolates():
    pts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    out = extend_plan(pts, 1.0, dt=0.5)
    assert out.shape == (6, 2)
    assert np.allclose(out[0], [0, 0]) and np.allclose(out[-1], [5.0, 0.0])
    assert extend_plan(pts, 0.0).shape == (4, 2)


def test_extend_path_by_distance_at_own_spacing():
    path = np.column_stack([np.linspace(0.5, 10.0, 20), np.zeros(20)])     # 20 points, 0.5 m apart
    out = extend_path(path, 3.0)
    assert np.allclose(out[0], [0, 0])
    assert out.shape == (1 + 20 + 6, 2)                                      # ceil(3.0 / 0.5) = 6 extra
    assert np.allclose(out[-1], [13.0, 0.0])
    assert extend_path(path, 0.0).shape == (21, 2)
    zero = extend_path(np.zeros((20, 2)), 5.0)                               # the model wants to stop
    assert zero.shape == (21, 2) and np.all(zero == 0)


def test_path_times_arc_length_over_speed():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert np.allclose(path_times(pts, 2.0), [0.0, 0.5, 1.0])
    assert np.allclose(path_times(pts, 0.0), [0.0, 0.5, 1.0])              # fallback: 0.5 s per point


def test_command_from_route_turns():
    straight = np.column_stack([np.arange(0, 60, 5.0), np.zeros(12)])
    assert command_from_route(straight, 0, 15.0, 35.0) == CMD_FOLLOW
    # a left turn 10 m ahead (counter-clockwise, +y)
    left = np.array([[0, 0], [5, 0], [10, 0], [10, 5], [10, 10], [10, 15], [10, 20]], float)
    assert command_from_route(left, 0, 15.0, 35.0) == CMD_LEFT
    right = left * np.array([1, -1])
    assert command_from_route(right, 0, 15.0, 35.0) == CMD_RIGHT


def test_inference_result_views():
    speed = np.array([[0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0], [0.0, 10.0], [0.0, 12.0]], np.float32)
    path = np.column_stack([np.zeros(20), np.linspace(0.5, 10, 20)]).astype(np.float32)
    r = InferenceResult(speed_preds=speed, path_preds=path, speed_value=3, path_value=2, forward_ms=900.0)
    assert np.allclose(r.speed_ros_ego[:, 0], speed[:, 1]) and np.allclose(r.speed_ros_ego[:, 1], 0)
    assert r.path_ros_ego.shape == (20, 2)
    assert abs(r.desired_speed - 4.0) < 1e-6
    assert r.speed_action == "speed up" and r.path_action == "turn left"
    assert InferenceResult(speed, path, None, 99, 1.0).text == "speed=?None path=?99"


def test_parse_overrides_types_and_validation():
    o = parse_overrides("compile_targets=,cuda_graph_vit=false,int8_alpha=0.7,vit_input_size=512,llm_attn=sdpa")
    assert o == {"compile_targets": "", "cuda_graph_vit": False, "int8_alpha": 0.7,
                 "vit_input_size": 512, "llm_attn": "sdpa"}
    assert parse_overrides("") == {}
    with pytest.raises(ValueError):
        parse_overrides("no_such_option=1")
    with pytest.raises(ValueError):
        parse_overrides("merge_lora=maybe")
    with pytest.raises(ValueError):
        parse_overrides("merge_lora")


def test_speed_profiles_consistent():
    assert set(SPEED_PROFILES) == {"baseline", "eager", "fast", "int8", "lite"}
    assert SPEED_PROFILES["baseline"]["precision"] == "config"
    assert not SPEED_PROFILES["eager"]["compile_targets"] and not SPEED_PROFILES["eager"]["cuda_graph_vit"]
    assert SPEED_PROFILES["fast"]["compile_targets"] == "llm,vit" and SPEED_PROFILES["fast"]["cuda_graph_vit"]
    assert SPEED_PROFILES["int8"]["int8_targets"] == "llm" and SPEED_PROFILES["int8"]["int8_calib_path"]
    assert SPEED_PROFILES["lite"]["vit_input_size"] == 512 and SPEED_PROFILES["lite"]["rear_view_refresh_every"] == 2
    keys = set(SPEED_PROFILES["baseline"])
    assert all(set(p) == keys for p in SPEED_PROFILES.values())


def test_frame_saver_render_shapes():
    """The overlay renders both trajectories without touching the model or ROS."""
    from minddrive_f1tenth.frame_saver import OUT_W, BEV_W, OUT_H, render

    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 255, (TRAIN_CAM_H, TRAIN_CAM_W, 3), np.uint8)
    speed = np.array([[0.0, 2.0], [0.1, 4.0], [0.3, 6.0], [0.6, 8.0], [1.0, 10.0], [1.5, 12.0]], np.float32)
    path = np.column_stack([0.01 * np.arange(1, 21) ** 2, np.linspace(0.5, 10, 20)]).astype(np.float32)
    plan_ego = np.vstack([np.zeros((1, 2)), np.stack([path[:, 1], -path[:, 0]], axis=1)])
    ext_ego = plan_ego[-1] + np.arange(0, 4)[:, None] * np.array([[0.5, 0.0]])
    route = np.column_stack([np.linspace(0, 40, 9), np.zeros(9)])
    near = np.array([[8.0, 0.5]])
    out = render(bgr, path, speed, plan_ego, ext_ego, route, near, "speed=speed up path=lanefollow",
                 ["plan 1  route-0001 frame 0", "speed 0.40 m/s real"], world_scale=10.0)
    assert out.dtype == np.uint8 and out.shape[2] == 3
    assert out.shape[1] == OUT_W + BEV_W and out.shape[0] > OUT_H
    # a zero-length path (the model wants to stop) and an empty route must not raise
    out = render(bgr, np.zeros((20, 2), np.float32), np.zeros((6, 2), np.float32), np.zeros((21, 2)),
                 np.zeros((1, 2)), np.zeros((0, 2)), near, "", [], world_scale=10.0)
    assert out.shape[1] == OUT_W + BEV_W
