"""ROS-free checks of orion_f1tenth.orion_model (no model, no CUDA needed)."""

import math

import numpy as np
import pytest

from orion_f1tenth.orion_model import (
    CMD_FOLLOW, CMD_LEFT, CMD_RIGHT, LIDAR2EGO, TRAIN_CAM_H, TRAIN_CAM_W, build_can_bus,
    command2hot, command2nohot, command_from_route, crop_to_hfov, desired_speed_from_waypoints,
    extend_plan, format_camera_frame, jpeg_roundtrip, lidar_to_ros_ego, make_views,
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
    assert len(v) == 6
    assert all(x is front for x in v[:3])
    assert all(x.max() == 0 and x.shape == front.shape for x in v[3:])
    v = make_views(front, "black")
    assert v[0] is front and all(x.max() == 0 for x in v[1:])
    with pytest.raises(ValueError):
        make_views(front, "mirror")
    with pytest.raises(ValueError):
        make_views(np.zeros((720, 1280, 3), np.uint8), "copy")


def test_jpeg_roundtrip_shape_and_bypass():
    img = np.random.randint(0, 255, (TRAIN_CAM_H, TRAIN_CAM_W, 3), np.uint8)
    out = jpeg_roundtrip(img, 20)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert jpeg_roundtrip(img, 0) is img


def test_command_encodings_match_agent():
    assert command2nohot(4) == 3 and command2nohot(-1) == 3
    assert list(command2hot(1)) == [1, 0, 0, 0, 0, 0]
    assert list(command2hot(6)) == [0, 0, 0, 0, 0, 1]


def test_build_can_bus_layout():
    can_bus, ego_pose, ego_pose_inv, l2g = build_can_bus((3.0, -2.0), math.pi / 2, 4.5, (0.1, 0.2, 9.8), (0, 0, 0.3))
    assert can_bus.shape == (18,)
    assert can_bus[0] == 3.0 and can_bus[1] == -2.0 and can_bus[7] == 4.5
    assert np.allclose(can_bus[10:13], [0.1, 0.2, 9.8]) and np.allclose(can_bus[13:16], [0, 0, 0.3])
    assert can_bus[16] == math.pi / 2 and abs(can_bus[17] - 90.0) < 1e-9
    # quaternion for yaw pi/2 about z: w = cos(pi/4), z = sin(pi/4)
    assert np.allclose(can_bus[3:7], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    # ego_pose = ego2world @ LIDAR2EGO, ego_pose_inv its inverse
    ego2world = np.eye(4)
    ego2world[:2, :2] = [[0, -1], [1, 0]]
    ego2world[:2, 3] = [3.0, -2.0]
    assert np.allclose(ego_pose, ego2world @ LIDAR2EGO) and np.allclose(l2g, ego_pose)
    assert np.allclose(ego_pose @ ego_pose_inv, np.eye(4), atol=1e-6)


def test_lidar_to_ros_ego_axes():
    # lidar index 1 is forward, index 0 is right
    p = np.array([[1.0, 2.0], [0.0, 5.0]])
    out = lidar_to_ros_ego(p)
    assert np.allclose(out, [[2.0, -1.0], [5.0, 0.0]])


def test_desired_speed_matches_control_pid():
    wps = np.array([[0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0], [0.0, 10.0], [0.0, 12.0]])
    # 0.75 * |wp0| * 2 + 0.25 * |wp1 - wp0| * 2 = 0.75*4 + 0.25*4 = 4
    assert abs(desired_speed_from_waypoints(wps) - 4.0) < 1e-9


def test_extend_plan_prepends_origin_and_extrapolates():
    pts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.1]])
    out = extend_plan(pts, 1.0, dt=0.5)
    assert np.allclose(out[0], [0, 0])
    assert len(out) == 1 + 3 + 2
    assert np.allclose(out[-1], [5.0, 0.3])
    assert len(extend_plan(pts, 0.0)) == 4


def test_command_from_route():
    straight = np.array([[i * 1.0, 0.0] for i in range(40)])
    assert command_from_route(straight, 0, 15.0, 35.0) == CMD_FOLLOW
    # 90 deg left turn 5 m ahead
    left = np.array([[i * 1.0, 0.0] for i in range(6)] + [[5.0, i * 1.0] for i in range(1, 30)])
    assert command_from_route(left, 0, 15.0, 35.0) == CMD_LEFT
    right = left * np.array([1.0, -1.0])
    assert command_from_route(right, 0, 15.0, 35.0) == CMD_RIGHT
    # past the turn: follow again
    assert command_from_route(left, 10, 15.0, 35.0) == CMD_FOLLOW
    assert command_from_route(left[:2], 0, 15.0, 35.0) == CMD_FOLLOW
