"""
SimLingo on a real F1TENTH: camera + inference + trajectory controller (Orin side).

  camera_node                 /camera/front/image/compressed ──┐
  Nano: particle_filter       /pf/pose/odom  (map frame)  ─────┼─► simlingo_realworld_node
  Nano: vesc_to_odom          /odom          (speed)      ─────┤        │  /simlingo/plan (map frame)
  Nano: waypoint_visualiser   /global_path   (nav_msgs/Path) ──┘        ▼
                                                            trajectory_controller_node
                                                                        │  /drive (AckermannDriveStamped)
                                                                        ▼
                                                    Nano: ackermann_mux -> ackermann_to_vesc -> VESC

Launch arguments (all optional except the two paths):
  checkpoint_path      SimLingo .ckpt / pytorch_model.pt (hydra config 3 levels up)
  simlingo_path        simlingo repository root
  world_scale          model metres per real metre (10 for the 1:10 replica track)
  speed_world_scale    model m/s per real m/s; 0 = world_scale. Lower = faster car
  route_csv            x,y[,v] CSV in the map frame; empty = wait for /global_path
  route_loop           true for a closed circuit
  camera_source        v4l2 | gstreamer | synthetic   (synthetic = no camera needed)
  camera_device        /dev/video0
  use_camera_node      false if the camera is published by something else
  max_speed_mps        hard cap on commanded speed, except the kick of the stuck recovery (start low)
  lateral_controller   pure_pursuit | simlingo_pid
  pid_gain             simlingo_pid only: multiplier on the PID's gains (1.0 = as tuned upstream)
  steer_smoothing_sec  low-pass time constant on the steering command, 0 = off
  control_hz           controller rate (keep >= 10: the mux times out after 0.2 s)
  start_estopped       true: hold zero speed until /simlingo/estop receives false
  creep_after_sec      > 0: after this long at standstill while tracking a plan, kick with
                       creep_speed_mps until the car rolls at creep_release_speed_mps, then
                       creep_hold_speed_mps, creep_duration_sec in total (stuck recovery)
  min_speed_mps        floor on the commanded speed whenever the plan wants to drive
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("checkpoint_path",
                              default_value="/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"),
        DeclareLaunchArgument("simlingo_path", default_value="/benchmarking/simlingo"),
        DeclareLaunchArgument("world_scale", default_value="10.0"),
        DeclareLaunchArgument("speed_world_scale", default_value="0.0"),
        DeclareLaunchArgument("route_csv", default_value=""),
        DeclareLaunchArgument("route_loop", default_value="false"),
        DeclareLaunchArgument("route_min_spacing_m", default_value="0.0"),
        DeclareLaunchArgument("image_topic", default_value="/camera/front/image/compressed"),
        DeclareLaunchArgument("pose_topic", default_value="/pf/pose/odom"),
        DeclareLaunchArgument("speed_topic", default_value="/odom"),
        DeclareLaunchArgument("route_topic", default_value="/global_path"),
        DeclareLaunchArgument("drive_topic", default_value="/drive"),
        DeclareLaunchArgument("pose_offset_x", default_value="-0.27"),
        DeclareLaunchArgument("use_camera_node", default_value="true"),
        DeclareLaunchArgument("camera_source", default_value="v4l2"),
        DeclareLaunchArgument("camera_device", default_value="/dev/video0"),
        DeclareLaunchArgument("camera_width", default_value="1280"),
        DeclareLaunchArgument("camera_height", default_value="720"),
        DeclareLaunchArgument("camera_fps", default_value="30.0"),
        DeclareLaunchArgument("gst_pipeline", default_value=""),
        DeclareLaunchArgument("fast_inference", default_value="true"),
        DeclareLaunchArgument("use_cot", default_value="false"),
        DeclareLaunchArgument("save_frames_dir", default_value=""),
        DeclareLaunchArgument("save_frames_every", default_value="1"),
        DeclareLaunchArgument("max_speed_mps", default_value="1.0"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument("max_steering_rad", default_value="0.34"),
        DeclareLaunchArgument("wheelbase_m", default_value="0.25"),
        DeclareLaunchArgument("lateral_controller", default_value="pure_pursuit"),
        DeclareLaunchArgument("pid_gain", default_value="1.0"),
        DeclareLaunchArgument("steer_smoothing_sec", default_value="0.0"),
        DeclareLaunchArgument("control_hz", default_value="20.0"),
        DeclareLaunchArgument("max_plan_age_sec", default_value="3.0"),
        DeclareLaunchArgument("start_estopped", default_value="false"),
        DeclareLaunchArgument("creep_after_sec", default_value="0.0"),
        DeclareLaunchArgument("creep_speed_mps", default_value="1.0"),
        DeclareLaunchArgument("creep_release_speed_mps", default_value="0.3"),
        DeclareLaunchArgument("creep_hold_speed_mps", default_value="0.5"),
        DeclareLaunchArgument("creep_duration_sec", default_value="2.0"),
        DeclareLaunchArgument("min_speed_mps", default_value="0.0"),
    ]

    def _f(name):
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    def _b(name):
        return ParameterValue(LaunchConfiguration(name), value_type=bool)

    def _i(name):
        return ParameterValue(LaunchConfiguration(name), value_type=int)

    cuda_home = "/usr/local/cuda-12.6"
    ld_library_path = ":".join([
        f"{cuda_home}/targets/aarch64-linux/lib", f"{cuda_home}/lib64",
        "/usr/lib/aarch64-linux-gnu", "/usr/lib/aarch64-linux-gnu/tegra",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ])

    camera = Node(
        package="simlingo_f1tenth", executable="camera_node", name="camera_node", output="screen",
        condition=IfCondition(LaunchConfiguration("use_camera_node")),
        parameters=[{
            "source":       LaunchConfiguration("camera_source"),
            "device":       LaunchConfiguration("camera_device"),
            "gst_pipeline": LaunchConfiguration("gst_pipeline"),
            "width":        _i("camera_width"),
            "height":       _i("camera_height"),
            "fps":          _f("camera_fps"),
            "topic":        "/camera/front/image",
        }],
    )

    inference = Node(
        package="simlingo_f1tenth", executable="simlingo_realworld_node",
        name="simlingo_realworld_node", output="screen",
        parameters=[{
            "checkpoint_path":     LaunchConfiguration("checkpoint_path"),
            "simlingo_path":       LaunchConfiguration("simlingo_path"),
            "fast_inference":      _b("fast_inference"),
            "use_cot":             _b("use_cot"),
            "save_frames_dir":     ParameterValue(LaunchConfiguration("save_frames_dir"), value_type=str),
            "save_frames_every":   _i("save_frames_every"),
            "image_topic":         LaunchConfiguration("image_topic"),
            "pose_topic":          LaunchConfiguration("pose_topic"),
            "speed_topic":         LaunchConfiguration("speed_topic"),
            "route_topic":         LaunchConfiguration("route_topic"),
            "route_csv":           LaunchConfiguration("route_csv"),
            "route_loop":          _b("route_loop"),
            "route_min_spacing_m": _f("route_min_spacing_m"),
            "world_scale":         _f("world_scale"),
            "speed_world_scale":   _f("speed_world_scale"),
            "pose_offset_x":       _f("pose_offset_x"),
            "plan_topic":          "/simlingo/plan",
        }],
        additional_env={"CUDA_HOME": cuda_home, "LD_LIBRARY_PATH": ld_library_path},
    )

    controller = Node(
        package="simlingo_f1tenth", executable="trajectory_controller_node",
        name="trajectory_controller", output="screen",
        parameters=[{
            "plan_topic":         "/simlingo/plan",
            "pose_topic":         LaunchConfiguration("pose_topic"),
            "speed_topic":        LaunchConfiguration("speed_topic"),
            "drive_topic":        LaunchConfiguration("drive_topic"),
            "pose_offset_x":      _f("pose_offset_x"),
            "control_hz":         _f("control_hz"),
            "max_plan_age_sec":   _f("max_plan_age_sec"),
            "start_estopped":     _b("start_estopped"),
            "creep_after_sec":    _f("creep_after_sec"),
            "creep_speed_mps":    _f("creep_speed_mps"),
            "creep_release_speed_mps": _f("creep_release_speed_mps"),
            "creep_hold_speed_mps": _f("creep_hold_speed_mps"),
            "creep_duration_sec": _f("creep_duration_sec"),
            "min_speed_mps":      _f("min_speed_mps"),
            "max_speed_mps":      _f("max_speed_mps"),
            "speed_scale":        _f("speed_scale"),
            "max_steering_rad":   _f("max_steering_rad"),
            "wheelbase_m":        _f("wheelbase_m"),
            "lateral_controller": LaunchConfiguration("lateral_controller"),
            "pid_gain":           _f("pid_gain"),
            "steer_smoothing_sec": _f("steer_smoothing_sec"),
            "world_scale":        _f("world_scale"),
        }],
    )

    return LaunchDescription(args + [
        SetEnvironmentVariable("CUDA_HOME", cuda_home),
        SetEnvironmentVariable("LD_LIBRARY_PATH", ld_library_path),
        camera,
        controller,      # starts publishing zero-speed /drive immediately
        inference,       # blocks ~60-90 s while the model loads
    ])
