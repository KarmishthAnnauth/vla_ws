"""
ORION on a real F1TENTH: camera + inference + trajectory controller (Orin side).

  camera_node (simlingo_f1tenth)   /camera/front/image/compressed ──┐
  Nano: particle_filter            /pf/pose/odom  (map frame)  ─────┼─► orion_realworld_node
  Nano: vesc_to_odom               /odom          (speed)      ─────┤        │  /orion/plan (map frame)
  Nano: waypoint_visualiser        /global_path   (nav_msgs/Path) ──┘        ▼
                                                    trajectory_controller_node (simlingo_f1tenth)
                                                                             │  /drive (AckermannDriveStamped)
                                                                             ▼
                                                         Nano: ackermann_mux -> ackermann_to_vesc -> VESC

Launch arguments (all optional):
  inference_mode       fast  = exact fp16 speedups + torch.compile, ~1.0 s per frame (default)
                       int8  = fast + W8A8 INT8 LLM MLPs (SmoothQuant), ~0.9 s, 3-5 cm trajectory shift
                       lite  = fast + 512 px ViT, rear views every other frame, ~0.75 s (0.47 m shift)
                       eager = exact speedups without torch.compile, ~1.5 s (no compile wait)
                       baseline = the agent as is, ~1.75 s
  precision            fp16 (default) | fp32 (does not fit next to 61 GB of unified memory)
  orion_repo_path      container's pre-built Orion checkout (compiled mmcv ops)
  orion_checkpoint_path, llm_int8_stats
  world_scale          model metres per real metre (10 for the 1:10 replica track)
  speed_world_scale    model m/s per real m/s; 0 = world_scale (keep them equal: temporal memory)
  route_csv            x,y[,v] CSV in the map frame; empty = wait for /global_path
  route_loop           true for a closed circuit
  command_source       geometry (default: LEFT/RIGHT from the route) | static (driving_command, default LANEFOLLOW=4)
  camera_hfov_deg      horizontal FOV of the camera; > 70 crops the central 70 deg (0 = no crop)
  side_views           copy (camera in the three front slots) | black (front slot only)
  camera_source        v4l2 | gstreamer | synthetic   (synthetic = no camera needed)
  camera_device        /dev/video0
  use_camera_node      false if the camera is published by something else
  plan_extend_sec      extrapolate the 3 s plan this long for the controller (default 2)
  max_speed_mps        hard cap on commanded speed, except the kick of the stuck recovery (start low)
  lateral_controller   pure_pursuit | simlingo_pid
  steer_smoothing_sec  low-pass time constant on the steering command, 0 = off
  control_hz           controller rate (keep >= 10: the mux times out after 0.2 s)
  start_estopped       true: hold zero speed until /orion/estop receives false
  creep_after_sec      > 0: stuck recovery (see simlingo_f1tenth), min_speed_mps: floor while driving
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("inference_mode", default_value="fast",
                              description="baseline | eager | fast | int8 | lite"),
        DeclareLaunchArgument("precision", default_value="fp16"),
        DeclareLaunchArgument("orion_repo_path", default_value="/root/Orion"),
        DeclareLaunchArgument("orion_config_path",
                              default_value="/root/Orion/adzoo/orion/configs/orion_stage3_agent.py"),
        DeclareLaunchArgument("orion_checkpoint_path", default_value="/models/Orion/Orion.pth"),
        DeclareLaunchArgument("llm_int8_stats",
                              default_value="/benchmarking/alpamayo-autoware/src/orion_ros/engines/llm_act_stats.pt"),
        DeclareLaunchArgument("profile_stages", default_value="false"),
        DeclareLaunchArgument("pipeline_prep", default_value="true"),
        DeclareLaunchArgument("timestamp_mode", default_value="sensor"),
        DeclareLaunchArgument("fake_model", default_value="false",
                              description="true: no ORION, straight plans (pipeline tests without the weights)"),
        DeclareLaunchArgument("save_frames_dir", default_value="",
                              description="directory for one JPEG per plan (frame + plan overlay); empty = off"),
        DeclareLaunchArgument("save_frames_every", default_value="1"),
        DeclareLaunchArgument("world_scale", default_value="10.0"),
        DeclareLaunchArgument("speed_world_scale", default_value="0.0"),
        DeclareLaunchArgument("route_csv", default_value=""),
        DeclareLaunchArgument("route_loop", default_value="false"),
        DeclareLaunchArgument("route_min_spacing_m", default_value="0.0"),
        DeclareLaunchArgument("command_source", default_value="geometry"),
        DeclareLaunchArgument("driving_command", default_value="4"),
        DeclareLaunchArgument("command_lookahead_m", default_value="15.0"),
        DeclareLaunchArgument("command_turn_deg", default_value="35.0"),
        DeclareLaunchArgument("camera_hfov_deg", default_value="0.0"),
        DeclareLaunchArgument("side_views", default_value="copy"),
        DeclareLaunchArgument("replicate_jpeg_quality", default_value="20"),
        DeclareLaunchArgument("plan_extend_sec", default_value="2.0"),
        DeclareLaunchArgument("image_topic", default_value="/camera/front/image/compressed"),
        DeclareLaunchArgument("pose_topic", default_value="/pf/pose/odom"),
        DeclareLaunchArgument("speed_topic", default_value="/odom"),
        DeclareLaunchArgument("imu_topic", default_value=""),
        DeclareLaunchArgument("route_topic", default_value="/global_path"),
        DeclareLaunchArgument("drive_topic", default_value="/drive"),
        DeclareLaunchArgument("estop_topic", default_value="/orion/estop"),
        DeclareLaunchArgument("pose_offset_x", default_value="-0.27"),
        DeclareLaunchArgument("gnss_mount_offset_x", default_value="0.0"),
        DeclareLaunchArgument("use_camera_node", default_value="true"),
        DeclareLaunchArgument("camera_source", default_value="v4l2"),
        DeclareLaunchArgument("camera_device", default_value="/dev/video0"),
        DeclareLaunchArgument("camera_width", default_value="1280"),
        DeclareLaunchArgument("camera_height", default_value="720"),
        DeclareLaunchArgument("camera_fps", default_value="30.0"),
        DeclareLaunchArgument("gst_pipeline", default_value=""),
        DeclareLaunchArgument("max_speed_mps", default_value="1.0"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument("max_steering_rad", default_value="0.34"),
        DeclareLaunchArgument("wheelbase_m", default_value="0.25"),
        DeclareLaunchArgument("lateral_controller", default_value="pure_pursuit"),
        DeclareLaunchArgument("pid_gain", default_value="1.0"),
        DeclareLaunchArgument("steer_smoothing_sec", default_value="0.0"),
        DeclareLaunchArgument("control_hz", default_value="20.0"),
        DeclareLaunchArgument("max_plan_age_sec", default_value="4.0"),
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

    def _s(name):
        return ParameterValue(LaunchConfiguration(name), value_type=str)

    # The ORION config hardcodes the LLM/tokenizer weights as the RELATIVE path
    # 'ckpts/pretrain_qformer/' under orion_repo_path; the weights live at
    # /models/Orion. Same symlink as orion_ros' launch files (-sfn: idempotent).
    link_ckpts = ExecuteProcess(
        cmd=["ln", "-sfn", "/models/Orion", "/root/Orion/ckpts"], output="screen")

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
        package="orion_f1tenth", executable="orion_realworld_node",
        name="orion_realworld_node", output="screen",
        parameters=[{
            "orion_repo_path":       _s("orion_repo_path"),
            "orion_config_path":     _s("orion_config_path"),
            "orion_checkpoint_path": _s("orion_checkpoint_path"),
            "precision":             _s("precision"),
            "inference_mode":        _s("inference_mode"),
            "llm_int8_stats":        _s("llm_int8_stats"),
            "profile_stages":        _b("profile_stages"),
            "pipeline_prep":         _b("pipeline_prep"),
            "timestamp_mode":        _s("timestamp_mode"),
            "fake_model":            _b("fake_model"),
            "save_frames_dir":       ParameterValue(LaunchConfiguration("save_frames_dir"), value_type=str),
            "save_frames_every":     _i("save_frames_every"),
            "image_topic":           _s("image_topic"),
            "pose_topic":            _s("pose_topic"),
            "speed_topic":           _s("speed_topic"),
            "imu_topic":             _s("imu_topic"),
            "route_topic":           _s("route_topic"),
            "route_csv":             _s("route_csv"),
            "route_loop":            _b("route_loop"),
            "route_min_spacing_m":   _f("route_min_spacing_m"),
            "command_source":        _s("command_source"),
            "driving_command":       _i("driving_command"),
            "command_lookahead_m":   _f("command_lookahead_m"),
            "command_turn_deg":      _f("command_turn_deg"),
            "world_scale":           _f("world_scale"),
            "speed_world_scale":     _f("speed_world_scale"),
            "pose_offset_x":         _f("pose_offset_x"),
            "gnss_mount_offset_x":   _f("gnss_mount_offset_x"),
            "camera_hfov_deg":       _f("camera_hfov_deg"),
            "side_views":            _s("side_views"),
            "replicate_jpeg_quality": _i("replicate_jpeg_quality"),
            "plan_extend_sec":       _f("plan_extend_sec"),
            "plan_topic":            "/orion/plan",
        }],
        additional_env={
            # torch.compile artefacts survive container removal; inductor would
            # otherwise recompile from scratch (minutes) on every launch.
            "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/benchmarking/.torchinductor_cache"),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        },
    )

    controller = Node(
        package="simlingo_f1tenth", executable="trajectory_controller_node",
        name="trajectory_controller", output="screen",
        parameters=[{
            "plan_topic":         "/orion/plan",
            "pose_topic":         _s("pose_topic"),
            "speed_topic":        _s("speed_topic"),
            "drive_topic":        _s("drive_topic"),
            "estop_topic":        _s("estop_topic"),
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
            "lateral_controller": _s("lateral_controller"),
            "pid_gain":           _f("pid_gain"),
            "steer_smoothing_sec": _f("steer_smoothing_sec"),
            "world_scale":        _f("world_scale"),
        }],
    )

    return LaunchDescription(args + [
        SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}"),
        camera,
        controller,      # starts publishing zero-speed /drive immediately
        link_ckpts,
        # the inference node only once the ckpts symlink exists (model load: minutes)
        RegisterEventHandler(OnProcessExit(target_action=link_ckpts, on_exit=[inference])),
    ])
