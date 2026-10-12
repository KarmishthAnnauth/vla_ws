from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Launch the ORION node on live CARLA camera + odometry topics.

    NOTE: topic names are placeholders — set the real CARLA ROS bridge topic
    names together later. The node subscribes to all six ORION cameras and feeds
    each view into its own input slot, as OrionAgent.tick() does.
    """
    # The six ORION cameras, in ORION_CAMERA_ORDER. Images are RAW
    # sensor_msgs/Image (rgba8/bgra8), same as the SimLingo reference node.
    # carla_ros_bridge publishes /carla/<role_name>/<sensor id>/image, so these
    # names require the CARLA-side agent's sensors() to declare all six ids
    # exactly as OrionAgent.sensors() does (CAM_FRONT, CAM_FRONT_LEFT,
    # CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT). If that agent
    # still spawns a single camera, only CAM_FRONT will ever publish and the node
    # will sit waiting on the other five rather than silently faking them.
    #
    # NOTE: six raw 1600x900 streams is ~6x the bandwidth of one. Over the
    # ZeroTier link use orion_withpid.launch.py, which moves JPEG across the wire
    # and decompresses locally.
    camera_topics = [
        "/carla/hero/CAM_FRONT/image",
        "/carla/hero/CAM_FRONT_LEFT/image",
        "/carla/hero/CAM_FRONT_RIGHT/image",
        "/carla/hero/CAM_BACK/image",
        "/carla/hero/CAM_BACK_LEFT/image",
        "/carla/hero/CAM_BACK_RIGHT/image",
    ]

    # Only dispatch inference when a NEW front frame has arrived since the
    # last run (True), vs. run continuously on whatever is latest to benchmark
    # raw model throughput (False). Override on the command line:
    #   ros2 launch orion_ros orion.launch.py require_new_frame:=false
    require_new_frame_arg = DeclareLaunchArgument(
        "require_new_frame",
        default_value="true",
        description="Gate inference on a fresh front frame (true) or run "
        "continuously on the latest data (false).",
    )

    # The ORION config hardcodes the tokenizer/LLM weights as a RELATIVE path
    # ('ckpts/pretrain_qformer/'), resolved against orion_repo_path (/root/Orion).
    # The actual weights live at /models/Orion/pretrain_qformer/. Rather than
    # edit the Orion repo, we (re)create the symlink /root/Orion/ckpts ->
    # /models/Orion at launch time so the relative path resolves. `-sfn`
    # makes it idempotent across restarts (force, no-deref of an existing link).
    # If this is missing, transformers treats 'ckpts/pretrain_qformer/' as a HF
    # repo id and dies with HFValidationError.
    link_ckpts = ExecuteProcess(
        cmd=["ln", "-sfn", "/models/Orion", "/root/Orion/ckpts"],
        output="screen",
    )
    orion_node = Node(
        package="orion_ros",
        executable="orion_node",
        name="orion_node",
        output="screen",
        parameters=[
            {
                        # --- ORION repo / model (container paths) ---
                        # Use the container's PRE-BUILT Orion at /root/Orion — it has the
                        # compiled mmcv._ext ops. The bind-mounted /benchmarking/Orion is
                        # source-only (no compiled ops) and must NOT be used here, or
                        # `import mmcv` fails with "No module named 'mmcv._ext'".
                        # The node chdir's into this dir, so the config's relative ckpts/
                        # paths resolve against it.
                        "orion_repo_path": "/root/Orion",
                        "orion_config_path":
                            "/root/Orion/adzoo/orion/configs/orion_stage3_agent.py",
                        # Weights live at /models/Orion (pretrain_qformer/, Orion.pth),
                        # the weights drive mount in this container — see `mount`.
                        "orion_checkpoint_path": "/models/Orion/Orion.pth",
                        # NOTE: the config hardcodes llm_path='ckpts/pretrain_qformer/',
                        # resolved against orion_repo_path. Link the weights in (in the
                        # container):  ln -s /models/Orion /root/Orion/ckpts
                        "precision": "fp16",  # "fp16" (faster) or "fp32"
                        # --- Topics (CARLA ros-bridge, role_name = hero) ---
                        "camera_topics": camera_topics,
                        "speed_topic": "/carla/hero/speed",
                        "odometry_topic": "/carla/hero/odometry",
                        "imu_topic": "/carla/hero/imu",
                        "route_topic": "/carla/hero/global_plan",
                        "trajectory_topic": "/orion/predicted_trajectory",
                        "cot_topic": "/orion/cot",
                        # --- Behaviour ---
                        "inference_period_sec": 0.05,  # dispatch-check interval
                        "require_new_frame": ParameterValue(
                            LaunchConfiguration("require_new_frame"), value_type=bool
                        ),
                        "publish_cot": True,           # silent on planning-only config
                        "driving_command": 4,          # follow-lane
                        "replicate_jpeg_quality": 20,  # matches the CARLA agent
                    }
                ],
            )

    # ── stanley_controller_node ───────────────────────────────────────────────
    # Consumes the Autoware trajectory published by orion_node and produces
    # AckermannDrive commands. carla_ackermann_control subscribes to
    # control_output_topic and converts AckermannDrive →
    # /carla/hero/vehicle_control_cmd for the leaderboard agent.
    stanley_node = Node(
        package="orion_ros",
        executable="stanley_controller_node",
        name="stanley_controller",
        output="screen",
        parameters=[
            {
                "trajectory_topic":       "/orion/predicted_trajectory",
                "odometry_topic":         "/carla/hero/odometry",
                "control_output_topic":   "/carla/hero/ackermann_cmd",
                "stanley_gain_k":         1.5,
                "stanley_softening_ks":   0.5,
                "wheelbase_m":            2.875,
                "max_steering_rad":       0.6,
                "speed_lookahead_steps":  5,
                "control_hz":             20.0,
                "max_acceleration_mps2":  3.0,
                # Generous timeout so Stanley coasts safely between inferences.
                "trajectory_timeout_sec": 5.0,
            }
        ],
    )

    return LaunchDescription(
        [
            require_new_frame_arg,
            # Create the ckpts symlink first, then start orion_node only once the
            # `ln` process has exited so the weights path is guaranteed to resolve.
            link_ckpts,
            RegisterEventHandler(
                OnProcessExit(target_action=link_ckpts, on_exit=[orion_node])
            ),
            # Stanley is lightweight; delay start until orion_node has had time to
            # load the model (~60-90 s on Orin) and publish a first trajectory.
            TimerAction(period=90.0, actions=[stanley_node]),
        ]
    )
