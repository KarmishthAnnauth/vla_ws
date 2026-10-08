# orion_f1tenth — ORION on a real F1TENTH car

Runs ORION (Fu et al., ICCV'25) on the Jetson AGX Orin against the camera attached to the
Orin and the F1TENTH stack (`vla_ws`) on the Jetson Nano, and drives the car through the
stack's `/drive` topic. Sibling of `simlingo_f1tenth`, whose camera node, route planner and
trajectory controller it reuses; only the inference node is ORION's.

Reference implementation: `orion_ros/orion_withpid_node.py` (CARLA-in-the-loop), which
reproduces `Orion/team_code/orion_b2d_agent.py::OrionAgent.run_step`. Nothing in either
was changed.

```
 Jetson Nano (vla_ws, ROS 2 Foxy)                Jetson AGX Orin (ROS 2 Humble, orion_env_ros container)
 ─────────────────────────────────                ──────────────────────────────────────────────────────
 vesc_to_odom      /odom (speed, yaw rate) ─────► orion_realworld_node ◄── camera_node (/dev/video0)
 particle_filter   /pf/pose/odom (map pose) ────►   │   image + pose + speed + route → six-slot payload → model
 waypoint_visual.  /global_path (nav_msgs/Path) ─►   │   → /orion/plan  (Trajectory, map frame, real units)
                                                     ▼
                                                  trajectory_controller_node  (simlingo_f1tenth, 20 Hz)
 ackermann_mux ◄── /drive (AckermannDriveStamped) ───┘   pure pursuit on the plan, plan speed capped
```

Files: `orion_model.py` (model loading, speed profiles, the agent's payload, forward; no
ROS), `orion_realworld_node.py` (sensors in, plan out), `launch/orion_f1tenth.launch.py`.

## What ORION needs, and where it comes from

The payload is built exactly as the agent builds it: the same `results` dict field for
field (calibration constants copied verbatim, `can_bus`, `ego_pose`, command), then ORION's
own `Compose(cfg.inference_only_pipeline)` and `mmcv.parallel.collate`, then
`model(batch, return_loss=False)`. Only the raw inputs differ.

| model input | agent (CARLA) | here |
|---|---|---|
| 6 cameras, 1600×900, BGR, JPEG q20 | six mounts: front/sides FOV 70°, back 110° | **one** camera. The frame is centre-cropped to 16:9 (optionally to 70° first, `camera_hfov_deg`), resized to 1600×900, JPEG-q20 re-encoded like the agent, and put in `CAM_FRONT`, `CAM_FRONT_LEFT` and `CAM_FRONT_RIGHT` (`side_views:=copy`, default) — the model is told the same pixels sit at yaw 0/−55/+55°. `CAM_BACK*` are black. `side_views:=black` fills only `CAM_FRONT`. |
| `can_bus[0:2]`, `ego_pose` | GNSS at x=−1.4 m of the vehicle origin (≈ rear axle) | `/pf/pose/odom` moved from the laser to base_link (`pose_offset_x` −0.27 m, the rear axle) × `world_scale`. No further offset (`gnss_mount_offset_x` 0). |
| `can_bus[7]` speed | speedometer | `/odom` `twist.linear.x` × `speed_world_scale` |
| `can_bus[10:16]` accel, gyro | IMU | zeros and `/odom` `twist.angular.z` as yaw rate; or `imu_topic` (accel × scale, gyro y flipped as the reference node does for the ros-bridge) |
| `timestamp` (temporal memory) | step/20 | the frame's stamp (`timestamp_mode:=sensor`); memory is kept while frames are < 2 s apart, i.e. always at ~1 s per frame |
| `ego_fut_cmd` RoadOption | route planner over the CARLA global plan | `command_source:=geometry` (default): LEFT/RIGHT where the route turns by more than `command_turn_deg` within `command_lookahead_m`, else LANEFOLLOW; `command_source:=static` feeds `driving_command` (default 4 = LANEFOLLOW) every frame |
| route progress | `RoutePlanner(4.0, 50.0)` | `simlingo_f1tenth.RoutePlanner` with the same window, in model metres (0.4 / 5 m on the floor at scale 10), on `/global_path` or `route_csv` |

Output: `ego_fut_preds`, 6 waypoints at 0.5 s in ORION's LIDAR_TOP frame (index 1
forward, index 0 **right**). They are divided by `world_scale`, mirrored to the ROS ego frame,
the ego origin is prepended and the last segment extrapolated `plan_extend_sec` (2 s) past
the 3 s horizon — a ~1 s forward pass eats a third of the plan before it arrives and another
third before the next one lands — then anchored in the map at the pose the image was taken
from and published as a Trajectory. The scalar desired speed is derived exactly as
Bench2Drive's `control_pid` does (`0.75·|wp0|·2 + 0.25·|wp1−wp0|·2`), divided by
`speed_world_scale`. The agent's own PID is not used: the VESC takes a speed and a steering
angle, which `trajectory_controller_node` produces (see `simlingo_f1tenth/README.md`).

## Inference variants (`inference_mode`)

All are post-build transforms from `orion_ros/orion_speedups.py` (ORION sources untouched);
numbers from `orion_ros/ORION_ROS_NODE.md` section 6, six real 1600×900 views, clocks pinned.

| mode | what | forward | trajectory vs. stock |
|---|---|---:|---:|
| `fast` (default) | merge LoRA, flash-attn prefill, fused ViT glue, map-head slice, down-proj layout, `torch.compile` of ViT/LLM/heads, prep of frame N+1 during forward N | ~0.93 s (period ~1.0 s) | < 1 cm |
| `int8` | `fast` + W8A8 INT8 for the LLM MLP projections (SmoothQuant α 0.8, layers 0,1,30,31 fp16; stats in `orion_ros/engines/llm_act_stats.pt`) | ~0.85 s (period ~0.93 s) | 3–5 cm |
| `lite` | `fast` + 512 px ViT input, rear views every other frame | ~0.7 s | 0.47 m |
| `eager` | the exact speedups without `torch.compile` (no compile wait) | ~1.5 s | < 1 cm |
| `baseline` | the agent as is | ~1.75 s | — |

Measured with this package on 2026-10-06 (real model, synthetic camera, fake pose/odometry,
clocks pinned, Intenso SSD; logs in `vla_ws/log/orion/test_synthetic_*_20261006.log`):

| mode | weights load | warm-up (cached compile) | forward, frames 2-20 | period between forwards |
|---|---:|---:|---:|---:|
| `fast` | 187 s | 81 s | 967 ms mean (953-983) | ~970 ms |
| `int8` | 182 s | 87 s | 898 ms mean (892-917) | ~900 ms |

The compiled modes spend ~3 minutes compiling in the warm-up forward the first time per
variant; the inductor cache (`/benchmarking/.torchinductor_cache`, shared with the CARLA
node) brought both warm-ups above under 90 s. Three black views cost the same as three
real ones: the ViT runs on all six.

## World scale

As for SimLingo (`world_scale` 10, `speed_world_scale` 0 = `world_scale`). ORION differs in
one respect: its temporal memory stores ego poses in model metres against real time, so a
speed factor other than the distance scale tells the model a speed that does not match the
displacement it sees between frames. Keep them equal unless the model's speed will not
move the car; then prefer the controller's `min_speed_mps`.

## Running

```bash
# Orin: benchmarking/start_orion_f1tenth.sh does all of this (container, weights, build, clocks)
./start_orion_f1tenth.sh                       # fast
./start_orion_f1tenth.sh --int8                # quantised LLM
./start_orion_f1tenth.sh --fast camera_hfov_deg:=110.0   # crop a 110° camera to the model's 70°

# no camera / no car yet: exercise the pipeline (the model still loads)
ros2 launch orion_f1tenth orion_f1tenth.launch.py camera_source:=synthetic route_csv:=/path/route.csv
ros2 topic pub /pf/pose/odom nav_msgs/msg/Odometry "{header: {frame_id: map}, pose: {pose: {orientation: {w: 1.0}}}}" -r 20
ros2 topic pub /odom nav_msgs/msg/Odometry "{twist: {twist: {linear: {x: 0.5}}}}" -r 20
ros2 topic echo /drive
```

Software stop: `ros2 topic pub /orion/estop std_msgs/msg/Bool "{data: true}" -1`.

On the real car use `vla_ws/scripts/run_orion_route.sh <N> [--int8]`: it sets up DDS on both
sides, starts the car's bringup and localisation, and launches with `start_estopped:=true`,
the stuck recovery and the speed floor of `run_simlingo_route.sh`.

Every plan is logged as `[ORION] frame N cmd ... | prep .. ms, forward .. ms | period .. ms`
followed by the six waypoints in the ROS ego frame, real metres.

## Known limits

* Three front slots from one camera and three black rear slots are **not** what the model
  was trained on: the detections and map it hallucinates beside and behind the car are
  unconstrained. A wide camera cropped into the three front sectors would be the next step.
* The 0.39 m (3.9 cm real) between ORION's lidar origin and its ego point is ignored, as in
  the reference node.
* Trajectory shifts quoted above were measured on CARLA frames with six real views.
