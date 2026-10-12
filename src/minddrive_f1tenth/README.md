# minddrive_f1tenth — MindDrive on a real F1TENTH car

Runs MindDrive (Fu et al., ECCV'26) on the Jetson AGX Orin against the camera attached to
the Orin and the F1TENTH stack (`vla_ws`) on the Jetson Nano, and drives the car through the
stack's `/drive` topic. Sibling of `orion_f1tenth` and `simlingo_f1tenth`, whose camera node,
route planner and trajectory controller it reuses; only the inference node is MindDrive's.

Reference implementation: `minddrive_ros/minddrive_node.py` (CARLA-in-the-loop), which
reproduces `MindDrive/team_code/minddrive_b2d_agent.py::MinddriveAgent.run_step`. Nothing in
either was changed; this package imports `minddrive_ros.agent_constants` (the model's input
contract) and `minddrive_ros.minddrive_speedups` (the bench-validated speedups).

```
 Jetson Nano (vla_ws, ROS 2 Foxy)                Jetson AGX Orin (ROS 2 Humble, minddrive_env_ros container)
 ─────────────────────────────────                ──────────────────────────────────────────────────────────
 vesc_to_odom      /odom (speed, yaw rate) ─────► minddrive_realworld_node ◄── camera_node (/dev/video0)
 particle_filter   /pf/pose/odom (map pose) ────►   │   image + pose + speed + route → six-slot payload → model
 waypoint_visual.  /global_path (nav_msgs/Path) ─►   │   → /minddrive/plan  (Trajectory, map frame, real units)
                                                     ▼
                                                  trajectory_controller_node  (simlingo_f1tenth, 20 Hz)
 ackermann_mux ◄── /drive (AckermannDriveStamped) ───┘   pure pursuit on the plan, plan speed capped
```

Files: `minddrive_model.py` (model loading, speed profiles, the agent's payload, forward; no
ROS), `minddrive_realworld_node.py` (sensors in, plan out), `frame_saver.py`,
`launch/minddrive_f1tenth.launch.py`.

## What MindDrive needs, and where it comes from

The payload is built exactly as the agent builds it: `agent_constants.build_agent_results`
(the same `results` dict field for field, calibration constants verbatim), then MindDrive's own
`Compose(cfg.inference_only_pipeline)` and `mmcv.parallel.collate`, the agent's H2D loop,
`custom_wrap_fp16_model` before every forward, `model(batch, return_loss=False)`. Only the raw
inputs differ. MindDrive's sensor rig is ORION's, so the table is the ORION one:

| model input | agent (CARLA) | here |
|---|---|---|
| 6 cameras, 1600×900, BGR, JPEG q20 | six mounts: front/sides FOV 70°, back 110° | **one** camera. The frame is centre-cropped to 16:9 (optionally to 70° first, `camera_hfov_deg`), resized to 1600×900, JPEG-q20 re-encoded like the agent, and put in `CAM_FRONT`, `CAM_FRONT_LEFT` and `CAM_FRONT_RIGHT` (`side_views:=copy`, default). `CAM_BACK*` are black. `side_views:=black` fills only `CAM_FRONT`. |
| `can_bus[0:2]`, `ego_pose` | GNSS at x=−1.4 m of the vehicle origin (≈ rear axle) | `/pf/pose/odom` moved from the laser to base_link (`pose_offset_x` −0.27 m, the rear axle) × `world_scale`. No further offset (`gnss_mount_offset_x` 0). |
| `can_bus[7]` speed | speedometer | `/odom` `twist.linear.x` × `speed_world_scale` |
| `can_bus[10:16]` accel, gyro | IMU | zeros and `/odom` `twist.angular.z` as yaw rate; or `imu_topic` |
| `timestamp` (temporal memory) | step/20 | the frame's stamp (`timestamp_mode:=sensor`); memory is kept while frames are < 2 s apart, i.e. always at ~1 s per frame |
| `ego_fut_cmd` RoadOption | `RoutePlanner(4.0, 50.0)` over the CARLA global plan | `command_source:=geometry` (default): LEFT/RIGHT where the route turns by more than `command_turn_deg` within `command_lookahead_m`, else LANEFOLLOW; `command_source:=static` feeds `driving_command` (default 4) |
| route progress | `RoutePlanner(4.0, 50.0)` | `simlingo_f1tenth.RoutePlanner` with the same window, in model metres (0.4 / 5 m on the floor at scale 10), on `/global_path` or `route_csv` |

## Two trajectories, one plan

MindDrive predicts two things per frame, both in its LIDAR_TOP frame (index 1 forward, index 0
**right**), plus the decision expert's discrete meta-actions:

| output | shape | the agent uses it for |
|---|---|---|
| `ego_fut_preds` — the **speed trajectory** | (6, 2), 0.5 s steps, 3 s horizon | `control_pid`'s desired speed: `0.75·|wp0|·2 + 0.25·|wp1−wp0|·2` |
| `pw_ego_fut_pred` — the **path trajectory** | (20, 2), a polyline, no time axis | `control_pid`'s aim point (steering) |
| `speed_value`, `path_value` | ints | logged ("maintain slow speed", "turn left", ...) |

The agent's own PID is not used (the VESC takes a speed and a steering angle, which
`trajectory_controller_node` produces), but the plan it gets is built the way the PID reads the
model: with `plan_source:=path` (default) the plan's geometry is the **path** trajectory and
every point carries the desired speed derived from the **speed** trajectory with the PID's
formula; `plan_source:=speed` publishes the six timed waypoints instead, as the ORION node
does. Either way the points are divided by `world_scale`, mirrored to the ROS ego frame, the
ego origin is prepended and the plan is extended past its end (`plan_extend_sec`, 2 s: the
path by desired speed × 2 s along its last segment, the speed trajectory at its last segment's
velocity) because a ~1 s forward pass eats a third of the plan before it arrives — then
anchored in the map at the pose the image was taken from and published as a Trajectory. A
zero-length path (the model wants to stop) gives a plan with every point at the ego, which the
controller treats as "plan exhausted" and commands zero speed. The meta-actions go out on
`/minddrive/meta_action` and into the log line.

## Inference variants (`inference_mode`)

All are post-build transforms from `minddrive_ros/minddrive_speedups.py` (MindDrive sources
untouched); numbers from `minddrive_ros/MINDDRIVE_ROS_NODE.md` section 6, 3B, six real
1600×900 views, clocks pinned. Trajectory shifts are in model metres (divide by 10 for the
floor).

| mode | what | forward | trajectory vs. fp32 reference |
|---|---|---:|---:|
| `fast` (default) | `fp16_infer`, merged LoRA, ViT glue + weight layouts + unpadded window attention, map-head slice, one-row decision logits, `torch.compile` of ViT and LLM, CUDA-graphed ViT, prep of frame N+1 during forward N (`start_minddrive.sh --fast`) | ~0.87 s | 0.05 m speed / 0.09 m path |
| `int8` | `fast` + W8A8 INT8 for the LLM projections of both experts (SmoothQuant α 0.5, stats in `/benchmarking/minddrive_int8_calib.pt`) | ~0.82 s | 0.41 m / 0.47 m — a precision change |
| `lite` | `fast` + 512 px ViT input, rear views every other frame | ~0.65 s | 1.8 m / 2.6 m — a different model in effect |
| `eager` | the fp16 exact set without `torch.compile` / CUDA graphs (no compile wait) | ~1.4 s | 0.06 m / 0.04 m |
| `baseline` | the reference config as is: fp32, nothing | ~2.3 s | — |

`speedup_overrides:="key=value,..."` changes single knobs on top of a profile with the names of
`minddrive_ros` (e.g. `compile_targets=,cuda_graph_vit=false` for the fast set in eager mode,
`int8_alpha=0.7`). `variant:=05b` loads the 0.5B model (78.04 DS) instead of the 3B (80.59 DS).

Not measured yet with this package on the Orin (the GPU was busy when it was written): the
table is the CARLA node's bench. The compiled modes spend minutes compiling in the warm-up the
first time per variant; the inductor cache (`/benchmarking/.torchinductor_cache`, shared with
the CARLA node) makes later starts fast. Three black views cost the same as three real ones:
the ViT runs on all six.

## World scale

As for ORION (`world_scale` 10, `speed_world_scale` 0 = `world_scale`). MindDrive's temporal
memory stores ego poses in model metres against real time, so a speed factor other than the
distance scale tells the model a speed that does not match the displacement it sees between
frames. Keep them equal unless the model's speed will not move the car; then prefer the
controller's `min_speed_mps`.

## Running

```bash
# Orin: benchmarking/start_minddrive_f1tenth.sh does all of this (container, weights, checkout, build, clocks)
./start_minddrive_f1tenth.sh                       # 3B, fast
./start_minddrive_f1tenth.sh --int8                # quantised LLM
./start_minddrive_f1tenth.sh --05b --eager         # the 0.5B model, no compile wait
./start_minddrive_f1tenth.sh camera_hfov_deg:=110.0 plan_source:=speed

# no camera / no car yet: exercise the pipeline (the model still loads)
ros2 launch minddrive_f1tenth minddrive_f1tenth.launch.py camera_source:=synthetic route_csv:=/path/route.csv
ros2 topic pub /pf/pose/odom nav_msgs/msg/Odometry "{header: {frame_id: map}, pose: {pose: {orientation: {w: 1.0}}}}" -r 20
ros2 topic pub /odom nav_msgs/msg/Odometry "{twist: {twist: {linear: {x: 0.5}}}}" -r 20
ros2 topic echo /drive
```

Software stop: `ros2 topic pub /minddrive/estop std_msgs/msg/Bool "{data: true}" -1`.

On the real car use `vla_ws/scripts/run_minddrive_route.sh <N> [--int8] [--05b]`: it sets up
DDS on both sides, starts the car's bringup and localisation, and launches with
`start_estopped:=true`, the stuck recovery and the speed floor of `run_orion_route.sh`.

Every plan is logged as `[MINDDRIVE] frame N cmd ... | decision speed="..." path="..." | prep
.. ms, forward .. ms | period .. ms` followed by the speed trajectory and (every second point
of) the path in the ROS ego frame, real metres.

ROS-free unit tests: `pytest vla_ws/src/minddrive_f1tenth/test` with `minddrive_ros` and
`minddrive_f1tenth` importable (no torch, no weights).

## Known limits

* Three front slots from one camera and three black rear slots are **not** what the model
  was trained on: the detections and map it hallucinates beside and behind the car are
  unconstrained. A wide camera cropped into the three front sectors would be the next step.
* The 0.39 m (3.9 cm real) between MindDrive's lidar origin and its ego point is ignored, as
  in the reference node.
* The decision expert was trained with LEFT/RIGHT only at junctions; `command_source:=geometry`
  reports them on bends, an out-of-distribution input (same caveat as `minddrive_ros`'s
  `curve_command_deg`). `command_source:=static` is the alternative.
* Trajectory shifts quoted above were measured on CARLA frames with six real views.
