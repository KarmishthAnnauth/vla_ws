# simlingo_f1tenth — SimLingo on a real F1TENTH car

Runs SimLingo on the Jetson AGX Orin against a camera attached to the Orin and the
F1TENTH stack (`vla_ws`) on the Jetson Nano, and drives the car through the stack's
`/drive` topic. The lab track is a 1:10 replica of the CARLA scene, so the node
rescales the world so the model sees the geometry it was trained on.

Reference implementation: `simlingo_ros` (CARLA-in-the-loop). Nothing in it was changed.

```
 Jetson Nano (vla_ws, ROS 2 Foxy)                Jetson AGX Orin (this package, ROS 2 Humble)
 ─────────────────────────────────                ────────────────────────────────────────────────
 vesc_to_odom      /odom (speed) ───────────────► simlingo_realworld_node ◄── camera_node (/dev/video0)
 particle_filter   /pf/pose/odom (map pose) ────►   │   image + pose + speed + route → model
 waypoint_visual.  /global_path (nav_msgs/Path) ─►   │   → /simlingo/plan  (Trajectory, map frame, real units)
                                                     ▼
                                                  trajectory_controller_node  (20 Hz)
 ackermann_mux ◄── /drive (AckermannDriveStamped) ───┘   pure pursuit on the plan, plan speed capped
   → ackermann_to_vesc → VESC
```

## Nodes

| node | in | out |
|---|---|---|
| `camera_node` | V4L2 / GStreamer / synthetic | `/camera/front/image/compressed` (JPEG), optional raw |
| `simlingo_realworld_node` | image, `/pf/pose/odom`, `/odom`, `/global_path` or `route_csv` | `/simlingo/plan` (Trajectory, map), `/simlingo/plan_path`, `/simlingo/language_output`, `/simlingo/markers` |
| `trajectory_controller_node` | `/simlingo/plan`, `/pf/pose/odom`, `/odom`, `/simlingo/estop` | `/drive` |

Files: `simlingo_model.py` (model loading / preprocessing / forward, no ROS),
`route_planner.py` (frames, target-point selection, CSV/Path routes).

## What SimLingo needs, and where it comes from

| model input | training source (CARLA) | here |
|---|---|---|
| front RGB, 1024×512, FOV 110°, bottom 30 % cropped | `rgb_0` at (−1.5, 0, 2.0) m | camera on the Orin; frame centre-cropped to 2:1, resized, bottom-cropped (`format_camera_frame`). Mount the camera forward-facing with ~110° horizontal FOV. |
| ego speed (m/s) | speedometer | `/odom` `twist.linear.x` from `vesc_to_odom` × `world_scale` |
| two target points, ego frame | leaderboard `RoutePlanner` on the sparse global plan | `RoutePlanner` (same discard rule, 7.5 / 50 model m) on `/global_path` or `route_csv`, with pose from the particle filter |
| camera K, extrinsics | fixed from config | fixed, as in the reference node |

Output: 20 route waypoints at 0.25 s and 10 speed waypoints in the model's ego
frame (x forward, **y right**). They are divided by `world_scale`, mirrored to the
ROS frame, anchored in the map at the pose the image was taken from and published
as a Trajectory; the scalar desired speed is derived exactly as `control_pid` does.

## World scale

`world_scale` (default 10): positions, route and speed are multiplied by it before
the model, waypoints and speed divided by it after. With 10, the target-point
window is 0.75–5 m on the floor, a 2 m/s lab speed is 20 m/s to the model, and a
predicted 8 m/s becomes 0.8 m/s commanded. Set it to 1 to feed the model raw
metres. `speed_world_scale` (default 0 = `world_scale`) scales the speed alone, in and
out: with 5, a predicted 2 m/s is commanded as 0.4 m/s and 0.4 m/s is fed back as 2 m/s,
while distances stay at 1:10. The car then covers the scene twice as fast as the model
assumes. The controller's `world_scale` must match when `lateral_controller:=simlingo_pid`.

## Control

The VESC stack takes a speed and a steering angle, not CARLA pedals, so SimLingo's
longitudinal PID is not used: the plan's speed is commanded directly (`speed_scale`,
clamped to `max_speed_mps`, zero below `brake_below_mps`). Steering is pure pursuit
with the real wheelbase (default) or SimLingo's lateral PID
(`lateral_controller:=simlingo_pid`): its [-1, 1] output is converted into the curvature it
produces on the CARLA ego (the agent's own bicycle model) and from there into this car's
steering angle for the same path at `world_scale`. The car's steering limit (0.34 rad) is
reached at about half of the PID's range. `steer_smoothing_sec` > 0 low-passes the steering
command: each new plan moves the path under the car and with it the steering angle (pure
pursuit, 2026-10-02: median 5-7 deg per plan switch). Meant for pure pursuit only: the PID
steers on a point ~0.25 m ahead, and with 0.2 s of smoothing the car swung from side to side.
`pid_gain` multiplies the PID's three gains (1.0 = as tuned upstream). With the upstream gains
the steering sat at full lock in 40-60 % of the samples on the car, because a 6 cm sideways
shift of the path, which every new plan can bring, already asks for full lock;
`vla_ws/scripts/run_simlingo_route.sh` passes 0.4: much lower (0.2-0.3) and the car runs wide
in the corners, because the PID then asks for too little steering on a curve.

The controller publishes every tick at `control_hz`. `ackermann_mux` drops the
navigation source 0.2 s after its last message, so silence equals zero. No plan,
stale plan (`max_plan_age_sec`), plan driven to its end ("OUTRUN"), stale pose or
`/simlingo/estop` = true all publish an explicit zero-speed command. The joystick
(`/teleop`, priority 100) overrides `/drive` whenever L1 is held.

Frames: `/pf/pose/odom` is the **laser** pose. `pose_offset_x` (−0.27 m, from the
`base_link → laser` static TF in `bringup_launch.py`) moves it to `base_link` in
both nodes.

## Route

Point-to-point by default (`route_loop:=false`), like a CARLA route. Two sources:

* `route_csv:=/path/to/route.csv` — `x,y[,v]` rows in the map frame, real metres
  (the same format `pure_pursuit` reads).
* `/global_path` (`nav_msgs/Path`, transient-local) — published by
  `waypoint_visualiser_node` on the Nano (added in `vla_ws`, see below). A Path
  received on the topic replaces the CSV route.

CARLA's global plan is sparse (route nodes tens of metres apart). If the new route
is a dense path, `route_min_spacing_m` (real metres) thins it so the target point
behaves like a plan node rather than hovering at the discard radius.

## Changes made to the F1TENTH stack (`vla_ws`)

One addition, nothing else touched:

* `pure_pursuit/src/waypoint_visualiser_node.cpp` also publishes the loaded CSV as
  a latched `nav_msgs/Path` on `/global_path` (parameter `global_path_topic`).
  `nav_msgs` added to its CMake/package deps. Rebuild `pure_pursuit` on the Nano.
  You do not need `pure_pursuit_node` itself running (it would fight for `/drive`);
  launch only `waypoint_visualiser_node`, or pass `route_csv` and skip it.

Everything else SimLingo needs the stack already publishes: `/odom` (speed),
`/pf/pose/odom` (map pose), `/scan` unused. Do **not** run `safety_node` together
with this controller: it publishes on `/drive` too (speed 1.0 when clear).

Before the network step (already listed in `vla_ws/README.md`): Nano has
`ROS_LOCALHOST_ONLY=1` and no exported `ROS_DOMAIN_ID`; both must match the Orin
container. Foxy ↔ Humble interoperate for these message types (AckermannDriveStamped,
Odometry, Path are identical, and neither distro uses type hashes) provided both
sides run the same RMW — use CycloneDDS on the Nano (`ros-foxy-rmw-cyclonedds-cpp`)
with a static peer list for the ZeroTier unicast overlay.

## Running

```bash
# Orin (container, see simlingo/start_orin_f1tenth.sh which does all of this):
ros2 launch simlingo_f1tenth simlingo_f1tenth.launch.py \
  checkpoint_path:=/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
  simlingo_path:=/benchmarking/simlingo \
  camera_device:=/dev/video0 max_speed_mps:=0.5

# no camera / no car yet: exercise the pipeline
ros2 launch simlingo_f1tenth simlingo_f1tenth.launch.py camera_source:=synthetic route_csv:=/path/route.csv
ros2 topic pub /pf/pose/odom nav_msgs/msg/Odometry "{header: {frame_id: map}, pose: {pose: {orientation: {w: 1.0}}}}" -r 20
ros2 topic pub /odom nav_msgs/msg/Odometry "{twist: {twist: {linear: {x: 0.5}}}}" -r 20
ros2 topic echo /drive
```

Software stop: `ros2 topic pub /simlingo/estop std_msgs/msg/Bool "{data: true}" -1`.

On the real car use `vla_ws/scripts/run_simlingo_route.sh <N>` instead of the launch by hand: it
sets up DDS on both sides, starts the car's bringup and localisation, and launches with

* `start_estopped:=true` — the controller holds zero speed until `/simlingo/estop` receives
  `false`, so the model can be watched on the live camera before the car is released;
* `creep_after_sec:=5` — stuck recovery after upstream's (`agent_simlingo.py::run_step`): after
  that long at standstill while tracking a plan, `creep_speed_mps` (1.0) is commanded until the
  car rolls at `creep_release_speed_mps` (0.3), then `creep_hold_speed_mps` (0.5), for
  `creep_duration_sec` (2.0) in total. From standstill the model keeps predicting standstill, and
  a short weak kick (0.5 m/s for 0.75 s) left the car at 0.1 m/s; 0 = off (the launch default).
  The kick is not limited by `max_speed_mps`, the hold is. With these defaults the car stalled
  again on 2026-10-02 (kick over after 0.4 s, then 0.5 m/s); `vla_ws/scripts/run_simlingo_route.sh`
  passes 1.5 / 0.7 / 0.7.
* `min_speed_mps` — floor on the commanded speed while the plan wants to drive. Commands below
  about 0.4 m/s do not keep the drivetrain turning.

The controller sends zero-speed commands when it is shut down (`ackermann_mux` only stops
forwarding when `/drive` goes quiet). `log_waypoints` (default true) prints the 20 predicted
waypoints of every plan in the ROS ego frame, real metres.
