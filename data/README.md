# Track map and recorded drives (2026-09-30)

Recorded on the car (Jetson, ROS 2 Foxy) while building the map of the track and choosing a
localiser. Everything here is for offline work; nothing in `data/` is used at runtime.

## Map: `track_20260930`

Files live in `src/particle_filter/maps/` (installed with `particle_filter`):

| File | What it is |
|---|---|
| `track_20260930.yaml` | map metadata: `resolution: 0.05` m/px, `origin: [-3.04, -8.78, 0]` (pose of the bottom-left pixel), trinary thresholds |
| `track_20260930.pgm` | occupancy image, 298 x 200 px (about 15 x 10 m): black = wall, white = free, grey = unknown |
| `track_20260930.png` | preview of the same image |
| `track_20260930.posegraph`, `.data` | slam_toolbox pose graph; required by `slam_localization` |

- Built with slam_toolbox online async mapping (stock `mapper_params_online_async.yaml`, `base_frame:
  base_footprint`, plus a static `base_link -> base_footprint` identity transform), two laps.
- The map origin (0, 0, 0) is the `base_link` pose when bringup was started for mapping, with the
  car in the start box facing along the top straight (+x).
- Pixel `(col, row)` to map: `x = -3.04 + (col + 0.5) * 0.05`, `y = -8.78 + (200 - row - 0.5) * 0.05`.
- The track edges are sparse markers (dotted lines in the map); the room walls, the two boxes
  inside the track and the clutter below the track carry most of the localisation information.

## Frames and topics in the bags

- `odom -> base_link`: wheel odometry (`vesc_to_odom_node`). Reads about 10 % short on distance
  (29.8 m vs 33.8 m SLAM on `lap_134654`); `speed_to_erpm_gain` 7914 is not calibrated.
- `base_link -> laser`: static, (0.27, 0, 0.11) m. Missing from most bags (Foxy's recorder did
  not pick up `/tf_static`); add it yourself when replaying.
- `/scan`: Hokuyo, 40 Hz, 1081 beams, 270 deg.
- `/pf/pose/odom` (`nav_msgs/Odometry`): the localiser output, pose of the **laser** frame in
  `map`. From `particle_filter` in `lap_134654`, from `slam_localization` in all other bags.
- `map -> odom` in `/tf`: published by slam_toolbox localisation (`sl_*` bags). In `lap_134654`
  the particle filter published `map -> laser` instead.

No drive commands were recorded, so replaying a bag cannot move the car.

## Bags (`data/bags/`, rosbag2 sqlite3)

| Bag | Length | Localiser | What happened | Gaps |
|---|---|---|---|---|
| `lap_134654` | 96 s | particle_filter, `motion_dispersion_theta: 0.05` | parked 0-33 s, one forward lap; the particle filter lost track on the top straight at ~44 s | no `/tf_static`; `/sensors/core` empty |
| `sl_lap_143220` | 75 s | slam_localization, `loop_search_maximum_distance: 3.0`, raw pose | one lap cutting back through the middle; at ~48 s loop closure snapped the pose 2.2 m to the wrong side of the middle box | - |
| `sl_lap_145448` | 30 s | slam_localization, 1.0 m, raw pose | car was driven before recording started; localisation already wrong (car stationary). Not a useful drive | no `/tf_static` |
| `sl_lap_145833` | 77 s | slam_localization, 1.0 m, raw pose | clean forward lap (~28 m, up to 1.6 m/s); ended within ~8 cm of an independent scan fit | - |
| `sl_man_150210` | 80 s | slam_localization, 1.0 m, raw pose | forward manoeuvres (~24 m, up to 1.5 m/s, turns up to 1.8 rad/s), started with a ~5 deg heading error left from the previous run | no `/tf_static` |
| `sl_smooth_151045` | 76 s | slam_localization, 1.0 m, **smoothed** (`smoothing_time: 0.3`) | forward lap (~28 m, up to 1.2 m/s), final pose 96 % on walls | `/odom` empty and no `odom -> base_link` in `/tf`: use `logs/sl_smooth_151045.csv` for odometry |

Replay: `ros2 bag play data/bags/<name>`. Prefer a separate `ROS_DOMAIN_ID` so it cannot mix with
a live car, and note that `/tf` in the bags already contains the recorded localiser's transforms
(`map -> odom` or `map -> laser`), which conflict with a localiser you run on top.

## Per-frame logs (`data/logs/`)

One CSV per drive, written at 10 Hz by the video recorder used during the tests (same names as
the bags; `pf_lap_134654.csv` belongs to `lap_134654`). Columns:

| Column | Meaning |
|---|---|
| `t` | seconds since the recorder started (starts a fraction of a second before the bag) |
| `x, y, yaw` | latest `/pf/pose/odom` (laser pose in `map`, yaw in rad) |
| `hit_frac` | share of scan points (range < 10 m) within 15 cm of a wall in the map, with the scan placed at that pose |
| `median_err_m` | median distance of those scan points to the nearest wall (5 cm grid) |
| `pf_hz` | `/pf/pose/odom` messages in the last second |
| `odom_x, odom_y, odom_yaw, odom_v, odom_wz` | latest `/odom` pose (base_link in `odom`), speed and yaw rate |

`hit_frac` is a rough health signal, not ground truth: people or objects that are not in the map
lower it even when the pose is right. For ground truth, running slam_toolbox mapping on the bag's
`/scan` + odometry worked well (walls lined up with the saved map).
