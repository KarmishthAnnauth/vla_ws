# vla_ws

Minimal ROS 2 Foxy workspace for driving the F1TENTH car from externally
generated `AckermannDriveStamped` commands (arriving over the ZeroTier network).
Copied from `~/f1tenth_ws` on 2026-09-28; the original workspace is untouched.

## Packages

| Package | Role |
|---|---|
| `f1tenth_system/f1tenth_stack` | Bringup launch + configs (VESC gains, Hokuyo LiDAR, mux, joystick) |
| `f1tenth_system/vesc/*` | `vesc_driver` (serial to VESC), `vesc_ackermann` (Ackermann -> ERPM/servo, VESC state -> `/odom`) |
| `f1tenth_system/ackermann_mux` | Priority mux: `/teleop` (joystick, prio 100) beats `/drive` (autonomy, prio 10) |
| `f1tenth_system/teleop_tools` | `joy_teleop` (deadman on L1 for manual driving) |
| `particle_filter` | MCL localisation on a saved map (range_libc GPU). Publishes `/pf/pose/odom`, `map -> laser` TF. Holds the maps (`maps/`). Fallback only: it loses track on the current track |
| `slam_localization` | Default localisation: slam_toolbox localisation on the saved pose graph, auto start pose, smoothed pose republished on `/pf/pose/odom` (same format as the particle filter) |
| `slam_toolbox` | Source copy, **COLCON_IGNORE'd**: the system package (`/opt/ros/foxy`, same version 2.4.1) is used. Delete `src/slam_toolbox/COLCON_IGNORE` to build from source instead. |
| `pure_pursuit` | Reference-path consumer: loads a CSV raceline (x, y, v in `map` frame) and publishes `/waypoints` markers. Proves the car can hold a global path. |
| `safety_node` | Optional automatic emergency brake on `/scan` + `/odom`, publishes stop on `/drive` |

## Command path (external source -> wheels)

```
external VLA  --AckermannDriveStamped-->  /drive
                                            |  ackermann_mux (nav prio 10, timeout 0.2 s)
                                            v
                                     /ackermann_drive
                                            |  ackermann_to_vesc_node (vesc.yaml gains)
                                            v
                 /commands/motor/speed  +  /commands/servo/position
                                            |  vesc_driver_node (/dev/sensors/vesc)
                                            v
                                          VESC
```

## Localisation

1. Map once: slam_toolbox mapping with the stock config plus a static `base_link -> base_footprint`
   transform, then save the map image and the pose graph into `src/particle_filter/maps/`.
   Current map: `track_20260930` (description, recorded drives and logs in [`data/README.md`](data/README.md)).
   The Lanelet2 road map of the track (`testtrack_base.osm`) is aligned to it:
   `src/particle_filter/maps/track_20260930_lanelet2.osm` (map-frame metres in `local_x`/`local_y`),
   produced by `scripts/align_osm_to_map.py`. `track_20260930_clean` is a tidied copy of the map
   (track area only, straight walls, no stray points; `scripts/clean_map.py`) for the particle
   filter to try; the original stays the default.
2. Localise: car in the start box, `ros2 launch slam_localization localize_launch.py`.

Full step-by-step procedure for the car (preflight checks, mapping lap, saving, localisation):
[`docs/mapping_localization_runbook.md`](docs/mapping_localization_runbook.md).

## Global path

`pure_pursuit` reads `waypoints_path` (CSV: `x,y,velocity`, map frame) from `src/pure_pursuit/config/config.yaml`
and `waypoint_visualiser_node` publishes them as a MarkerArray on `/waypoints` **and** as a latched
`nav_msgs/Path` on `/global_path` (2026-09-28) for the external planner (SimLingo on the Orin,
`vla_ws/src/simlingo_f1tenth`). Only `waypoint_visualiser_node` is needed for that;
`pure_pursuit_node` must NOT run alongside the external planner (both publish `/drive`).
The bundled racelines are from an older track; new ones for the 1:10 cs3 replica are still to be recorded.
How to record them, the CSV format and why the route is only the target-point source (the car drives the model's prediction): [`docs/record_global_routes.md`](docs/record_global_routes.md).

## Launch order

```
source /opt/ros/foxy/setup.bash && source ~/vla_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py        # VESC, LiDAR, mux, joystick
ros2 launch slam_localization localize_launch.py   # car in the start box; pose on /pf/pose/odom
ros2 launch pure_pursuit pure_pursuit_launch.py    # optional: waypoints + rviz
```

## Running SimLingo along a recorded route (from the Orin)

```
scripts/run_simlingo_route.sh        # lists the routes on the car, asks for the number
scripts/run_simlingo_route.sh 1      # route_1.csv; --dry keeps the car e-stopped, --help for the rest
scripts/run_simlingo_route.sh 1 --think   # thinking mode: the model writes a commentary before its waypoints
```

Car in the start box. The script copies `routes/route_*.csv` from the car, starts bringup and a fresh
localisation there (`scripts/simlingo_car.sh` over ssh, logs in `~/vla_ws/log/simlingo/`), launches
camera + SimLingo + controller in the Orin container and prints every plan with its 20 waypoints and
the model time. The controller starts e-stopped; Enter releases the car, Enter again toggles the
e-stop, Ctrl-C stops the Orin side. The run log and a timing summary land in `log/simlingo/`.

Things that are not obvious (found 2026-10-01):

- The car is not on ZeroTier; the link is the Wi-Fi, where the car and the Orin are in different
  subnets, so DDS discovery is unicast: CycloneDDS on the Orin with the car as peer, Fast DDS on the
  car (no CycloneDDS there, no sudo) with the Orin as initial peer. The car's profile must also
  restrict Fast DDS to the Wi-Fi address and loopback, otherwise the Orin is handed the LiDAR-network
  address (192.168.0.15) and nothing matches. Both configs are generated per run; `~/.bashrc` on
  the car (`ROS_LOCALHOST_ONLY=1`) is left alone, `source scripts/simlingo_car.sh env` gives a shell
  on the car the same settings.
- `/odom` stays silent after bringup until the VESC has received one command (`vesc_to_odom` waits
  for a servo command). `simlingo_car.sh` sends one zero-speed `/drive` command for that.
- From standstill SimLingo predicts standstill (desired speed ~0.1 model m/s at the start box);
  once rolling it asks for ~2 model m/s. The controller therefore kick-starts the car
  (`--creep-after`: `--creep-speed` 1.5 m/s, not limited by the cap, until it rolls at
  `--creep-hold` 0.7 m/s, then that speed, 2 s in total). Weaker kicks failed: 0.5 m/s for 0.75 s
  left it at 0.1 m/s, and 1.0 m/s released at 0.3 m/s (after 0.4 s) and held at 0.5 m/s let it
  stall again. The upstream agent has the same recovery after 40 s.
- 2 model m/s is 0.2 m/s at the track's 1:10 scale, too slow to move the car. The script therefore
  scales the speed alone by less than 10 (`--speed-factor`, in and out of the model; distances
  stay 1:10), so the car covers the scene faster than the model assumes. At 5 the car drove a
  median of ~0.9 m/s (the model asks for a median of ~4.5 model m/s while rolling). The default
  is 7, chosen to match two manual reference drives (2026-10-02, `/odom` at 50 Hz, launches cut:
  median 0.65 m/s over 14 s and 0.83 m/s over 17 s, the latter nearly constant). At 7 the plans
  ask for a median of 0.71 m/s but vary far more than the manual drives (10-90 %: 0.37-1.17 m/s),
  so no factor reproduces them alone: the cap (`--speed`, 0.85) is the steady manual pace, and
  commands under ~0.5 m/s (`--min-speed`) do not keep the drivetrain turning.

## Running ORION along a recorded route (from the Orin)

```
scripts/run_orion_route.sh 1          # route_1.csv, fast inference (~1.0 s per frame)
scripts/run_orion_route.sh 1 --int8   # the INT8-quantised LLM (~0.9 s per frame); --dry, --help as above
```

Same car side and e-stop flow as the SimLingo script (`scripts/simlingo_car.sh`, `scripts/simlingo_estop.py`
on `/orion/estop`), a different Orin container (`benchmarking/start_orion_f1tenth.sh`, image
`orion_env_ros`, weights on the KINGSTON SSD, which must be mounted by hand after a reboot) and
`vla_ws/src/orion_f1tenth` instead of `simlingo_f1tenth`. ORION wants six cameras: the one
camera fills the front and both front-side slots, the rear slots are black (`--side-views`,
`--hfov` to crop a wide camera to the model's 70 deg). The model load takes minutes (weights plus the
`torch.compile` warm-up of the `fast`/`int8` variants the first time). Logs land in `log/orion/`.
Differences from SimLingo: the speed factor defaults to the track scale (10) because ORION keeps a
temporal memory of ego poses, and the driving command fed to the model is LANEFOLLOW unless
`--command geometry` derives LEFT/RIGHT from the route.

## Known issues

- Bringup expects the VESC at `/dev/sensors/vesc` (udev rule present) and the Hokuyo at 192.168.0.10 on eth0.
- With the joystick connected, L1 must NOT be held for `/drive` to reach the VESC (teleop has priority).
