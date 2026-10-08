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
2. Localise: car in the start box, `ros2 launch slam_localization localize_launch.py`.

Full step-by-step procedure for the car (preflight checks, mapping lap, saving, localisation):
[`docs/mapping_localization_runbook.md`](docs/mapping_localization_runbook.md).

## Global path

`pure_pursuit` reads `waypoints_path` (CSV: `x,y,velocity`, map frame) from `src/pure_pursuit/config/config.yaml`
and `waypoint_visualiser_node` publishes them as a MarkerArray on `/waypoints` **and** as a latched
`nav_msgs/Path` on `/global_path`.

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

## Running ORION along a recorded route (from the Orin)

```
scripts/run_orion_route.sh 1          # route_1.csv, fast inference (~1.0 s per frame)
scripts/run_orion_route.sh 1 --int8   # the INT8-quantised LLM (~0.9 s per frame); --dry, --help as above
```
