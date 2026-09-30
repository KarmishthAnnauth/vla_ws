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
| `particle_filter` | MCL localisation on a saved map (range_libc GPU). Publishes `/pf/pose/odom`, `map -> laser` TF |
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

1. Map once: `ros2 launch slam_toolbox online_async_launch.py params_file:=<f1tenth_stack>/config/f1tenth_online_async.yaml`, then save the map into `src/particle_filter/maps/`.
2. Localise: set `map` in `src/particle_filter/config/localize.yaml`, then `ros2 launch particle_filter localize_launch.py`.

Full step-by-step procedure for the car (preflight checks, mapping lap, saving, particle filter test):
[`docs/mapping_localization_runbook.md`](docs/mapping_localization_runbook.md).

## Global path

`pure_pursuit` reads `waypoints_path` (CSV: `x,y,velocity`, map frame) from `src/pure_pursuit/config/config.yaml`
and `waypoint_visualiser_node` publishes them as a MarkerArray on `/waypoints` **and** as a latched
`nav_msgs/Path` on `/global_path` (2026-09-28) for the external planner (SimLingo on the Orin,
`alpamayo-autoware/src/simlingo_f1tenth`). Only `waypoint_visualiser_node` is needed for that;
`pure_pursuit_node` must NOT run alongside the external planner (both publish `/drive`).
The bundled racelines are from an older track; new ones for the 1:10 cs3 replica are still to be recorded.

## Launch order

```
source /opt/ros/foxy/setup.bash && source ~/vla_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py        # VESC, LiDAR, mux, joystick
ros2 launch particle_filter localize_launch.py     # map server + particle filter
ros2 launch pure_pursuit pure_pursuit_launch.py    # optional: waypoints + rviz
```

## Known issues to fix before remote driving

- `~/.bashrc` sets `export ROS_LOCALHOST_ONLY=1`: DDS traffic never leaves the machine, so commands over ZeroTier will not arrive. Set it to 0 (and `export ROS_DOMAIN_ID=5`, currently not exported).
- ZeroTier is a unicast overlay; if DDS multicast discovery fails, configure a static peer list (CycloneDDS `Peers` or FastDDS initial peers) on both ends.
- Bringup expects the VESC at `/dev/sensors/vesc` (udev rule present) and the Hokuyo at 192.168.0.10 on eth0.
- With the joystick connected, L1 must NOT be held for `/drive` to reach the VESC (teleop has priority).
