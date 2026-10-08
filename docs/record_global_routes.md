# Recording global routes for SimLingo

A global route is the list of map-frame waypoints from which the SimLingo bridge
(`vla_ws/src/simlingo_f1tenth` on the Orin) picks the model's two
**target points**. It is *not* what the car drives: SimLingo predicts its own
trajectory from the camera image, the speed and those two points, and
`trajectory_controller_node` (in the bridge package) follows *that* prediction
and publishes `/drive`. The route only says where to go.

```
route CSV ──► /global_path ──► RoutePlanner ──► two target points ──► SimLingo ──► /simlingo/plan ──► trajectory_controller ──► /drive
 (this doc)   (waypoint_      (discard < 0.75 m real,                           (20 wps, map frame)      (pure pursuit on
               visualiser)     take the next two)                                                          the prediction)
```

Consequences:

- Only `waypoint_visualiser_node` runs on the car. **Not** `pure_pursuit_node`
  (it would drive the raceline itself and fight over `/drive`) and **not**
  `safety_node` (same topic).
- The `velocity` column of the CSV is not used for driving. The speed comes from
  the model's speed waypoints, capped by `max_speed_mps` on the Orin.
- Waypoints are consumed like a CARLA route plan: sparse nodes, each entry within
  0.75 m real (7.5 model m at `world_scale` 10) of the car is discarded and the
  next two become the target points. Dense logs are thinned at record time.

## File format

`x,y,velocity` per line, map frame (the `track_20260930` map), real metres,
**base_link** position, **no header line and no comments** (the C++ reader in
`waypoint_visualiser_node` calls `stod` on every field and a header crashes it).
The same file is accepted by `route_csv:=` on the Orin.

## Recording on the car

Preconditions: bringup running, `slam_localization` running and converged (car
started in the start box, pose on `/pf/pose/odom` matches RViz), see
[`mapping_localization_runbook.md`](mapping_localization_runbook.md).

```bash
source /opt/ros/foxy/setup.bash && source ~/vla_ws/install/setup.bash
mkdir -p ~/vla_ws/routes
python3 ~/vla_ws/scripts/record_global_route.py --out ~/vla_ws/routes/<name>.csv
```

Drive the route with the joystick (L1 deadman). Every kept waypoint is printed.
`Ctrl-C` writes the file. Options:

| option | default | meaning |
|---|---|---|
| `--min-spacing` | `1.0` | metres between kept waypoints (real). 1 m = 10 model m, sensible for CARLA-like sparsity; 0.5 m is the densest worth using |
| `--pose-offset-x` | `-0.27` | laser -> base_link along the heading (`/pf/pose/odom` is the laser pose) |
| `--close-loop` | off | append the first waypoint at the end (closed circuits) |
| `--velocity` | `0.5` | velocity column when `/odom` is not available |
| `--pose-topic` / `--speed-topic` | `/pf/pose/odom` / `/odom` | |

Offline from a bag (no ROS graph, no car):

```bash
python3 scripts/record_global_route.py --bag data/bags/sl_lap_145833 --out routes/lap_145833.csv
```

## What to record

- **Where the car starts matters less than in CARLA**: the bridge starts route
  progress at the nearest waypoint. Still, start each route from the start box
  so the localiser is converged.
- **Drive in the correct lane.** SimLingo was trained on right-hand traffic.
  The recorded laps of 2026-09-30 ran 10-20 cm left of the divider
  (`data/README.md`); the model gets its target points from the route, so a
  route in the wrong lane asks it to drive on the wrong side. For lane-exact
  routes, generate the waypoints from the aligned Lanelet2 map
  (`src/particle_filter/maps/track_20260930_lanelet2.osm`, `local_x`/`local_y`
  tags are map metres) instead of driving them.
- **One route per manoeuvre**, point-to-point like a CARLA route: a lap, a
  lane change, a turn at the junction, a U-turn. Name them by what they are.
- **Loops**: record with `--close-loop` and launch the bridge with
  `route_loop:=true`. Point-to-point routes end with the target points behind
  the car; stop the car or send `/simlingo/estop` when the bridge logs
  `route finished`.
- Keep the speed low and steady; the recorder does not care, but a smooth
  localiser track gives a smooth route.

## Using a route

On the car, point `waypoint_visualiser_node` at it (`waypoints_path` in
`src/pure_pursuit/config/config.yaml`, then rebuild `pure_pursuit`, or override
on the command line) and run only that node:

```bash
ros2 run pure_pursuit waypoint_visualiser_node --ros-args -p waypoints_path:=/home/f1tenth/vla_ws/routes/<name>.csv
```

It publishes `/waypoints` (markers) and a latched `/global_path` (`nav_msgs/Path`,
reliable, transient local) that the Orin picks up whenever it starts. Or copy
the CSV to the Orin and pass `route_csv:=/path/<name>.csv` to the bridge
launch; a `/global_path` message replaces a CSV route if both are present.

Sanity check before driving: in RViz the `/waypoints` markers must lie on the
road of the map, and the bridge's log line `route set from ...` must report the
expected number of waypoints and length.
