# vla_ws

A ROS2 Foxy F1tenth side + ROS2 Humble for running VLA models on an F1Tenth vehicle. 

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

## System Architecture

![System architecture](<Screenshot 2026-10-08 at 10.54.53.png>)

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

Model weights: `MODELS_HOST` (default `/media/orin-vlm/Intenso/SimLingo/simlingo_hf`); the InternVL2 cache must be in `../simlingo/pretrained/InternVL2-1B/`.

```bash
export MODELS_HOST=/path/to/simlingo_hf
```

```bash
docker pull karmishthannauth/simlingo:humble-cyclonedd
../simlingo/start_orin_f1tenth.sh --setup   # creates the container, builds the workspace
```

```
scripts/run_simlingo_route.sh        # lists the routes on the car, asks for the number
scripts/run_simlingo_route.sh 1      # route_1.csv; --dry keeps the car e-stopped, --help for the rest
scripts/run_simlingo_route.sh 1 --think   # thinking mode: the model writes a commentary before its waypoints
```

## Running ORION along a recorded route (from the Orin)

Model weights (`Orion.pth` + `pretrain_qformer/`): `ORION_WEIGHTS_HOST` (default: first of `/media/orin-vlm/Intenso/Orin`, `/media/orin-vlm/KINGSTON/models/Orion`).

```bash
export ORION_WEIGHTS_HOST=/path/to/Orion
```

```bash
docker pull karmishthannauth/orion_env_ros:v01
scripts/start_orion_f1tenth.sh --setup     # creates the container, builds the workspace
```

```
scripts/run_orion_route.sh 1          # route_1.csv, fast inference (~1.0 s per frame)
scripts/run_orion_route.sh 1 --int8   # the INT8-quantised LLM (~0.9 s per frame); --dry, --help as above
```

### On another Jetson (e.g. a Thor)

Everything the ORION bridge needs from source is in this repo: `src/orion_f1tenth`,
`src/simlingo_f1tenth` and, in [`deps/`](deps/README.md), `orion_ros` (with the INT8
calibration stats), `simlingo_ros` and the message packages. Not in the repo:

- the image `karmishthannauth/orion_env_ros:v01`. It is built for the Orin (JetPack 6,
  CUDA 12.2, PyTorch and mmcv compiled for the Orin GPU) and likely needs rebuilding on a
  JetPack 7 base for the Thor. Check the GPU and the mmcv ops first:
  `docker run --rm --runtime nvidia --entrypoint bash karmishthannauth/orion_env_ros:v01 -c "python3 -c 'import torch; torch.zeros(1).cuda()' && cd /root/Orion && python3 -c 'import mmcv._ext'"`
- the weights (`Orion.pth` + `pretrain_qformer/`, ~52 GB): plug in the SSD (looked for under
  `/media/$USER/...`) or set `ORION_WEIGHTS_HOST`.

Clone into a directory of its own: the container mounts the directory **above** `vla_ws` at
`/benchmarking` (the torch inductor cache goes there).

```bash
mkdir -p ~/benchmarking && cd ~/benchmarking && git clone https://github.com/KarmishthAnnauth/vla_ws.git
ssh-keygen -t ed25519                         # if there is no key yet
ssh-copy-id f1tenth@10.183.247.250            # the car's current eduroam address (DHCP: may change)
cd vla_ws && scripts/start_orion_f1tenth.sh --check
CAR=f1tenth@<car-ip> scripts/run_orion_route.sh 1 --dry
```

The DDS config is generated per run with both addresses as unicast peers, so any network on
which the Jetson can `ping` the car works (eduroam does today, from the Orin).

## Running MindDrive along a recorded route (from the Orin)

Model weights (`MindDrive/minddrive_3b_rltrain.pth`, `MindDrive/minddrive_rltrain.pth` and the two `llava-qwen*`
LLM directories): on the SSD at `MINDDRIVE_SSD_MOUNT` (default `/media/orin-vlm/Intenso`), mounted read-only
at `/models` in the container. The code is the bind-mounted checkout `../MindDrive` (see
`../start_minddrive_f1tenth.sh --help` for why it is not on the SSD).

```bash
docker build --network host -t karmishthannauth/minddrive_env_ros:v01 ../minddrive_env/   # once
../start_minddrive_f1tenth.sh --setup       # creates the container, builds the workspace
```

```
scripts/run_minddrive_route.sh 1          # route_1.csv, 3B model, fast inference (~0.9 s per frame)
scripts/run_minddrive_route.sh 1 --int8   # the INT8-quantised LLM (~0.8 s per frame; 0.4 model m shift)
scripts/run_minddrive_route.sh 1 --05b    # the 0.5B model; --plan-source speed, --dry, --help as above
```

## Measuring how far the car strays from the route

Every `run_*_route.sh` run also records the car's driven path (`scripts/record_run_trajectory.py` in the
container: `/pf/pose/odom` moved to base_link, e-stop state, route changes) to
`log/<model>/route_<N>_..._traj.csv`. After Ctrl-C the summary scores it, appends a row per route to
`log/<model>/route_tracking.csv` and saves a `.png` of the route against the driven path. By hand:

```
scripts/eval_route_tracking.py log/orion/route_1_*_traj.csv
scripts/eval_route_tracking.py log/*/route_*_traj.csv --csv results.csv --plot
```

Only driving while released counts. The routes have no timing, so the metrics are the spatial forms of
ADE/FDE: cross-track error to the route polyline per metre driven (mean = path ADE, RMS, p95, max, signed
mean, + = left), FDE (last pose to the route's last waypoint), per-waypoint closest approach of the
driven path (minADE-style), and completion (progress along the route while within 0.5 m of it).
