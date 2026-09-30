# Mapping and localisation runbook (runs on the car)

Goal: record an occupancy map of the physical track with `slam_toolbox`, save it into
`src/particle_filter/maps/`, and confirm the particle filter localises on it. Everything here
runs on the car's Jetson (ROS 2 Foxy, user `f1tenth`, workspace `~/vla_ws`). The Orin is not
involved.

This was written on 2026-09-30 from the Orin, using this repo and the TUM F1TENTH course labs
4 (Mapping) and 5 (Localization). The car itself was not inspected, so every "expected" below
is an assumption to check, not a verified fact. The Phase 0 checks exist for that reason.

## Ground rules

- Never modify or rebuild `~/f1tenth_ws`. All work happens in `~/vla_ws`.
- Build only from the workspace root (`~/vla_ws`), never from inside `src/`.
- A human drives the car with the joystick. Nothing may publish on `/drive` during mapping.
- Put the car on its stand before running anything that could turn the wheels unattended
  (for example replaying a bag recorded with `-a`).
- `sudo` commands need the user at the keyboard.
- RViz needs the car's desktop. The user reaches it over VNC: in an SSH session run
  `x11vnc -usepw -display :0 -loop` and keep it open, then connect with RealVNC. Only one VNC
  client at a time. Start RViz from a terminal inside that desktop.
- If a check fails, stop and report what you saw instead of changing configs to make it pass.
  The VESC gains in `f1tenth_stack/config/vesc.yaml` in particular may have been tuned on the car.

## Who does what

| Human | Claude session on the car |
|---|---|
| Battery, power switch, joystick pairing | Preflight checks, builds |
| `sudo` commands | Launching and monitoring nodes, reading logs |
| Placing the car, driving the lap | Checking topics and TF while the human drives |
| Clicking in RViz (Save Map, 2D Pose Estimate) | Editing `localize.yaml`, inspecting the saved map, git |

## Facts from this repo

- Frames: `odom -> base_link` from `vesc_to_odom_node` (`publish_tf: true`), `base_link -> laser`
  static (0.27, 0, 0.11) from bringup. SLAM config uses `base_frame: laser`, `odom_frame: odom`,
  `scan_topic: /scan`, so the `base_footprint` static transform from Lab 4 is not needed.
- SLAM config: `src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml`
  (resolution 0.05 m, loop closing on).
- Joystick (`joy_teleop.yaml`): hold L1 (button 4) as deadman, axis 1 is speed (full stick is
  5 m/s), axis 2 is steering.
- Mux: `/teleop` (joystick, priority 100) beats `/drive` (priority 10). `/drive` reaches the VESC
  whenever L1 is not held.
- Particle filter: reads `/scan` and `/odom`, gets the map from `/map_server/map`, publishes
  `/pf/pose/odom`, `/pf/viz/inferred_pose`, `/pf/viz/particles` and the `map -> laser` transform.
  The map name comes from `map_server.ros__parameters.map` in
  `src/particle_filter/config/localize.yaml` (file name without extension).
- Maps and configs are copied into `install/` at build time, so a new map or a config edit needs
  `colcon build --packages-select particle_filter` before it takes effect.
- `src/slam_toolbox` is a source copy with `COLCON_IGNORE`. The repo assumes the apt package
  `ros-foxy-slam-toolbox` is installed.

## Phase 0: preflight

Run in order. Each line says what to expect.

1. Sync the repo.
   ```
   cd ~/vla_ws && git status && git remote -v
   ```
   - If it is a clone of `github.com/KarmishthAnnauth/vla_ws` with a clean tree: `git pull`.
   - If it has local changes, or is not a git repo at all (it may be a plain copy of
     `~/f1tenth_ws`): stop and ask the user how to reconcile. Do not reset, overwrite or
     re-clone over it.
2. ROS environment.
   ```
   source /opt/ros/foxy/setup.bash && printenv ROS_DISTRO
   ```
   Expect `foxy`.
3. Required packages.
   ```
   ros2 pkg prefix slam_toolbox
   ros2 pkg prefix nav2_map_server
   ros2 pkg prefix nav2_lifecycle_manager
   ```
   Expect `/opt/ros/foxy` for all three. Lab 5 has students remove the apt `slam_toolbox` and
   build it in `~/lab_ws25`, so it may be missing. If it is, either the user runs
   `sudo apt-get install ros-foxy-slam-toolbox`, or delete `src/slam_toolbox/COLCON_IGNORE` and
   build it from source (slow on the Jetson). Ask the user which.
4. range_libc for the particle filter.
   ```
   export PYTHONPATH=${PYTHONPATH}:/usr/lib/python3.8/site-packages/range_libc-0.1-py3.8-linux-aarch64.egg
   python3 -c "import range_libc; print(range_libc.__file__)"
   ```
   Expect a path, not `ModuleNotFoundError`. This export is needed in every terminal that launches
   the particle filter, unless `~/.bashrc` already does it.
5. Build and source.
   ```
   cd ~/vla_ws && colcon build && source install/setup.bash
   ```
   The `pure_pursuit` change in commit `319b9d1` (`/global_path` publisher) has not been built on
   Foxy yet. If it fails, report the error; mapping does not depend on it, so
   `colcon build --packages-skip pure_pursuit` is an acceptable way to continue.
6. LiDAR network (user runs the `sudo` part).
   ```
   sudo nmcli connection up Hokuyo
   ping -c 2 192.168.0.10
   ```
   Expect replies.
7. VESC device.
   ```
   ls -l /dev/sensors/vesc
   ```
   Expect a symlink to a tty device.
8. Joystick: the controller's LED is solid after pressing the PS button.

## Phase 1: track check (human)

- Walls are continuous, opaque and taller than the LiDAR scan plane. Gaps, glass and glossy black
  surfaces leave holes in the map.
- Nothing moves during the mapping lap, and the track stays as it is afterwards. If the track is
  rearranged, the map has to be redone.
- The car sits in the start box, pointing in the driving direction. This pose becomes the map
  origin (0, 0, heading 0), and the racelines will later be recorded in this frame.

## Phase 2: mapping

1. With the car in the start box, (re)start bringup. Restarting matters: it zeroes the odometry
   that the map origin comes from.
   ```
   ros2 launch f1tenth_stack bringup_launch.py
   ```
2. Check the inputs in a second sourced terminal.
   ```
   ros2 topic hz /scan                     # steady rate
   ros2 topic echo /odom --once            # pose near zero
   ros2 run tf2_ros tf2_echo odom laser    # resolves, about 0.27 m forward
   ros2 topic info /drive                  # publisher count 0
   ```
3. Start SLAM.
   ```
   ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false \
     params_file:=$(ros2 pkg prefix f1tenth_stack)/share/f1tenth_stack/config/f1tenth_online_async.yaml
   ```
   The launch file defaults `use_sim_time` to `true`; the course runs it that way on the real
   car, so the override is tidy rather than required. Check the log does not say the params file
   "does not contain slam_toolbox parameters" (that means it fell back to the default config with
   `base_footprint`).
4. Optional but recommended: record a bag so SLAM can be rerun without redriving.
   ```
   mkdir -p ~/bags && cd ~/bags && ros2 bag record -o track_$(date +%Y%m%d_%H%M) /scan /odom /tf /tf_static
   ```
   Only these topics, not `-a`, so a replay cannot drive the car. Delete old bags afterwards;
   storage on the Jetson is limited.
5. RViz, started inside the VNC desktop from the maps folder so Save Map writes there:
   ```
   source /opt/ros/foxy/setup.bash && source ~/vla_ws/install/setup.bash
   cd ~/vla_ws/src/particle_filter/maps && rviz2
   ```
   Fixed frame `map`, add a Map display on `/map`, then Panels > Add New Panel >
   `SlamToolboxPlugin`.
6. The human drives with L1 held: slowly (about 1 m/s or less, a small stick deflection),
   smooth steering, one full lap plus some overlap past the start so the loop closes. In RViz
   the walls should be single lines with no doubled or sheared sections.
7. Save while SLAM is still running. Pick a name without spaces, e.g. `track_20261001`.
   - RViz: type the name next to Save Map in the plugin panel and click it, or
   - CLI:
     ```
     ros2 run nav2_map_server map_saver_cli -f ~/vla_ws/src/particle_filter/maps/<name> \
       --ros-args -p map_subscribe_transient_local:=true
     ```
8. Verify the result.
   ```
   ls -l ~/vla_ws/src/particle_filter/maps/<name>.*
   cat ~/vla_ws/src/particle_filter/maps/<name>.yaml
   ```
   Expect `<name>.pgm` and `<name>.yaml`, `resolution: 0.05`, and `image:` holding just the file
   name (fix it if it is an absolute path). Convert the `.pgm` to PNG and look at it: closed,
   single-line walls, no stray blobs on the track. Stray pixels can be cleaned in an image
   editor as long as the image size stays the same.
9. Stop `slam_toolbox` (Ctrl+C) and the bag recording. SLAM must not run alongside the particle
   filter; both publish a `map` transform.

If the map is skewed or the loop did not close, restart from step 1 (bringup restart included)
and drive slower, or rerun SLAM on the bag.

## Phase 3: localisation

1. Point the particle filter at the new map: in `src/particle_filter/config/localize.yaml` set
   ```
   map_server:
     ros__parameters:
       map: '<name>'
   ```
2. Rebuild and source.
   ```
   cd ~/vla_ws && colcon build --packages-select particle_filter && source install/setup.bash
   ```
3. With bringup running, SLAM stopped and the `PYTHONPATH` export from Phase 0 in place:
   ```
   ros2 launch particle_filter localize_launch.py
   ```
4. RViz in the VNC desktop: `rviz2 -d ~/vla_ws/src/particle_filter/rviz/pf.rviz`, fixed frame
   `map`, displays for `/map`, `/scan`, `/pf/viz/particles` and `/pf/viz/inferred_pose`.
   If the map does not appear, leave RViz open, Ctrl+C the particle filter and launch it again.
5. The human sets the initial pose with "2D Pose Estimate" at the car's real position and heading.
6. Checks while the human drives a slow lap:
   ```
   ros2 topic hz /pf/pose/odom
   ros2 run tf2_ros tf2_echo map laser
   ```
   The laser scan drawn in RViz should stay on the map's walls for the whole lap, and the pose
   should return to the start box when the car does.
7. `max_particles` is 4000, sized for large spaces. If `/pf/pose/odom` is slow, try 400 (edit
   `localize.yaml`, rebuild, relaunch) and compare.
8. If the particle filter dies with a CUDA or `rmgpu` error, report it to the user. CPU ray
   casting methods are described in `src/particle_filter/docs/RangeLibcUsageandInformation.pdf`.

## Phase 4: keep the result

1. Commit the map and config from the car.
   ```
   cd ~/vla_ws && git add src/particle_filter/maps/<name>.* src/particle_filter/config/localize.yaml
   git commit -m "Add <name> track map and use it for localisation" && git push
   ```
   If the car has no GitHub credentials, tell the user; the files can be copied off with `scp`.
2. Treat the map as frozen from here on. The racelines for this track
   (`src/pure_pursuit/racelines/*.csv`, `waypoints_path` in `src/pure_pursuit/config/config.yaml`)
   will be recorded in this map's frame, so any remap means re-recording them. The bundled
   racelines belong to older tracks and do not match the new map.

## Shutting down (human)

Ctrl+C all launches, `sudo shutdown now`, wait for the Jetson's green light to go out, flip the
powerboard switch, then unplug the battery by the plugs, not the cables.

## Troubleshooting

| Symptom | Check |
|---|---|
| No `/scan` | `sudo nmcli connection up Hokuyo`, `ping 192.168.0.10` |
| Wheels do not respond to the joystick | L1 held? `ros2 topic echo /teleop`; controller paired? |
| SLAM starts but no map, or "message filter dropping" | `ros2 run tf2_tools view_frames.py`; `odom -> base_link -> laser` must exist; confirm the `params_file` was accepted |
| SlamToolboxPlugin not in RViz's panel list | RViz was started without the ROS environment (or the workspace that holds `slam_toolbox`) sourced |
| `map_saver_cli` times out | Keep `-p map_subscribe_transient_local:=true`; SLAM must still be running |
| Particle filter: `No module named range_libc` | The `PYTHONPATH` export from Phase 0 |
| Map server cannot find `<name>.yaml` | `particle_filter` was not rebuilt after adding the map, or the name in `localize.yaml` has an extension |
| Map not visible in RViz | Relaunch the particle filter with RViz already open |
| Pose jumps or drifts off the walls | Set the initial pose again; check the track has not changed since mapping |
