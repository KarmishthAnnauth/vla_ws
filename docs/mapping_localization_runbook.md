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
3. Start SLAM with the stock config plus the `base_footprint` transform from Lab 4. The repo's
   `f1tenth_online_async.yaml` (`base_frame: laser`) does not work on this car: slam_toolbox sits
   at 100 % CPU, never registers the LiDAR and never publishes `/map` (checked 2026-09-30).
   ```
   ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link base_footprint
   ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false
   ```
   Start the static transform first. Expect `Registering sensor: [Custom Described Lidar]` in the
   log. A steady stream of "Message Filter dropping message ... reason 'Unknown'" is normal here.
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
   Also save the pose graph; `slam_localization` (Phase 3) localises on it, not on the image:
   ```
   ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
     "{filename: '$HOME/vla_ws/src/particle_filter/maps/<name>'}"
   ```
8. Verify the result.
   ```
   ls -l ~/vla_ws/src/particle_filter/maps/<name>.*
   cat ~/vla_ws/src/particle_filter/maps/<name>.yaml
   ```
   Expect `<name>.pgm`, `<name>.yaml`, `<name>.posegraph` and `<name>.data`, `resolution: 0.05`, and `image:` holding just the file
   name (fix it if it is an absolute path). Convert the `.pgm` to PNG and look at it: closed,
   single-line walls, no stray blobs on the track. Stray pixels can be cleaned in an image
   editor as long as the image size stays the same.
9. Stop `slam_toolbox` (Ctrl+C), the static transform and the bag recording. SLAM must not run
   alongside a localiser; both publish a `map` transform.

If the map is skewed or the loop did not close, restart from step 1 (bringup restart included)
and drive slower, or rerun SLAM on the bag.

## Phase 3: localisation

Default: `slam_localization` (slam_toolbox localisation mode on the saved pose graph). It publishes
the pose on `/pf/pose/odom` (`nav_msgs/Odometry`, laser pose in `map`), the same topic and format as
the particle filter, so consumers do not change. On 2026-09-30 the particle filter lost track on
the top straight in most runs, live and in replays, whatever the settings; slam_localization
tracked every forward lap. The recordings are in `data/` (see `data/README.md`).

1. Bringup running, SLAM stopped, car in the start box facing along the top straight, nobody
   next to it. Then:
   ```
   ros2 launch slam_localization localize_launch.py
   ```
   Defaults (see `src/slam_localization/launch/localize_launch.py`):
   - `map:=track_20260930`: loads `<map>.posegraph` from `particle_filter/maps`.
   - `start_pose:=auto`: runs `fit_start_pose`, which matches the current scan against the map
     around the start box and logs e.g. `laser at (0.45, 0.10, -0.5 deg), 84% of scan points on
     walls`. Below 70 % it warns; fix the car's position rather than continuing. slam_toolbox only
     corrects the start pose once the car moves, and cannot recover from more than ~0.5 m off.
     Pass `start_pose:=x,y,yaw` (base_link in `map`) to skip the fit.
   - `smoothing_time:=0.3`: `pose_relay` follows the odometry and blends each slam_toolbox
     correction in over 0.3 s. Without it the pose steps by up to ~40 cm every ~0.7 s; with it
     the largest step per 0.1 s was 6.5 cm. `0` publishes the raw pose.
   - `udp_only:=true`: these nodes use Fast DDS over UDP only (`config/fastdds_udp_only.xml`).
     Processes that die uncleanly leave shared-memory lock files, after which new processes
     randomly miss `/scan`, `/odom` or `/tf` (seen repeatedly on 2026-09-30). A consumer of
     `/pf/pose/odom` that receives nothing can use the same profile:
     `export FASTRTPS_DEFAULT_PROFILES_FILE=$(ros2 pkg prefix slam_localization)/share/slam_localization/config/fastdds_udp_only.xml`.
   - `config/localization.yaml`: stock slam_toolbox localisation parameters except
     `loop_search_maximum_distance: 1.0`. At 3.0 it snapped the pose 2.2 m onto the wrong long
     side of the middle box.
2. Checks:
   ```
   ros2 topic hz /pf/pose/odom             # 50 Hz, one message per /odom message
   ros2 run tf2_ros tf2_echo map odom      # slam_toolbox's correction
   ```
3. Drive forwards. Known weak spots: pulling away from the start and the top straight (few
   features along the direction of travel), the bottom-right corner. The pose recovered from
   both in every test.
4. If the car is moved by hand, or reversed a lot before the pose has settled, localisation is
   lost (hand-carrying does not show in the odometry). Put it back in the start box and restart
   the launch.

Fallback, the particle filter (`particle_filter`, the course's Lab 5 setup), for comparison:
set `map: '<name>'` in `src/particle_filter/config/localize.yaml`, rebuild `particle_filter`, then
with the `PYTHONPATH` export from Phase 0 run `ros2 launch particle_filter localize_launch.py` and
set the pose with "2D Pose Estimate" (or publish `/initialpose`; the filter tracks the laser, so
the start box is about `x: 0.27` plus the fitted offset).

## Phase 4: keep the result

1. Commit the map and config from the car.
   ```
   cd ~/vla_ws && git add src/particle_filter/maps/<name>.* src/particle_filter/config/localize.yaml
   # and point the default map in src/slam_localization/launch/localize_launch.py at <name>
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
| slam_localization pose wrong from the start | Car not in the start box or someone next to it: check the `fit_start_pose` line in the launch log; restart the launch |
| New ROS processes see topics but receive no messages; `RTPS_TRANSPORT_SHM ... open_and_lock_file failed`; slam_localization pose stuck at a wrong constant | Stale Fast DDS shared-memory files from killed processes. Reboot, or start the process with `FASTRTPS_DEFAULT_PROFILES_FILE` set to `slam_localization/config/fastdds_udp_only.xml` (the localisation launch does this by default). `fastdds shm clean` crashes on this Foxy install |
| `/odom` and `/sensors/core` stop while bringup still runs | The VESC's USB device re-enumerated (power blip; `ls -l /dev/ttyACM0` shows a new time). Restart bringup, then the localisation |
| `ros2 bag record` bag is missing `/tf_static`, `/odom` or some `/tf` frames | Foxy recorder discovery quirk. Check `ros2 bag info` right after starting, restart the recorder if counts stay at 0 |
