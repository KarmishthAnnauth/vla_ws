# deps

ROS packages the ORION bridge (`src/orion_f1tenth`) needs inside its container, vendored
from `alpamayo-autoware` on 2026-10-12 so that a clone of this repo is enough to build it:

| Package | From | Needed by |
|---|---|---|
| `orion_ros` | `alpamayo-autoware/src/orion_ros` (working tree; `engines/` holds only `llm_act_stats.pt`, the INT8 SmoothQuant stats) | `orion_f1tenth` (`orion_speedups`), `--int8` |
| `simlingo_ros` | `alpamayo-autoware/src/simlingo_ros` | `simlingo_f1tenth` (`simlingo_pid`) |
| `autoware_msgs` | `alpamayo-autoware/src/autoware_msgs` | `autoware_planning_msgs/Trajectory` |
| `ackermann_msgs` | `alpamayo-autoware/ackermann_msgs` | controller output |
| `carla_msgs` | `alpamayo-autoware/src/ros-bridge/ros-carla-msgs` | `orion_ros`, `simlingo_ros` (package dependency) |

`COLCON_IGNORE` keeps a plain `colcon build` (the car's Foxy workspace) out of this
directory. The container build names each package directory as a base path, which bypasses it:

```bash
colcon build --base-paths deps/autoware_msgs deps/ackermann_msgs deps/carla_msgs deps/simlingo_ros \
  deps/orion_ros src/simlingo_f1tenth src/orion_f1tenth --packages-up-to orion_f1tenth
```

These are copies: edits made in `alpamayo-autoware` do not reach them, and vice versa.
