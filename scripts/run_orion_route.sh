#!/usr/bin/env bash
#
# run_orion_route.sh — run ORION on the real F1TENTH along one recorded global route.
#
# The ORION counterpart of run_simlingo_route.sh: same car side (scripts/simlingo_car.sh),
# same routes, same e-stop flow; a different container (benchmarking/start_orion_f1tenth.sh)
# and launch (vla_ws/src/orion_f1tenth).
#
# Start it on the Orin, car in the start box:
#
#   scripts/run_orion_route.sh            list the routes and ask which one
#   scripts/run_orion_route.sh 1          route_1.csv, fast inference (~1.0 s per frame)
#   scripts/run_orion_route.sh 1 --int8   the INT8-quantised LLM (~0.9 s per frame)
#   scripts/run_orion_route.sh 1 --dry    inference only: the car stays e-stopped
#   scripts/run_orion_route.sh 1 --no-save-frames   do not save the per-plan JPEGs (saved by default)
#   scripts/run_orion_route.sh 1 -- camera_hfov_deg:=110.0   (extra launch arguments after --)
#   scripts/run_orion_route.sh --list | --car-status | --stop-car
#
# What it does:
#   1. copies the route CSVs from the car (~/vla_ws/routes) to vla_ws/routes here;
#   2. prepares the Orin container (benchmarking/start_orion_f1tenth.sh --setup);
#   3. starts bringup + a fresh localisation on the car (scripts/simlingo_car.sh over ssh);
#   4. launches camera, ORION and the trajectory controller in the container with the
#      route as route_csv. The controller starts E-STOPPED: the model runs on the live
#      camera and prints plans, waypoints and inference times while the car stands still;
#   5. Enter releases the car, Enter again toggles the e-stop, Ctrl-C stops everything here.
#      "route finished" from the bridge engages the e-stop automatically.
#   6. n + Enter switches to another route WITHOUT reloading the model: car back in the start
#      box, fresh localisation on the car, the route sent to ORION (its temporal memory is
#      reset), the car e-stopped again. Takes under a minute instead of the ~5 min load.
#
# Options:
#   --fast | --int8 | --lite | --eager | --baseline
#                          inference variant (default --fast; see start_orion_f1tenth.sh --help)
#   --speed V              cap on the commanded speed in m/s (default 0.85)
#   --min-speed V          floor on the commanded speed while the plan wants to drive
#                          (default 0.5; 0 = off)
#   --speed-factor F       the model's speed is divided by F for the car, and the car's speed
#                          multiplied by F for the model (default 10 = the track's scale; also the
#                          distance scale). Unlike SimLingo, ORION keeps a temporal memory of ego
#                          poses, so a speed factor other than the distance scale feeds it
#                          displacements that do not match the speed it is told. Try lower
#                          values only if the model's speed does not move the car.
#   --command N|geometry   RoadOption fed to the model: geometry (default) derives LEFT/RIGHT from
#                          the route; a fixed 1 left, 2 right, 3 straight, 4 lanefollow otherwise
#   --hfov DEG             horizontal FOV of the camera; > 70 crops the central 70 deg the
#                          model's CAM_FRONT intrinsics describe (default 0 = no crop)
#   --side-views M         copy (default: camera in the three front slots) | black
#   --creep-speed V        speed of the kick-start in m/s (default 1.5, NOT limited by --speed)
#   --creep-hold V         the kick ends once the car rolls at V m/s, then held (default 0.7)
#   --creep-after S        kick-start after S seconds at standstill while released (default 5, 0 = off)
#   --steer-smooth S       low-pass on the steering command, time constant in s (default 0.2)
#   --save-frames          (default) save one JPEG per plan to log/orion/frames_route_<N>_<time>/: the
#                          CAM_FRONT frame with the predicted plan (red; its 2 s extrapolation thin),
#                          the route ahead (yellow) and the route node (blue) drawn in ORION's
#                          CAM_FRONT geometry, a top-down view in real metres, and a panel with
#                          command, speeds and forward time. Written on a separate thread.
#   --no-save-frames       do not save them
#   --dry                  never release the e-stop
#   --keep-localization    do not restart the localisation on the car (car not moved by hand)
#
# Environment: CAR (default f1tenth@10.183.247.250), CAMERA_DEV (default /dev/video0).
#
# The car and the Orin talk DDS over the Wi-Fi by unicast: CycloneDDS here (peer = car),
# Fast DDS on the car (peer = Orin). Both configs are generated per run from the
# addresses in use, into log/orion/ here and log/simlingo/ on the car.
set -o pipefail
# The whole script is one { ... } block: bash parses it completely before running it, so
# editing this file while a run is in progress no longer derails that run (2026-10-07).
{

CAR=${CAR:-f1tenth@10.183.247.250}
CAR_IP=${CAR#*@}
CAMERA_DEV=${CAMERA_DEV:-/dev/video0}
DOMAIN=5                                   # must equal DOMAIN in simlingo_car.sh
NAME=orion_f1tenth
VLA_WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=$(dirname "$VLA_WS")                 # mounted at /benchmarking in the container
CTR_WS=/benchmarking/$(basename "$VLA_WS")
START_ORIN="$BENCH/start_orion_f1tenth.sh"
ROUTES="$VLA_WS/routes"
LOGDIR="$VLA_WS/log/orion"
CYCLONE_XML="$LOGDIR/cyclone_orin.xml"
OF_WS=/opt/orion_f1tenth_ws
ESTOP_TOPIC=/orion/estop
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 "$CAR")

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; BLD=$'\e[1m'; RST=$'\e[0m'
ok(){   echo "  ${GRN}ok${RST}   $*"; }
warn(){ echo "  ${YEL}warn${RST} $*"; }
die(){  echo "  ${RED}FAIL${RST} $*" >&2; exit 1; }
step(){ echo; echo "==> $*"; }
banner(){ echo; echo "${BLD}${YEL}>>> $*${RST}"; echo; }

# This image has no /opt/ros_ws overlay (cyclonedds is inside /opt/ros/humble).
ENVSETUP="source /opt/ros/humble/setup.bash;
          source $OF_WS/install/setup.bash 2>/dev/null || true;
          export TORCHINDUCTOR_CACHE_DIR=/benchmarking/.torchinductor_cache TORCHINDUCTOR_FX_GRAPH_CACHE=1;"
DDS_ENV=(-e ROS_DOMAIN_ID=$DOMAIN -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
         -e "CYCLONEDDS_URI=file://$CTR_WS/log/orion/cyclone_orin.xml" -e PYTHONUNBUFFERED=1)

# ── arguments ────────────────────────────────────────────────────────────────
ROUTE_N=""; INFER=fast; SPEED=0.85; MIN_SPEED=0.5; FACTOR=10; COMMAND=geometry; HFOV=0; SIDE=copy
CREEP=5; CREEP_SPEED=1.5; CREEP_HOLD=0.7; STEER_SMOOTH=0.2; DRY=0; KEEP_LOC=""; SAVE_FRAMES=1; MODE=run; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --fast|--int8|--lite|--eager|--baseline) INFER=${1#--} ;;
    --speed)             SPEED=${2:-}; shift ;;
    --min-speed)         MIN_SPEED=${2:-}; shift ;;
    --speed-factor)      FACTOR=${2:-}; shift ;;
    --command)           COMMAND=${2:-}; shift ;;
    --hfov)              HFOV=${2:-}; shift ;;
    --side-views)        SIDE=${2:-}; shift ;;
    --creep-after)       CREEP=${2:-}; shift ;;
    --creep-speed)       CREEP_SPEED=${2:-}; shift ;;
    --creep-hold)        CREEP_HOLD=${2:-}; shift ;;
    --steer-smooth)      STEER_SMOOTH=${2:-}; shift ;;
    --dry)               DRY=1 ;;
    --save-frames)       SAVE_FRAMES=1 ;;
    --no-save-frames)    SAVE_FRAMES=0 ;;
    --keep-localization) KEEP_LOC=--keep-localization ;;
    --list)              MODE=list ;;
    --car-status)        MODE=car-status ;;
    --stop-car)          MODE=stop-car ;;
    -h|--help)           sed -n '2,/^set -o/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    --)                  shift; EXTRA=("$@"); break ;;
    [0-9]*)              ROUTE_N=$1 ;;
    *)                   die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done
num(){ [[ "$2" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "$1 needs a number, got '$2'"; }
asfloat(){ [[ "$1" == *.* ]] && echo "$1" || echo "$1.0"; }    # launch arguments are typed
num --speed "$SPEED"; num --min-speed "$MIN_SPEED"; num --speed-factor "$FACTOR"; num --hfov "$HFOV"
num --creep-after "$CREEP"; num --creep-speed "$CREEP_SPEED"; num --creep-hold "$CREEP_HOLD"; num --steer-smooth "$STEER_SMOOTH"
[[ "$FACTOR" =~ ^[0.]+$ ]] && die "--speed-factor must be > 0"
SPEED=$(asfloat "$SPEED"); MIN_SPEED=$(asfloat "$MIN_SPEED"); FACTOR=$(asfloat "$FACTOR"); HFOV=$(asfloat "$HFOV")
CREEP=$(asfloat "$CREEP"); CREEP_SPEED=$(asfloat "$CREEP_SPEED"); CREEP_HOLD=$(asfloat "$CREEP_HOLD"); STEER_SMOOTH=$(asfloat "$STEER_SMOOTH")
if [ "$COMMAND" = geometry ]; then CMD_ARGS="command_source:=geometry"
elif [[ "$COMMAND" =~ ^[1-6]$ ]]; then CMD_ARGS="command_source:=static driving_command:=$COMMAND"
else die "--command needs 1..6 or geometry, got '$COMMAND'"; fi
[[ "$SIDE" =~ ^(copy|black)$ ]] || die "--side-views needs copy or black, got '$SIDE'"

"${SSH[@]}" true 2>/dev/null || die "cannot ssh to $CAR (car on? same Wi-Fi? key installed with ssh-copy-id?)"

case "$MODE" in
  car-status) "${SSH[@]}" 'bash ~/vla_ws/scripts/simlingo_car.sh status'; exit ;;
  stop-car)   "${SSH[@]}" 'bash ~/vla_ws/scripts/simlingo_car.sh stop'; exit ;;
esac

# ── routes ───────────────────────────────────────────────────────────────────
step "Routes (from $CAR:vla_ws/routes)"
mkdir -p "$ROUTES" "$LOGDIR"
scp -q -p -o BatchMode=yes "$CAR:vla_ws/routes/route_*.csv" "$ROUTES/" || warn "could not copy routes from the car; using the local copies"
mapfile -t NUMS < <(ls "$ROUTES"/route_*.csv 2>/dev/null | sed 's/.*route_\([0-9]*\)\.csv/\1/' | sort -n)
[ ${#NUMS[@]} -gt 0 ] || die "no route_N.csv in $ROUTES"
for n in "${NUMS[@]}"; do
  awk -F, -v n="$n" 'NF >= 2 { if (c) len += sqrt(($1-x)^2 + ($2-y)^2); else { sx=$1; sy=$2 }; x=$1; y=$2; c++ }
       END { printf "  route %-2s  %2d waypoints  %5.1f m   (%.1f, %.1f) -> (%.1f, %.1f)\n", n, c, len, sx, sy, x, y }' \
      "$ROUTES/route_$n.csv"
done
[ "$MODE" = list ] && exit 0
if [ -z "$ROUTE_N" ]; then
  [ -t 0 ] || die "no route number given"
  read -r -p "Route number: " ROUTE_N
fi
ROUTE_FILE="$ROUTES/route_$ROUTE_N.csv"
[ -s "$ROUTE_FILE" ] || die "no such route: $ROUTE_FILE"
ok "route $ROUTE_N: $ROUTE_FILE"

# ── network ──────────────────────────────────────────────────────────────────
step "Network"
read -r IFACE ORIN_IP < <(ip -4 route get "$CAR_IP" | sed -n 's/.* dev \([^ ]*\) src \([0-9.]*\).*/\1 \2/p')
[ -n "$ORIN_IP" ] || die "no route to the car at $CAR_IP"
cat > "$CYCLONE_XML" <<XML
<!-- Generated by scripts/run_orion_route.sh. Orin $ORIN_IP ($IFACE), car $CAR_IP.
     Unicast only; the Orin's own address is a peer so its nodes find each other. -->
<CycloneDDS><Domain>
  <General><Interfaces><NetworkInterface name="$IFACE"/></Interfaces><AllowMulticast>false</AllowMulticast></General>
  <Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>
    <Peers><Peer Address="$CAR_IP"/><Peer Address="$ORIN_IP"/></Peers></Discovery>
</Domain></CycloneDDS>
XML
ok "Orin $ORIN_IP on $IFACE <-> car $CAR_IP, ROS domain $DOMAIN"

# ── Orin container ───────────────────────────────────────────────────────────
ZT_IFACE=$IFACE NANO_IP=$CAR_IP DOMAIN=$DOMAIN CYCLONE_XML=$CYCLONE_XML CAMERA_DEV=$CAMERA_DEV \
  "$START_ORIN" --setup --$INFER || die "Orin setup failed (see above)"
[ -e "$CAMERA_DEV" ] || die "no camera at $CAMERA_DEV"
# The install is a copy, not a symlink: pick up edits to the bridge packages (5 s).
docker exec "$NAME" bash -c "source /opt/ros/humble/setup.bash;
    cd /benchmarking/alpamayo-autoware &&
    colcon build --base-paths src /benchmarking/alpamayo-autoware/ackermann_msgs \
      /benchmarking/vla_ws/src/simlingo_f1tenth /benchmarking/vla_ws/src/orion_f1tenth \
      --packages-select orion_f1tenth simlingo_f1tenth --build-base $OF_WS/build --install-base $OF_WS/install" >/dev/null 2>&1 \
  || die "colcon build of orion_f1tenth / simlingo_f1tenth failed"
# leftovers of an earlier run would fight over the camera and /drive
docker exec "$NAME" bash -c 'pkill -INT -f "ros2 launch orion_f1tent[h]"; pkill -f "simlingo_esto[p].py"; sleep 1;
                             pkill -KILL -f "lib/orion_f1tent[h]/"; pkill -KILL -f "lib/simlingo_f1tent[h]/"; true'
ok "bridge packages up to date, no earlier launch running"

# ── car ──────────────────────────────────────────────────────────────────────
echo
scp -q -p -o BatchMode=yes "$VLA_WS/scripts/simlingo_car.sh" "$CAR:vla_ws/scripts/" || die "could not copy simlingo_car.sh to the car"
"${SSH[@]}" "bash ~/vla_ws/scripts/simlingo_car.sh start $ORIN_IP $KEEP_LOC" || die "car side did not come up (see above)"

# /pf/pose/odom and /odom from the car, as the container sees them; fails if one is dead
link_check() {
  step "Link check (car topics as seen from the Orin container)"
  LINK=$(docker exec -i "${DDS_ENV[@]}" "$NAME" bash -c "$ENVSETUP python3 -" <<'PY'
import time
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
rclpy.init()
node = rclpy.create_node('orion_link_check')
stamps = {'/pf/pose/odom': [], '/odom': []}
for t in stamps:
    node.create_subscription(Odometry, t, lambda m, t=t: stamps[t].append(time.time()), qos_profile_sensor_data)
end = time.time() + 15
while time.time() < end and not all(len(s) > 100 for s in stamps.values()):
    rclpy.spin_once(node, timeout_sec=0.2)
for t, s in stamps.items():
    if len(s) < 2:
        print(f'{t} DEAD')
    else:
        gaps = [b - a for a, b in zip(s, s[1:])]
        print(f'{t} ALIVE {len(gaps) / (s[-1] - s[0]):.0f} Hz, longest gap {max(gaps) * 1000:.0f} ms')
node.destroy_node()
rclpy.shutdown()
PY
)
  echo "$LINK" | sed 's/^/  /'
  [ -n "$LINK" ] && ! echo "$LINK" | grep -q DEAD
}
link_check || die "the Orin does not receive the car's topics (Wi-Fi changed? rerun; check log/orion/cyclone_orin.xml)"

# ── launch ───────────────────────────────────────────────────────────────────
RUNSTAMP=$(date +%Y%m%d_%H%M%S)
RUNLOG="$LOGDIR/route_${ROUTE_N}_${INFER}_$RUNSTAMP.log"
MEMLOG="$LOGDIR/route_${ROUTE_N}_${INFER}_$RUNSTAMP.tegrastats.log"
# Writes each line and fsyncs it: the Orin has frozen hard during runs (2026-10-07), and
# a frozen kernel never flushes the page cache, so plain tee loses the last minutes.
SYNC_TEE='import os, sys
f = open(sys.argv[1], "ab")
for line in sys.stdin.buffer:
    f.write(line); f.flush(); os.fsync(f.fileno())
    if len(sys.argv) < 3:
        sys.stdout.buffer.write(line); sys.stdout.flush()'
FRAMES_ARG=""; FRAMES_DIR=""
if [ "$SAVE_FRAMES" = 1 ]; then
  FRAMES_DIR="$LOGDIR/frames_route_${ROUTE_N}_${INFER}_$RUNSTAMP"
  FRAMES_ARG="save_frames_dir:=$CTR_WS/log/orion/$(basename "$FRAMES_DIR")"
  mkdir -p "$FRAMES_DIR"     # owned by this user; the node gives every frame the same owner
fi
FIFO="$LOGDIR/.estop_fifo.$$"
LAUNCHED=0
NOISE='warnings\.warn|UserWarning|FutureWarning|^\[orion_realworld_node-[0-9]+\] (- |\. Make sure)'

summary() {
  [ -s "$RUNLOG" ] || return 0
  echo
  echo "==> Summary (full log: $RUNLOG)"
  python3 - "$RUNLOG" <<'PY'
import re, sys
fwd, per = [], []
for line in open(sys.argv[1], errors="replace"):
    if "[ORION] frame" not in line:
        continue
    if (m := re.search(r"forward (\d+) ms", line)):
        fwd.append(int(m.group(1)))
    if (m := re.search(r"period (\d+) ms", line)):
        per.append(int(m.group(1)))
if not fwd:
    print("  no plan was produced")
else:
    used = fwd[2:] if len(fwd) > 4 else fwd          # the first plans include warm-up effects
    note = " (first two plans excluded)" if len(used) < len(fwd) else ""
    print(f"  {len(fwd)} plans, forward mean {sum(used) / len(used):.0f} ms, min {min(used)}, max {max(used)}{note}")
    if per:
        print(f"  period between forwards mean {sum(per) / len(per):.0f} ms, min {min(per)}, max {max(per)}")
PY
  grep "\[PROF\]" "$RUNLOG" | tail -1 | sed 's/.*\[PROF\]/  node total, /'
}

cleanup() {
  trap - INT TERM EXIT
  if [ "$LAUNCHED" = 1 ]; then
    echo; echo "==> Stopping ORION on the Orin"
    echo stop >&9 2>/dev/null
    docker exec "$NAME" bash -c 'pkill -INT -f "ros2 launch orion_f1tent[h]"
        for i in $(seq 32); do pgrep -f "lib/orion_f1tent[h]/" >/dev/null || break; sleep 0.25; done
        pkill -KILL -f "lib/orion_f1tent[h]/"; pkill -KILL -f "lib/simlingo_f1tent[h]/"; pkill -f "simlingo_esto[p].py"; true'
    exec 9>&-
    wait 2>/dev/null
    summary
    if [ -n "$FRAMES_DIR" ] && [ -d "$FRAMES_DIR" ]; then
      # written as root inside the container
      docker exec "$NAME" chown -R "$(id -u):$(id -g)" "$CTR_WS/log/orion/$(basename "$FRAMES_DIR")" 2>/dev/null
      echo "  $(ls "$FRAMES_DIR" | wc -l) plan frames in $FRAMES_DIR"
    fi
    echo "  The car side (bringup + localisation) keeps running; stop it with: $0 --stop-car"
  fi
  [ -n "${TEGRA_PID:-}" ] && { pkill -P "$TEGRA_PID" 2>/dev/null; kill "$TEGRA_PID" 2>/dev/null; pkill -x tegrastats 2>/dev/null; }
  rm -f "$FIFO"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

step "Launch: route $ROUTE_N, $INFER inference, model speed / $FACTOR, commanded speed $MIN_SPEED-$SPEED m/s, command $COMMAND, camera hfov $HFOV, side views $SIDE, controller starts e-stopped (model load takes minutes)"
mkfifo "$FIFO" && exec 9<>"$FIFO" || die "could not create $FIFO"
LAUNCHED=1
# memory / GPU / temperature / power once a second, to see what precedes a freeze
( tegrastats --interval 1000 | python3 -c "$SYNC_TEE" "$MEMLOG" quiet ) </dev/null >/dev/null 2>&1 &
TEGRA_PID=$!
docker exec -i "${DDS_ENV[@]}" "$NAME" bash -c "$ENVSETUP exec python3 -u $CTR_WS/scripts/simlingo_estop.py $ESTOP_TOPIC" <&9 >/dev/null 2>&1 &
docker exec "${DDS_ENV[@]}" "$NAME" bash -c "$ENVSETUP
  exec ros2 launch orion_f1tenth orion_f1tenth.launch.py \
    inference_mode:=$INFER camera_device:=$CAMERA_DEV \
    route_csv:=$CTR_WS/routes/route_$ROUTE_N.csv max_speed_mps:=$SPEED start_estopped:=true estop_topic:=$ESTOP_TOPIC \
    speed_world_scale:=$FACTOR min_speed_mps:=$MIN_SPEED steer_smoothing_sec:=$STEER_SMOOTH \
    $CMD_ARGS camera_hfov_deg:=$HFOV side_views:=$SIDE \
    creep_after_sec:=$CREEP creep_speed_mps:=$CREEP_SPEED creep_release_speed_mps:=$CREEP_HOLD \
    creep_hold_speed_mps:=$CREEP_HOLD $FRAMES_ARG ${EXTRA[*]}" 2>&1 </dev/null \
  | python3 -u -c "$SYNC_TEE" "$RUNLOG" | grep --line-buffered -vE "$NOISE" &
LAUNCH_PID=$!

# weights off the SSD (~200 s) + warm-up compile (minutes the first time per variant)
for _ in $(seq 900); do
  grep -q "ready: world_scale" "$RUNLOG" 2>/dev/null && break
  kill -0 "$LAUNCH_PID" 2>/dev/null || die "the launch exited before the model was loaded (log: $RUNLOG)"
  sleep 1
done
grep -q "ready: world_scale" "$RUNLOG" || die "model not loaded after 15 minutes (log: $RUNLOG)"
for _ in $(seq 20); do grep -q "\[ORION\] frame" "$RUNLOG" && break; sleep 1; done
grep -q "\[ORION\] frame" "$RUNLOG" || warn "model loaded but no plan yet: see the 'waiting for ...' lines above"

if [ "$DRY" = 1 ] || [ ! -t 0 ]; then
  banner "Dry run: ORION is planning on the live camera, the car stays e-stopped. Ctrl-C to quit."
  wait "$LAUNCH_PID"
  exit 0
fi

# ── routes, one model load ───────────────────────────────────────────────────
# The model stays loaded between routes: "n" switches to another route (car back in the
# start box, fresh localisation on the car, the route sent on /global_path, which also
# resets ORION's temporal memory) in under a minute instead of a 5-minute reload.

logsize() { stat -c %s "$RUNLOG" 2>/dev/null || echo 0; }
# true once PATTERN appears in the run log after byte offset OFF
logged_since() { tail -c +$(( $2 + 1 )) "$RUNLOG" 2>/dev/null | grep -qa -- "$1"; }

# Publishes route_N.csv as a latched nav_msgs/Path on /global_path for the ORION node.
publish_route() {
  docker exec -i "${DDS_ENV[@]}" "$NAME" bash -c "$ENVSETUP python3 - $CTR_WS/routes/route_$1.csv" <<'PY'
import sys, time
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from simlingo_f1tenth.route_planner import load_route_csv
wps = load_route_csv(sys.argv[1])
rclpy.init()
node = rclpy.create_node('orion_route_switch')
qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                 depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
pub = node.create_publisher(Path, '/global_path', qos)
msg = Path()
msg.header.frame_id = 'map'
msg.header.stamp = node.get_clock().now().to_msg()
for x, y in wps:
    ps = PoseStamped()
    ps.header = msg.header
    ps.pose.position.x, ps.pose.position.y = float(x), float(y)
    ps.pose.orientation.w = 1.0
    msg.poses.append(ps)
end = time.time() + 10
while time.time() < end and pub.get_subscription_count() == 0:
    rclpy.spin_once(node, timeout_sec=0.1)
pub.publish(msg)
end = time.time() + 2                  # stay up for late joiners of the latched topic
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.1)
node.destroy_node()
rclpy.shutdown()
PY
}

# Asks for the next route and switches the running ORION to it; the car stays e-stopped.
next_route() {
  local n def=$(( CUR_ROUTE + 1 )) off
  [ -s "$ROUTES/route_$def.csv" ] || def=$CUR_ROUTE
  echo
  read -r -p "  Next route number [$def], q = back: " n
  case "$n" in q|Q) return 1 ;; "") n=$def ;; esac
  [ -s "$ROUTES/route_$n.csv" ] || { warn "no such route: $ROUTES/route_$n.csv"; return 1; }
  read -r -p "  Put the car in the START BOX, then Enter (q = back): " ans
  case "$ans" in q|Q) return 1 ;; esac
  step "Fresh localisation on the car"
  "${SSH[@]}" "bash ~/vla_ws/scripts/simlingo_car.sh start $ORIN_IP" || { warn "car side did not come up (see above)"; return 1; }
  link_check || { warn "the Orin does not receive the car's topics; try n again"; return 1; }
  step "Route $n -> ORION"
  off=$(logsize)
  publish_route "$n" >/dev/null 2>&1
  for _ in $(seq 30); do logged_since "route set from" "$off" && break; sleep 0.5; done
  logged_since "route set from" "$off" || { warn "ORION did not take the route (see the log); try n again"; return 1; }
  for _ in $(seq 40); do logged_since "\[ORION\] frame" "$off" && break; sleep 0.5; done
  CUR_ROUTE=$n
  ok "route $n: $ROUTES/route_$n.csv, ORION memory reset"
  return 0
}

CUR_ROUTE=$ROUTE_N
[ "$CREEP" = 0.0 ] || KICK_NOTE="
>>> After release the car is kick-started ($CREEP_SPEED m/s until it rolls at $CREEP_HOLD m/s) whenever it has stood still for $CREEP s."
stopped_banner() {
  banner "Route $CUR_ROUTE: ORION is planning, the car is E-STOPPED.
>>> Enter = RELEASE the car (max $SPEED m/s), n + Enter = next route, Ctrl-C = quit.$1"
}
released_banner() { banner "Route $CUR_ROUTE: RELEASED. Enter = e-stop, Ctrl-C = stop everything."; }

STATE=stop
WATCH_OFF=$(logsize)
stopped_banner "${KICK_NOTE:-}"
while :; do
  kill -0 "$LAUNCH_PID" 2>/dev/null || die "the launch exited (log: $RUNLOG)"
  if [ "$STATE" = go ] && logged_since "route finished" "$WATCH_OFF"; then
    STATE=stop; echo stop >&9
    stopped_banner "
>>> Route $CUR_ROUTE finished: e-stop engaged."
  fi
  read -r -t 0.3 line || continue
  case "$line" in
    "")
      if [ "$STATE" = go ]; then
        STATE=stop; echo stop >&9; stopped_banner ""
      else
        WATCH_OFF=$(logsize); STATE=go; echo go >&9; released_banner
      fi ;;
    n|N|next)
      STATE=stop; echo stop >&9
      next_route || true
      stopped_banner "" ;;
    *)
      [ "$STATE" = go ] && { STATE=stop; echo stop >&9; }    # anything unexpected: stop first
      stopped_banner "" ;;
  esac
done
exit
}
