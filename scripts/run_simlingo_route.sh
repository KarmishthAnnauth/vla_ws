#!/usr/bin/env bash
#
# run_simlingo_route.sh — run SimLingo on the real F1TENTH along one recorded global route.
#
# Start it on the Orin, car in the start box:
#
#   scripts/run_simlingo_route.sh            list the routes and ask which one
#   scripts/run_simlingo_route.sh 1          route_1.csv
#   scripts/run_simlingo_route.sh 1 --dry    inference only: the car stays e-stopped
#   scripts/run_simlingo_route.sh 1 --speed 0.8 --keep-localization
#   scripts/run_simlingo_route.sh 1 -- route_min_spacing_m:=0.5   (extra launch arguments after --)
#   scripts/run_simlingo_route.sh --list | --car-status | --stop-car
#
# What it does:
#   1. copies the route CSVs from the car (~/vla_ws/routes) to vla_ws/routes here;
#   2. prepares the Orin container (simlingo/start_orin_f1tenth.sh --setup);
#   3. starts bringup + a fresh localisation on the car (scripts/simlingo_car.sh over ssh);
#   4. launches camera, SimLingo and the trajectory controller in the container with the
#      route as route_csv. The controller starts E-STOPPED: the model runs on the live
#      camera and prints plans, waypoints and inference times while the car stands still;
#   5. Enter releases the car, Enter again toggles the e-stop, Ctrl-C stops everything here.
#      "route finished" from the bridge engages the e-stop automatically.
#
# Options:
#   --speed V              cap on the commanded speed in m/s (default 0.85, the steady pace of
#                          the manual drives of 2026-10-02). With the cap at 1.0 the car was
#                          faster than that pace a fifth of the time. At 0.5 the cap, not the
#                          model, set the speed.
#   --min-speed V          floor on the commanded speed while the plan wants to drive
#                          (default 0.5; 0 = off). Below that the drivetrain's friction stops
#                          the car. Plans under 0.04 m/s still stop it.
#   --speed-factor F       the model's speed is divided by F for the car, and the car's speed
#                          multiplied by F for the model (default 7). 10 is the track's true
#                          scale, at which the model's speed is too slow to move the car;
#                          distances always use 10. 7 matches the manual reference drives of
#                          2026-10-02 (medians 0.65 and 0.83 m/s without the launches): at 7
#                          the plans ask for a median of 0.71 m/s (10-90 %: 0.37-1.17 m/s).
#   --creep-speed V        speed of the kick-start in m/s (default 1.5, NOT limited by --speed).
#                          0.5 m/s for 0.75 s left the car at 0.1 m/s on 2026-10-01.
#   --creep-hold V         the kick ends once the car rolls at V m/s, and V is then held for the
#                          rest of the 2 s (default 0.7, the pace of the manual drive; limited
#                          by --speed). On 2026-10-02 a kick of 1.0 m/s that ended at 0.3 m/s
#                          (after 0.4 s) and was held at 0.5 m/s let the car stall again.
#   --simlingo-pid         steer with SimLingo's own lateral PID instead of pure pursuit
#   --pid-gain G           with --simlingo-pid: multiplier on the PID's gains k_p, k_i, k_d
#                          (default 0.4; 1 = as tuned upstream for CARLA). At 1 the steering
#                          was at full lock in 40-60 % of the samples (2026-10-01/02): full
#                          lock is reached at 13/G deg of heading error to a point ~0.25 m
#                          ahead, and each new plan moves the path by centimetres. Too low and
#                          the car cannot take the corners. Route 2 on 2026-10-02, distance
#                          from the route in the turn (max) and heading wobble on the straight:
#                          1.0: 28 cm, 21 deg | 0.4: 26 cm, 7 deg | 0.3: 31-44 cm, 5-11 deg |
#                          0.2: 62 cm, 7 deg | pure pursuit: 26-38 cm, 3-5 deg.
#   --steer-smooth S       low-pass on the steering command, time constant in s (default 0.2
#                          with pure pursuit, 0 = off with --simlingo-pid). Each new plan
#                          (every 0.5 s) shifts the path and made the steering jump by a
#                          median of 5-7 deg; higher is smoother but the car reacts later and
#                          cuts or overshoots corners more. Leave it off with --simlingo-pid:
#                          the PID looks only ~0.25 m ahead, and with the filter's delay the
#                          car swung from side to side (2026-10-02, 0.2 s).
#   --think                thinking mode: SimLingo's chain-of-thought prompt ("What should the
#                          ego do next?" instead of "Predict the waypoints."), the default of
#                          upstream agent_simlingo.py. The model first writes a commentary
#                          (printed as "language: ...") and decodes the waypoints after it;
#                          each word of it adds ~10 ms per plan.
#   --save-frames          save one JPEG per plan to log/simlingo/frames_route_<N>_<time>/: the
#                          frame the model saw with its predicted route (red), speed waypoints
#                          (green) and target points (blue) drawn in the training camera's
#                          geometry, and a panel with the language output (the commentary with
#                          --think), speeds and model time. Written on a separate thread.
#   --dry                  never release the e-stop
#   --creep-after S        kick-start: after S seconds at standstill while released, command
#                          the --creep-speed until the car rolls at the --creep-hold speed, then
#                          that speed, 2 s in total (default 5, 0 = off). From standstill the model
#                          keeps predicting standstill; once the car rolls it plans a speed.
#                          This is the upstream agent's stuck recovery (there: after 40 s).
#   --keep-localization    do not restart the localisation on the car (car not moved by hand)
#
# Environment: CAR (default f1tenth@10.183.247.250), CAMERA_DEV (default /dev/video0).
#
# The car and the Orin talk DDS over the Wi-Fi by unicast: CycloneDDS here (peer = car),
# Fast DDS on the car (peer = Orin). Both configs are generated per run from the
# addresses in use, into log/simlingo/ on either side.
set -o pipefail

CAR=${CAR:-f1tenth@10.183.247.250}
CAR_IP=${CAR#*@}
CAMERA_DEV=${CAMERA_DEV:-/dev/video0}
DOMAIN=5                                   # must equal DOMAIN in simlingo_car.sh
NAME=sim_f1tenth
VLA_WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=$(dirname "$VLA_WS")                 # mounted at /benchmarking in the container
CTR_WS=/benchmarking/$(basename "$VLA_WS")
START_ORIN="$BENCH/simlingo/start_orin_f1tenth.sh"
ROUTES="$VLA_WS/routes"
LOGDIR="$VLA_WS/log/simlingo"
CYCLONE_XML="$LOGDIR/cyclone_orin.xml"
CKPT=/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
F1_WS=/opt/f1tenth_ws
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 "$CAR")

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; BLD=$'\e[1m'; RST=$'\e[0m'
ok(){   echo "  ${GRN}ok${RST}   $*"; }
warn(){ echo "  ${YEL}warn${RST} $*"; }
die(){  echo "  ${RED}FAIL${RST} $*" >&2; exit 1; }
step(){ echo; echo "==> $*"; }
banner(){ echo; echo "${BLD}${YEL}>>> $*${RST}"; echo; }

ENVSETUP="source /opt/ros/humble/setup.bash;
          source /opt/ros_ws/install/setup.bash;
          source $F1_WS/install/setup.bash 2>/dev/null || true;
          export PYTHONPATH=\"/benchmarking/simlingo:/benchmarking/simlingo/team_code:\$PYTHONPATH\";"
DDS_ENV=(-e ROS_DOMAIN_ID=$DOMAIN -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
         -e "CYCLONEDDS_URI=file://$CTR_WS/log/simlingo/cyclone_orin.xml" -e PYTHONUNBUFFERED=1)

# ── arguments ────────────────────────────────────────────────────────────────
ROUTE_N=""; SPEED=0.85; MIN_SPEED=0.5; FACTOR=7; CREEP=5; CREEP_SPEED=1.5; CREEP_HOLD=0.7; STEER_SMOOTH=""; PID_GAIN=0.4; DRY=0; KEEP_LOC=""; LATERAL=pure_pursuit; USE_COT=false; SAVE_FRAMES=0; MODE=run; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --speed)             SPEED=${2:-}; shift ;;
    --min-speed)         MIN_SPEED=${2:-}; shift ;;
    --speed-factor)      FACTOR=${2:-}; shift ;;
    --creep-after)       CREEP=${2:-}; shift ;;
    --creep-speed)       CREEP_SPEED=${2:-}; shift ;;
    --creep-hold)        CREEP_HOLD=${2:-}; shift ;;
    --simlingo-pid)      LATERAL=simlingo_pid ;;
    --steer-smooth)      STEER_SMOOTH=${2:-}; shift ;;
    --pid-gain)          PID_GAIN=${2:-}; shift ;;
    --think)             USE_COT=true ;;
    --save-frames)       SAVE_FRAMES=1 ;;
    --dry)               DRY=1 ;;
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
[[ "$SPEED" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--speed needs a number in m/s, got '$SPEED'"
[[ "$CREEP" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--creep-after needs a number of seconds, got '$CREEP'"
[[ "$FACTOR" =~ ^[0-9]+(\.[0-9]+)?$ && ! "$FACTOR" =~ ^[0.]+$ ]] || die "--speed-factor needs a number > 0, got '$FACTOR'"
[[ "$FACTOR" == *.* ]] || FACTOR="$FACTOR.0"
[[ "$CREEP_SPEED" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--creep-speed needs a number in m/s, got '$CREEP_SPEED'"
[[ "$CREEP_SPEED" == *.* ]] || CREEP_SPEED="$CREEP_SPEED.0"
[[ "$MIN_SPEED" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--min-speed needs a number in m/s, got '$MIN_SPEED'"
[[ "$MIN_SPEED" == *.* ]] || MIN_SPEED="$MIN_SPEED.0"
[[ "$CREEP_HOLD" =~ ^[0-9]+(\.[0-9]+)?$ && ! "$CREEP_HOLD" =~ ^[0.]+$ ]] || die "--creep-hold needs a number > 0 in m/s, got '$CREEP_HOLD'"
[[ "$CREEP_HOLD" == *.* ]] || CREEP_HOLD="$CREEP_HOLD.0"
if [ -z "$STEER_SMOOTH" ]; then STEER_SMOOTH=0.2; [ "$LATERAL" = simlingo_pid ] && STEER_SMOOTH=0.0; fi
[[ "$STEER_SMOOTH" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--steer-smooth needs a number of seconds, got '$STEER_SMOOTH'"
[[ "$STEER_SMOOTH" == *.* ]] || STEER_SMOOTH="$STEER_SMOOTH.0"
[[ "$PID_GAIN" =~ ^[0-9]+(\.[0-9]+)?$ && ! "$PID_GAIN" =~ ^[0.]+$ ]] || die "--pid-gain needs a number > 0, got '$PID_GAIN'"
[[ "$PID_GAIN" == *.* ]] || PID_GAIN="$PID_GAIN.0"
STEERING=$LATERAL; [ "$LATERAL" = simlingo_pid ] && STEERING="simlingo_pid with gains x $PID_GAIN"
[[ "$SPEED" == *.* ]] || SPEED="$SPEED.0"      # launch arguments are typed: 1 would be an integer
[[ "$CREEP" == *.* ]] || CREEP="$CREEP.0"

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
# Written in place: the container reads it through the /benchmarking mount.
cat > "$CYCLONE_XML" <<XML
<!-- Generated by scripts/run_simlingo_route.sh. Orin $ORIN_IP ($IFACE), car $CAR_IP.
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
  "$START_ORIN" --setup || die "Orin setup failed (see above)"
[ -e "$CAMERA_DEV" ] || die "no camera at $CAMERA_DEV"
# The install is a copy, not a symlink: pick up edits to the bridge (3 s).
docker exec "$NAME" bash -c "source /opt/ros/humble/setup.bash; source /opt/ros_ws/install/setup.bash;
    cd /benchmarking/alpamayo-autoware &&
    colcon build --base-paths src /benchmarking/alpamayo-autoware/ackermann_msgs \
      /benchmarking/vla_ws/src/simlingo_f1tenth /benchmarking/vla_ws/src/orion_f1tenth \
      --packages-select simlingo_f1tenth --build-base $F1_WS/build --install-base $F1_WS/install" >/dev/null 2>&1 \
  || die "colcon build of simlingo_f1tenth failed"
# leftovers of an earlier run would fight over the camera and /drive
docker exec "$NAME" bash -c 'pkill -INT -f "ros2 launch simlingo_f1tent[h]"; pkill -f "simlingo_esto[p].py"; sleep 1;
                             pkill -KILL -f "lib/simlingo_f1tent[h]/"; true'
ok "bridge package up to date, no earlier launch running"

# ── car ──────────────────────────────────────────────────────────────────────
echo
scp -q -p -o BatchMode=yes "$VLA_WS/scripts/simlingo_car.sh" "$CAR:vla_ws/scripts/" || die "could not copy simlingo_car.sh to the car"
"${SSH[@]}" "bash ~/vla_ws/scripts/simlingo_car.sh start $ORIN_IP $KEEP_LOC" || die "car side did not come up (see above)"

step "Link check (car topics as seen from the Orin container)"
LINK=$(docker exec -i "${DDS_ENV[@]}" "$NAME" bash -c "$ENVSETUP python3 -" <<'PY'
import time
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
rclpy.init()
node = rclpy.create_node('simlingo_link_check')
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
echo "$LINK" | grep -q DEAD && die "the Orin does not receive the car's topics (Wi-Fi changed? rerun; check log/simlingo/cyclone_orin.xml)"
[ -n "$LINK" ] || die "link check did not run"

# ── launch ───────────────────────────────────────────────────────────────────
RUNSTAMP=$(date +%Y%m%d_%H%M%S)
RUNLOG="$LOGDIR/route_${ROUTE_N}_$RUNSTAMP.log"
FRAMES_ARG=""; FRAMES_DIR=""
if [ "$SAVE_FRAMES" = 1 ]; then
  FRAMES_DIR="$LOGDIR/frames_route_${ROUTE_N}_$RUNSTAMP"
  FRAMES_ARG="save_frames_dir:=$CTR_WS/log/simlingo/$(basename "$FRAMES_DIR")"
  mkdir -p "$FRAMES_DIR"     # owned by this user; the node gives every frame the same owner
fi
FIFO="$LOGDIR/.estop_fifo.$$"
LAUNCHED=0
NOISE='language: Waypoints:$|huggingface\.co|^\[simlingo_realworld_node-[0-9]+\] (- |\. Make sure)|Axes3D|warnings\.warn'

summary() {
  [ -s "$RUNLOG" ] || return 0
  echo
  echo "==> Summary (full log: $RUNLOG)"
  python3 - "$RUNLOG" <<'PY'
import re, sys
ms = [int(m.group(1)) for line in open(sys.argv[1], errors="replace")
      if "[SIMLINGO] tp0" in line and (m := re.search(r"model (\d+) ms", line))]
if not ms:
    print("  no plan was produced")
else:
    used = ms[2:] if len(ms) > 4 else ms          # the first two plans include warm-up
    note = " (first two plans excluded)" if len(used) < len(ms) else ""
    print(f"  {len(ms)} plans, model time mean {sum(used) / len(used):.0f} ms, "
          f"min {min(used)}, max {max(used)}{note}")
PY
  grep "\[PROF\]" "$RUNLOG" | tail -1 | sed 's/.*\[PROF\]/  node total, /'
}

cleanup() {
  trap - INT TERM EXIT
  if [ "$LAUNCHED" = 1 ]; then
    echo; echo "==> Stopping SimLingo on the Orin"
    echo stop >&9 2>/dev/null
    docker exec "$NAME" bash -c 'pkill -INT -f "ros2 launch simlingo_f1tent[h]"
        for i in $(seq 32); do pgrep -f "lib/simlingo_f1tent[h]/" >/dev/null || break; sleep 0.25; done
        pkill -KILL -f "lib/simlingo_f1tent[h]/"; pkill -f "simlingo_esto[p].py"; true'
    exec 9>&-
    wait 2>/dev/null
    summary
    if [ -n "$FRAMES_DIR" ] && [ -d "$FRAMES_DIR" ]; then
      # written as root inside the container
      docker exec "$NAME" chown -R "$(id -u):$(id -g)" "$CTR_WS/log/simlingo/$(basename "$FRAMES_DIR")" 2>/dev/null
      echo "  $(ls "$FRAMES_DIR" | wc -l) plan frames in $FRAMES_DIR"
    fi
    echo "  The car side (bringup + localisation) keeps running; stop it with: $0 --stop-car"
  fi
  rm -f "$FIFO"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

step "Launch: route $ROUTE_N, model speed / $FACTOR, commanded speed $MIN_SPEED-$SPEED m/s, steering $STEERING (smoothing $STEER_SMOOTH s), thinking $USE_COT, controller starts e-stopped (model load takes 60-90 s)"
mkfifo "$FIFO" && exec 9<>"$FIFO" || die "could not create $FIFO"
LAUNCHED=1
docker exec -i "${DDS_ENV[@]}" "$NAME" bash -c "$ENVSETUP exec python3 -u $CTR_WS/scripts/simlingo_estop.py" <&9 >/dev/null 2>&1 &
docker exec "${DDS_ENV[@]}" "$NAME" bash -c "$ENVSETUP
  exec ros2 launch simlingo_f1tenth simlingo_f1tenth.launch.py \
    checkpoint_path:=$CKPT simlingo_path:=/benchmarking/simlingo camera_device:=$CAMERA_DEV \
    route_csv:=$CTR_WS/routes/route_$ROUTE_N.csv max_speed_mps:=$SPEED start_estopped:=true \
    speed_world_scale:=$FACTOR min_speed_mps:=$MIN_SPEED lateral_controller:=$LATERAL pid_gain:=$PID_GAIN steer_smoothing_sec:=$STEER_SMOOTH use_cot:=$USE_COT $FRAMES_ARG \
    creep_after_sec:=$CREEP creep_speed_mps:=$CREEP_SPEED creep_release_speed_mps:=$CREEP_HOLD \
    creep_hold_speed_mps:=$CREEP_HOLD ${EXTRA[*]}" 2>&1 </dev/null \
  | tee "$RUNLOG" | grep --line-buffered -vE "$NOISE" &
LAUNCH_PID=$!

for _ in $(seq 300); do
  grep -q "ready: world_scale" "$RUNLOG" 2>/dev/null && break
  kill -0 "$LAUNCH_PID" 2>/dev/null || die "the launch exited before the model was loaded (log: $RUNLOG)"
  sleep 1
done
grep -q "ready: world_scale" "$RUNLOG" || die "model not loaded after 5 minutes (log: $RUNLOG)"
for _ in $(seq 20); do grep -q "\[SIMLINGO\] tp0" "$RUNLOG" && break; sleep 1; done
grep -q "\[SIMLINGO\] tp0" "$RUNLOG" || warn "model loaded but no plan yet: see the 'waiting for ...' lines above"

if [ "$DRY" = 1 ] || [ ! -t 0 ]; then
  banner "Dry run: SimLingo is planning on the live camera, the car stays e-stopped. Ctrl-C to quit."
  wait "$LAUNCH_PID"
  exit 0
fi

# ── arming ───────────────────────────────────────────────────────────────────
PROMPT="SimLingo is planning, the car is E-STOPPED. Enter = RELEASE the car (max $SPEED m/s), Ctrl-C = quit."
[ "$CREEP" = 0.0 ] || PROMPT="$PROMPT
>>> After release the car is kick-started ($CREEP_SPEED m/s until it rolls at $CREEP_HOLD m/s) whenever it has stood still for $CREEP s."
banner "$PROMPT"
while :; do
  read -r -t 15 _; rc=$?
  [ $rc -eq 0 ] && break
  kill -0 "$LAUNCH_PID" 2>/dev/null || die "the launch exited (log: $RUNLOG)"
  [ $rc -gt 128 ] && banner "$PROMPT"
done

FINISHED="$LOGDIR/.route_finished.$$"
rm -f "$FINISHED"
( while kill -0 "$LAUNCH_PID" 2>/dev/null; do
    if grep -q "route finished" "$RUNLOG"; then
      echo stop >&9; touch "$FINISHED"
      banner "Route finished: e-stop engaged. Enter = release again, Ctrl-C = quit."
      break
    fi
    sleep 0.3
  done ) &

STATE=go
echo go >&9
banner "RELEASED. Enter = e-stop, Ctrl-C = stop everything."
while kill -0 "$LAUNCH_PID" 2>/dev/null; do
  read -r -t 2 _; rc=$?
  [ $rc -eq 0 ] || continue
  if [ -e "$FINISHED" ]; then rm -f "$FINISHED"; STATE=stop; fi
  if [ "$STATE" = go ]; then
    STATE=stop; echo stop >&9; banner "E-STOP engaged. Enter = release, Ctrl-C = quit."
  else
    STATE=go; echo go >&9; banner "RELEASED. Enter = e-stop, Ctrl-C = stop everything."
  fi
done
