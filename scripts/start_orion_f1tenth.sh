#!/usr/bin/env bash
#
# start_orion_f1tenth.sh — bring up the Orin side of ORION on the real F1TENTH.
#
# Derived from start_orion.sh (CARLA eval) and simlingo/start_orin_f1tenth.sh
# (SimLingo on the car): the ORION container, weights and clock checks of the
# first, the car-facing network, camera device and workspace handling of the
# second. Launches vla_ws/src/orion_f1tenth.
#
# Self-contained in vla_ws: the ROS packages the bridge needs besides src/ (orion_ros,
# simlingo_ros, autoware/ackermann/carla msgs) are vendored in deps/. The container
# mounts the directory ABOVE vla_ws at /benchmarking (the torch inductor cache lives
# there, /benchmarking/.torchinductor_cache), so clone vla_ws into a directory of its
# own, e.g. ~/benchmarking/vla_ws. Outside the repo: the image and the weights.
#
#   scripts/start_orion_f1tenth.sh           launch (sets everything up first, idempotent)
#   scripts/start_orion_f1tenth.sh --check         preflight only, don't launch
#   scripts/start_orion_f1tenth.sh --setup         preflight + container + workspace, don't launch
#                                            (scripts/run_orion_route.sh launches itself)
#   scripts/start_orion_f1tenth.sh --shell         drop into a shell in the container
#   scripts/start_orion_f1tenth.sh --stop          stop and remove the container
#   scripts/start_orion_f1tenth.sh --rebuild       force a colcon rebuild, then launch
#
# Inference variant (default --fast):
#   --fast      exact fp16 speedups + torch.compile: ~1.0 s per frame, trajectory
#               within 1 cm of the stock model (ORION_ROS_NODE.md section 6)
#   --int8      --fast plus W8A8 INT8 for the LLM MLP projections (SmoothQuant
#               calibration in deps/orion_ros/engines/llm_act_stats.pt): ~0.9 s per
#               frame, 3-5 cm trajectory shift on the bench frames
#   --lite      --fast plus 512 px ViT input and rear views every other frame:
#               ~0.75 s, 0.47 m shift -- a different model in effect
#   --eager     exact speedups without torch.compile (~1.5 s, no compile wait)
#   --baseline  the agent as is (~1.75 s)
#
# Anything of the form key:=value is forwarded to `ros2 launch`, e.g.
#   scripts/start_orion_f1tenth.sh --int8 camera_hfov_deg:=110.0 side_views:=black
#
# Environment overrides:
#   NANO_IP        car's address (default 10.241.27.5 -- set it!)
#   ZT_IFACE       interface towards the car (default the ZeroTier one; the Wi-Fi works too)
#   ORION_WEIGHTS_HOST  directory with Orion.pth + pretrain_qformer/ (default: auto-detect)
#   DOMAIN         ROS_DOMAIN_ID shared with the car (default 5, vla_ws/README.md)
#   CAMERA_DEV     camera device passed into the container (default /dev/video0)
#   CYCLONE_XML    CycloneDDS config (default: generated per NANO_IP in /tmp)
#
# Four things carried over from start_orion.sh, all load-bearing:
#   1. Weights live on an external SSD (Intenso/Orin, or KINGSTON/models/Orion
#      over its manual exfat-FUSE mount). Preflight finds them or stops.
#   2. The model must be driven from the container's PRE-BUILT /root/Orion (the
#      only checkout with compiled mmcv ops), not the bind-mounted one.
#   3. This image has no /opt/ros_ws overlay -- cyclonedds ships inside
#      /opt/ros/humble -- so only two overlays get sourced.
#   4. Model load is minutes: ~200 s of weights off the SSD, then the warm-up
#      compile (~3 min the first time per variant; inductor caches it).
set -eo pipefail

IMAGE=karmishthannauth/orion_env_ros:v01
NAME=orion_f1tenth
VLA_WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=$(dirname "$VLA_WS")                 # mounted at /benchmarking in the container
CTR_WS=/benchmarking/$(basename "$VLA_WS")
# Where the ORION weights are (Orion.pth + pretrain_qformer/). Either SSD works;
# the Intenso is the one in use since 2026-10 (auto-mounted by systemd), the
# KINGSTON needs the manual exfat-FUSE mount of start_orion.sh.
ORION_WEIGHTS_HOST=${ORION_WEIGHTS_HOST:-}
for d in /media/"$USER"/Intenso/Orin /media/"$USER"/KINGSTON/models/Orion; do
  [ -n "$ORION_WEIGHTS_HOST" ] && break
  [ -s "$d/Orion.pth" ] && ORION_WEIGHTS_HOST=$d
done
ZT_IFACE=${ZT_IFACE:-ztr4nscp5j}
NANO_IP=${NANO_IP:-10.241.27.5}
CAMERA_DEV=${CAMERA_DEV:-/dev/video0}
DOMAIN=${DOMAIN:-5}
CYCLONE_XML=${CYCLONE_XML:-/tmp/cyclone_f1tenth.xml}
if [ ! -r "$CYCLONE_XML" ]; then
  cat > "$CYCLONE_XML" <<XML
<CycloneDDS><Domain>
  <General><Interfaces><NetworkInterface name="$ZT_IFACE"/></Interfaces></General>
  <Discovery><Peers><Peer Address="$NANO_IP"/></Peers></Discovery>
</Domain></CycloneDDS>
XML
fi

ORION_REPO=/root/Orion
CKPT_CTR=/models/Orion/Orion.pth
CKPT_HOST="$ORION_WEIGHTS_HOST/Orion.pth"
CKPT_SIZE=38459589956          # HF poleyzdk/Orion, sha256 4977a050…
QFORMER_HOST="$ORION_WEIGHTS_HOST/pretrain_qformer"
INT8_STATS_REL=deps/orion_ros/engines/llm_act_stats.pt
INT8_STATS="$VLA_WS/$INT8_STATS_REL"
WS=/opt/orion_f1tenth_ws

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RST=$'\e[0m'
ok(){   echo "  ${GRN}ok${RST}   $*"; }
warn(){ echo "  ${YEL}warn${RST} $*"; }
die(){  echo "  ${RED}FAIL${RST} $*" >&2; exit 1; }
step(){ echo; echo "==> $*"; }

in_ctr(){ docker exec "$NAME" bash -c "$1"; }
ENVSETUP='source /opt/ros/humble/setup.bash;
          source '"$WS"'/install/setup.bash 2>/dev/null || true;
          export TORCHINDUCTOR_CACHE_DIR=/benchmarking/.torchinductor_cache TORCHINDUCTOR_FX_GRAPH_CACHE=1;'

MODE=launch
REBUILD=0
INFER=fast
LAUNCH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --stop|--check|--setup|--shell) MODE="${1#--}" ;;
    --rebuild)   REBUILD=1 ;;
    --fast|--int8|--lite|--eager|--baseline) INFER="${1#--}" ;;
    -h|--help)   sed -n '2,/^set -eo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *:=*)        LAUNCH_ARGS+=("$1") ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ "$MODE" = "stop" ]; then
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

# ── preflight ────────────────────────────────────────────────────────────────
step "Preflight"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found (docker pull it)"
ok "image $IMAGE"

if [ -z "$ORION_WEIGHTS_HOST" ]; then
  echo "  ${RED}FAIL${RST} no ORION weights found (Orion.pth + pretrain_qformer/)" >&2
  echo "       looked in /media/$USER/Intenso/Orin and /media/$USER/KINGSTON/models/Orion;" >&2
  echo "       plug in / mount the SSD, or set ORION_WEIGHTS_HOST=<dir>" >&2
  exit 1
fi
ok "weights at $ORION_WEIGHTS_HOST ($(df -h "$ORION_WEIGHTS_HOST" | awk 'NR==2{print $4}') free on $(df "$ORION_WEIGHTS_HOST" | awk 'NR==2{print $1}'))"

[ -s "$CKPT_HOST" ] || die "checkpoint missing: $CKPT_HOST"
sz=$(stat -c %s "$CKPT_HOST")
[ "$sz" -eq "$CKPT_SIZE" ] || warn "checkpoint is $sz bytes, expected $CKPT_SIZE (truncated copy?)"
[ "$sz" -eq "$CKPT_SIZE" ] && ok "checkpoint $(( sz/1024/1024/1024 )) GiB"

for f in config.json pytorch_model-00001-of-00002.bin \
         pytorch_model-00002-of-00002.bin tokenizer.model; do
  [ -s "$QFORMER_HOST/$f" ] || die "pretrain_qformer/$f missing — LLM will not load"
done
ok "pretrain_qformer (LLM + tokenizer)"

if [ "$INFER" = int8 ]; then
  [ -s "$INT8_STATS" ] || die "--int8 needs the SmoothQuant stats $INT8_STATS (tools/int8_sweep.py --phase 0 --save-stats)"
  ok "int8 calibration stats"
fi

[ -r "$CYCLONE_XML" ] || die "missing $CYCLONE_XML"
ok "cyclonedds config"

ip -4 addr show "$ZT_IFACE" 2>/dev/null | grep -q inet || die "iface $ZT_IFACE has no IPv4"
ok "iface $ZT_IFACE $(ip -4 -o addr show "$ZT_IFACE" | awk '{print $4}')"

if ping -c 1 -W 3 "$NANO_IP" >/dev/null 2>&1; then ok "car $NANO_IP reachable"
else warn "car $NANO_IP NOT reachable — nodes will wait for topics"; fi

CAM_ARGS=()
if [ -e "$CAMERA_DEV" ]; then ok "camera $CAMERA_DEV"; CAM_ARGS=(--device "$CAMERA_DEV")
else warn "no camera at $CAMERA_DEV — pass camera_source:=synthetic to test without one"; fi

# ── clocks ───────────────────────────────────────────────────────────────────
# jetson_clocks does not survive a reboot. It pins by setting min_freq =
# max_freq, so check the frequency, not the governor name.
step "Clocks"
# Orin: 17000000.gpu; other Jetsons (Thor) name the GPU devfreq node differently.
GPU_DEV=$(ls -d /sys/class/devfreq/*gpu* 2>/dev/null | head -1)
GPU_MIN=$GPU_DEV/min_freq
GPU_MAX=$GPU_DEV/max_freq
if [ -r "$GPU_MIN" ] && [ "$(cat "$GPU_MIN")" = "$(cat "$GPU_MAX")" ]; then
  ok "clocks already pinned (GPU $(( $(cat "$GPU_MAX") / 1000000 )) MHz)"
elif sudo -n jetson_clocks 2>/dev/null; then
  ok "jetson_clocks applied (GPU $(( $(cat "$GPU_MAX") / 1000000 )) MHz)"
else
  warn "clocks NOT pinned — inference will be substantially slower"
  warn "fix with:  sudo jetson_clocks"
fi

[ "$MODE" = "check" ] && { echo; echo "preflight only; not launching."; exit 0; }

# ── container ────────────────────────────────────────────────────────────────
step "Container"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  ok "$NAME already running"
else
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # /models is ro: the launch file's `ln -sfn /models/Orion /root/Orion/ckpts`
  # writes into the container layer, not the mount.
  docker run -d --runtime nvidia --network host --shm-size=8g \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e ROS_DOMAIN_ID="$DOMAIN" \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///cyclone.xml \
    -v "$CYCLONE_XML":/cyclone.xml:ro \
    -v "$BENCH/":/benchmarking \
    -v "$ORION_WEIGHTS_HOST":/models/Orion:ro \
    "${CAM_ARGS[@]}" \
    --name "$NAME" --entrypoint bash "$IMAGE" -c 'sleep infinity' >/dev/null
  sleep 2
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || die "container failed to start"
  ok "started $NAME"
fi

step "ORION repo (container)"
in_ctr "test -d $ORION_REPO" >/dev/null 2>&1 || die "$ORION_REPO missing from image"
in_ctr "cd $ORION_REPO && python3 -c 'import mmcv._ext'" >/dev/null 2>&1 \
  || die "$ORION_REPO has no compiled mmcv._ext — wrong image?"
ok "mmcv._ext compiled"
in_ctr "test -f $CKPT_CTR && test -d /models/Orion/pretrain_qformer" >/dev/null 2>&1 \
  || die "weights not visible in container at /models/Orion (SSD mount lost?)"
ok "weights visible at /models/Orion"

# ── ROS packages (host-built installs are not portable) ──────────────────────
step "ROS workspace ($WS)"
NEED_BUILD=1
# deps/ carries a COLCON_IGNORE (keeps the car's Foxy build out of it); naming each
# package directory as a base path bypasses it.
COLCON_PATHS="deps/autoware_msgs deps/ackermann_msgs deps/carla_msgs deps/simlingo_ros deps/orion_ros src/simlingo_f1tenth src/orion_f1tenth"
PKGS="src/orion_f1tenth src/simlingo_f1tenth deps/orion_ros"
if [ "$REBUILD" = 1 ]; then
  echo "  --rebuild: forcing"
  in_ctr "rm -rf $WS" >/dev/null 2>&1 || true
elif in_ctr "$ENVSETUP python3 -c 'import orion_f1tenth.orion_realworld_node, simlingo_f1tenth.trajectory_controller_node'" >/dev/null 2>&1; then
  # colcon copies (no --symlink-install): rebuild when any source is newer
  # than the installed package.
  NEWER=$(in_ctr "for p in $PKGS; do find $CTR_WS/\$p/\${p##*/} $CTR_WS/\$p/launch \
                  -newer $WS/install/orion_f1tenth/share/orion_f1tenth/package.xml -name '*.py' 2>/dev/null; done | head -1")
  if [ -n "$NEWER" ]; then echo "  node sources changed ($NEWER): rebuilding"
  else NEED_BUILD=0; ok "workspace already built"; fi
fi
if [ "$NEED_BUILD" = 1 ]; then
  echo "  building autoware_*, ackermann_msgs, carla_msgs, orion_ros, simlingo_ros, simlingo_f1tenth, orion_f1tenth (~60s)…"
  in_ctr "source /opt/ros/humble/setup.bash;
          cd $CTR_WS &&
          colcon build --base-paths $COLCON_PATHS \
            --packages-up-to orion_f1tenth \
            --build-base $WS/build --install-base $WS/install" >/dev/null 2>&1 \
    || die "colcon build failed — rerun with: docker exec $NAME bash -c 'cd $CTR_WS && colcon build --base-paths $COLCON_PATHS --packages-up-to orion_f1tenth …'"
  in_ctr "$ENVSETUP python3 -c 'import orion_f1tenth.orion_realworld_node, simlingo_f1tenth.trajectory_controller_node'" >/dev/null 2>&1 \
    || die "build finished but imports still fail"
  ok "workspace built"
fi

[ "$MODE" = "setup" ] && { echo; echo "setup only; not launching."; exit 0; }

if [ "$MODE" = "shell" ]; then
  step "Shell"; exec docker exec -it "$NAME" bash -c "$ENVSETUP cd $ORION_REPO; exec bash"
fi

# ── is the car there? ────────────────────────────────────────────────────────
step "F1TENTH side"
TOPICS=$(in_ctr "$ENVSETUP timeout 15 ros2 topic list 2>/dev/null" || true)
for t in /odom /pf/pose/odom /global_path; do
  if echo "$TOPICS" | grep -qx "$t"; then ok "$t visible"; else warn "$t not visible on domain $DOMAIN"; fi
done

# ── launch ───────────────────────────────────────────────────────────────────
step "Launch orion_f1tenth ($INFER; model load ~200 s, plus the compile warm-up; controller publishes zero speed until a plan exists)"
exec docker exec -i "$NAME" bash -c "$ENVSETUP
  exec ros2 launch orion_f1tenth orion_f1tenth.launch.py \
    inference_mode:=$INFER camera_device:=$CAMERA_DEV llm_int8_stats:=$CTR_WS/$INT8_STATS_REL ${LAUNCH_ARGS[*]}"
