#!/usr/bin/env bash
#
# simlingo_car.sh — car side of a SimLingo run: bringup + localisation, visible to the Orin.
#
# Runs on the car (Jetson, ROS 2 Foxy, ~/vla_ws). Normally called over ssh by
# scripts/run_simlingo_route.sh on the Orin, but works by hand too:
#
#   simlingo_car.sh start <orin_ip> [--keep-localization]   bringup (if not up) + fresh localisation
#   simlingo_car.sh stop                                    stop both
#   simlingo_car.sh status                                  what runs, which topics are alive
#   source simlingo_car.sh env                              same ROS/DDS environment in this shell
#
# Why not just the two `ros2 launch` commands: ~/.bashrc has ROS_LOCALHOST_ONLY=1
# and no exported ROS_DOMAIN_ID, and the car and the Orin sit in different
# subnets of the Wi-Fi, so DDS multicast discovery never reaches the other side.
# Everything started here uses domain 5 and a Fast DDS profile that
#   * announces to the Orin by unicast (initial peer),
#   * uses only the Wi-Fi address and loopback (otherwise the Orin is handed the
#     LiDAR network address 192.168.0.15 first and sends its replies there),
#   * uses UDP only (no shared memory: stale lock files, see the runbook).
# ~/.bashrc and ~/f1tenth_ws are not touched.
#
# The localisation is restarted on every `start`: it fits the start pose from
# the scan, so the car must stand in the start box. --keep-localization skips
# that when the car has not been moved by hand since.

WS="$HOME/vla_ws"
LOGDIR="$WS/log/simlingo"
PROFILE="$LOGDIR/fastdds_orin_peer.xml"
DOMAIN=5
LIDAR_IP=192.168.0.10

car_env() {
  source /opt/ros/foxy/setup.bash
  source "$WS/install/setup.bash"
  export ROS_DOMAIN_ID=$DOMAIN
  export ROS_LOCALHOST_ONLY=0
  export FASTRTPS_DEFAULT_PROFILES_FILE="$PROFILE"
}

if [ "${1:-}" = "env" ]; then
  car_env
  echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=0 FASTRTPS_DEFAULT_PROFILES_FILE=$PROFILE"
  return 0 2>/dev/null || exit 0
fi

set -o pipefail
RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RST=$'\e[0m'
ok(){   echo "  ${GRN}ok${RST}   $*"; }
warn(){ echo "  ${YEL}warn${RST} $*"; }
die(){  echo "  ${RED}FAIL${RST} $*" >&2; exit 1; }

# pid of a launch started by this script, if it is still that launch
launch_pid() {            # $1 = bringup | localization
  local f="$LOGDIR/$1.pid" pid
  [ -r "$f" ] || return 1
  pid=$(cat "$f")
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && grep -qa "ros2" "/proc/$pid/cmdline" 2>/dev/null \
    && echo "$pid"
}

# a launch of that file that this script did not start (another terminal, other DDS settings)
foreign_launch() {        # $1 = launch file name, $2 = bringup | localization
  local mine; mine=$(launch_pid "$2" || true)
  pgrep -f "ros2 launch .*$1" | grep -vx "${mine:-0}" | head -1
}

stop_launch() {           # $1 = bringup | localization
  local pid; pid=$(launch_pid "$1") || { rm -f "$LOGDIR/$1.pid"; return 0; }
  kill -INT "$pid" 2>/dev/null
  for _ in $(seq 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
  # the launch ran under setsid: its pid is the process group
  kill -KILL -- "-$pid" 2>/dev/null
  rm -f "$LOGDIR/$1.pid"
  echo "  stopped $1"
}

start_launch() {          # $1 = bringup | localization, rest = ros2 launch arguments
  local name=$1; shift
  setsid nohup ros2 launch "$@" > "$LOGDIR/$name.log" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/$name.pid"
}

# Which of the given topics deliver a message within the timeout. No ros2 CLI:
# its daemon may have been started with other DDS settings.
alive_topics() {          # $1 = timeout in s, rest = topics
  python3 - "$@" <<'PY'
import sys, time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
timeout, topics = float(sys.argv[1]), sys.argv[2:]
rclpy.init()
node = rclpy.create_node('simlingo_car_check')
got = {}
for t in topics:
    typ = LaserScan if t == '/scan' else Odometry
    node.create_subscription(typ, t, lambda m, t=t: got.setdefault(t, m), qos_profile_sensor_data)
end = time.time() + timeout
while time.time() < end and len(got) < len(topics):
    rclpy.spin_once(node, timeout_sec=0.2)
for t in topics:
    m = got.get(t)
    if m is None:
        print(f'{t} DEAD')
    elif t == '/scan':
        print(f'{t} ALIVE {len(m.ranges)} ranges')
    else:
        p, v = m.pose.pose.position, m.twist.twist.linear.x
        print(f'{t} ALIVE x={p.x:.2f} y={p.y:.2f} v={v:.2f}')
node.destroy_node()
rclpy.shutdown()
PY
}

# Zero-speed, zero-steering commands on /drive for $1 seconds.
zero_drive() {
  python3 - "$1" <<'PY'
import sys, time
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
rclpy.init()
node = rclpy.create_node('simlingo_car_zero_drive')
pub = node.create_publisher(AckermannDriveStamped, '/drive', 10)
end = time.time() + float(sys.argv[1]) + 1.0      # 1 s for discovery
while time.time() < end:
    msg = AckermannDriveStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)
    time.sleep(0.05)
node.destroy_node()
rclpy.shutdown()
PY
}

write_profile() {         # $1 = orin ip, $2 = car ip; returns 0 if the file changed
  local new
  new=$(cat <<XML
<?xml version="1.0" encoding="UTF-8" ?>
<!-- Generated by scripts/simlingo_car.sh. Orin $1, car $2. -->
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udp_wifi</transport_id>
      <type>UDPv4</type>
      <maxInitialPeersRange>40</maxInitialPeersRange>
      <interfaceWhiteList><address>$2</address><address>127.0.0.1</address></interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="simlingo_orin_peer" is_default_profile="true">
    <rtps>
      <userTransports><transport_id>udp_wifi</transport_id></userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
      <builtin>
        <initialPeersList>
          <locator><udpv4><address>239.255.0.1</address></udpv4></locator>
          <locator><udpv4><address>$1</address></udpv4></locator>
        </initialPeersList>
      </builtin>
    </rtps>
  </participant>
</profiles>
XML
)
  if [ -r "$PROFILE" ] && [ "$new" = "$(cat "$PROFILE")" ]; then return 1; fi
  echo "$new" > "$PROFILE"
}

cmd=${1:-status}
mkdir -p "$LOGDIR"

case "$cmd" in
  start)
    ORIN_IP=${2:-}
    [ -n "$ORIN_IP" ] || die "usage: simlingo_car.sh start <orin_ip> [--keep-localization]"
    KEEP_LOC=0; [ "${3:-}" = "--keep-localization" ] && KEEP_LOC=1
    CAR_IP=$(ip -4 route get "$ORIN_IP" 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')
    [ -n "$CAR_IP" ] || die "no route to the Orin at $ORIN_IP"

    echo "==> Car preflight"
    [ -e /dev/sensors/vesc ] && ok "VESC /dev/sensors/vesc" || die "no VESC at /dev/sensors/vesc (car powered? USB?)"
    if ping -c 1 -W 1 "$LIDAR_IP" >/dev/null 2>&1; then ok "LiDAR $LIDAR_IP"
    else die "LiDAR $LIDAR_IP not reachable: run 'sudo nmcli connection up Hokuyo' on the car"; fi
    if ls /dev/input/js* >/dev/null 2>&1; then ok "joystick $(ls /dev/input/js* | head -1)"
    else warn "NO JOYSTICK connected: no manual override (L1) while SimLingo drives"; fi
    for pair in "bringup_launch.py bringup" "localize_launch.py localization"; do
      set -- $pair
      other=$(foreign_launch "$1" "$2")
      [ -z "$other" ] || die "$1 is already running from another terminal (pid $other), with DDS settings the Orin cannot see. Stop it there first."
    done
    pgrep -f "pure_pursuit_node|safety_node" >/dev/null && die "pure_pursuit_node or safety_node is running: both publish /drive. Stop them first."

    car_env
    if write_profile "$ORIN_IP" "$CAR_IP"; then
      ok "DDS profile written (peer $ORIN_IP, interface $CAR_IP)"
      # running nodes still hold the old peer/interface
      stop_launch localization; stop_launch bringup
    else
      ok "DDS profile unchanged (peer $ORIN_IP, interface $CAR_IP)"
    fi

    echo "==> Bringup"
    if launch_pid bringup >/dev/null; then
      ok "already running (pid $(launch_pid bringup))"
    else
      start_launch bringup f1tenth_stack bringup_launch.py
      echo "  started, waiting for /scan and /odom ..."
    fi
    res=$(alive_topics 20 /scan)
    echo "$res" | sed 's/^/  /'
    if echo "$res" | grep -q DEAD; then
      echo "  --- tail of $LOGDIR/bringup.log"; tail -15 "$LOGDIR/bringup.log" | sed 's/^/  /'
      die "bringup is up but /scan is silent"
    fi
    res=$(alive_topics 3 /odom)
    if echo "$res" | grep -q DEAD; then
      # vesc_to_odom publishes nothing until it has seen a servo command
      echo "  /odom silent: sending a zero /drive command (speed 0, steering 0) to start the odometry"
      zero_drive 1.5
      res=$(alive_topics 10 /odom)
    fi
    echo "$res" | sed 's/^/  /'
    if echo "$res" | grep -q DEAD; then
      echo "  --- tail of $LOGDIR/bringup.log"; tail -15 "$LOGDIR/bringup.log" | sed 's/^/  /'
      die "bringup is up but /odom is silent"
    fi

    echo "==> Localisation (slam_localization)"
    if [ "$KEEP_LOC" = 1 ] && launch_pid localization >/dev/null; then
      ok "kept running (pid $(launch_pid localization))"
    else
      stop_launch localization
      # udp_only:=false: the launch would otherwise replace the profile above with its own
      start_launch localization slam_localization localize_launch.py udp_only:=false
      echo "  started, fitting the start pose from the scan (car must be in the start box) ..."
      for _ in $(seq 100); do
        grep -q "fit_start_pose" "$LOGDIR/localization.log" 2>/dev/null && break
        launch_pid localization >/dev/null || break
        sleep 1
      done
      fit=$(grep -m1 "fit_start_pose" "$LOGDIR/localization.log" 2>/dev/null)
      if [ -z "$fit" ] || echo "$fit" | grep -q "failed\|timed out"; then
        echo "  --- tail of $LOGDIR/localization.log"; tail -15 "$LOGDIR/localization.log" | sed 's/^/  /'
        die "start-pose fit failed"
      fi
      echo "  ${fit#*fit_start_pose: }"
      pct=$(echo "$fit" | sed -n 's/.* \([0-9]*\)% of scan.*/\1/p')
      if [ -n "$pct" ] && [ "$pct" -lt 70 ]; then
        stop_launch localization
        die "only $pct% of the scan matches the map. Put the car in the start box (nobody next to it) and rerun."
      fi
    fi
    res=$(alive_topics 30 /pf/pose/odom)
    echo "$res" | sed 's/^/  /'
    if echo "$res" | grep -q DEAD; then
      echo "  --- tail of $LOGDIR/localization.log"; tail -15 "$LOGDIR/localization.log" | sed 's/^/  /'
      die "no pose on /pf/pose/odom"
    fi
    ok "car side ready (logs: $LOGDIR)"
    ;;

  stop)
    stop_launch localization
    stop_launch bringup
    echo "car side stopped"
    ;;

  status)
    for n in bringup localization; do
      if pid=$(launch_pid $n); then ok "$n running (pid $pid)"; else echo "  --   $n not running (not started by this script)"; fi
    done
    for pair in "bringup_launch.py bringup" "localize_launch.py localization"; do
      set -- $pair
      other=$(foreign_launch "$1" "$2"); [ -z "$other" ] || warn "$1 also running from another terminal (pid $other)"
    done
    if [ -r "$PROFILE" ]; then
      car_env
      alive_topics 5 /scan /odom /pf/pose/odom | sed 's/^/  /'
    fi
    ;;

  *)
    echo "usage: simlingo_car.sh start <orin_ip> [--keep-localization] | stop | status" >&2
    exit 2 ;;
esac
