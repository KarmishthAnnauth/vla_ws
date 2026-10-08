#!/usr/bin/env python3
"""E-stop keeper for a SimLingo / ORION run (runs in the Orin container, fed by
run_simlingo_route.sh or run_orion_route.sh).

Publishes the current e-stop state on the topic given as the first argument
(default /simlingo/estop; the ORION launch uses /orion/estop) as std_msgs/Bool at 5 Hz and
switches it on lines read from stdin: ``go`` releases, ``stop`` engages, EOF exits.
Starts engaged, like the controller with start_estopped:=true.  Staying alive
between switches is the point: a fresh ``ros2 topic pub`` needs 1-2 s for start-up
and discovery before its message arrives.
"""

import sys
import threading

import rclpy
from std_msgs.msg import Bool


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("simlingo_estop_keeper")
    topic = sys.argv[1] if len(sys.argv) > 1 else "/simlingo/estop"
    pub = node.create_publisher(Bool, topic, 10)
    state = {"stop": True}

    def publish() -> None:
        pub.publish(Bool(data=state["stop"]))

    def read_stdin() -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("go", "stop"):
                state["stop"] = cmd == "stop"
                publish()
        rclpy.try_shutdown()

    node.create_timer(0.2, publish)
    threading.Thread(target=read_stdin, daemon=True).start()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
