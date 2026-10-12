#!/usr/bin/env python3
"""Verify each ORION camera slot receives ITS OWN image, not a replicated one."""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage

from orion_ros.camera_input import (
    MultiCameraBuffer, ORION_CAMERA_ORDER, REFERENCE_CAMERA, decode_image,
)

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST, depth=5)
H, W = 48, 64
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def make_img(value: int, stamp_sec: int) -> RosImage:
    m = RosImage()
    m.header.stamp.sec = stamp_sec
    m.height, m.width = H, W
    m.encoding = "bgr8"
    m.step = W * 3
    m.data = (np.full((H, W, 3), value, dtype=np.uint8)).tobytes()
    return m


def spin(node, n=40):
    for _ in range(n):
        rclpy.spin_once(node, timeout_sec=0.05)


def main():
    rclpy.init()
    node = Node("cam_test")
    topics = [f"/test/{c}/image" for c in ORION_CAMERA_ORDER]
    buf = MultiCameraBuffer(node, topics, QOS, sync_tolerance_sec=0.1)
    pubs = [node.create_publisher(RosImage, t, QOS) for t in topics]

    check(not buf.replicate, "6 topics -> surround mode (not replicate)")
    check(len(buf.missing()) == 6, "all 6 cameras missing before any frame")
    check(buf.snapshot() is None, "snapshot withheld until every camera has a frame")

    # Publish 5 of 6: must still refuse, otherwise a dead camera would silently
    # be filled by whatever happened to be in the dict.
    for i in range(5):
        pubs[i].publish(make_img(10 + i * 20, 100))
    spin(node)
    check(buf.snapshot() is None, "snapshot still withheld with 5/6 cameras")
    check(buf.missing() == ["CAM_BACK_RIGHT"], "missing() names the absent camera")

    pubs[5].publish(make_img(10 + 5 * 20, 100))
    spin(node)
    snap = buf.snapshot()
    check(snap is not None, "snapshot available once all 6 arrived")

    # The point of the whole change: distinct pixels per slot.
    vals = {}
    for i, cam in enumerate(ORION_CAMERA_ORDER):
        img = decode_image(snap[cam], jpeg_quality=0)
        vals[cam] = int(np.median(img))
        check(vals[cam] == 10 + i * 20,
              f"{cam} holds its own image (got {vals[cam]}, want {10 + i * 20})")
    check(len(set(vals.values())) == 6, "all six slots differ (no replication)")

    # Skew detection: a stale rear camera must be reported, not hidden.
    before = buf.skew_count
    pubs[0].publish(make_img(200, 105))          # front jumps 5 s ahead
    spin(node)
    buf.snapshot()
    check(buf.skew_count > before, "skew against the reference camera is detected")

    # Replicate mode still works for bring-up.
    node2 = Node("cam_test_single")
    buf2 = MultiCameraBuffer(node2, ["/test/solo/image"], QOS, 0.1)
    pub2 = node2.create_publisher(RosImage, "/test/solo/image", QOS)
    check(buf2.replicate, "1 topic -> replicate mode")
    pub2.publish(make_img(77, 100))
    spin(node2)
    snap2 = buf2.snapshot()
    check(snap2 is not None and len(snap2) == 6, "replicate mode fills all 6 slots")
    check(all(snap2[c] is snap2[REFERENCE_CAMERA] for c in ORION_CAMERA_ORDER),
          "replicate mode shares one record (single decode)")

    # A wrong topic count must fail loudly rather than half-configure.
    try:
        MultiCameraBuffer(Node("cam_test_bad"), ["/a", "/b", "/c"], QOS, 0.1)
        check(False, "3 topics rejected")
    except ValueError:
        check(True, "3 topics rejected with ValueError")

    node.destroy_node()
    rclpy.shutdown()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL CAMERA CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
