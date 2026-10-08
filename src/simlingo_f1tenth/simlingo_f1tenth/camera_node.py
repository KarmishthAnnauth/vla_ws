#!/usr/bin/env python3
"""Front camera publisher for the Orin (OpenCV capture, no image_transport needed).

Publishes sensor_msgs/CompressedImage (JPEG) and optionally raw sensor_msgs/Image.
JPEG is the default for the same reason the CARLA path uses it: SimLingo was
trained on JPEG frames, so the codec artefacts belong in the input, and the
inference node then skips its own JPEG round-trip.

Parameters
  source          "v4l2" (default) | "gstreamer" | "synthetic"
  device          V4L2 device: index ("0") or path ("/dev/video0")
  gst_pipeline    GStreamer pipeline string ending in appsink (source=gstreamer),
                  e.g. a CSI camera through nvarguscamerasrc.  Needs an OpenCV
                  built with GStreamer -- the node fails loudly if it is not.
  width, height, fps, fourcc     capture settings requested from the driver
  flip_code       -2 = none, else cv2.flip code (0 vertical, 1 horizontal, -1 both)
  topic           base topic; JPEG goes to <topic>/compressed, raw to <topic>
  publish_raw     also publish the raw Image (bgr8)
  jpeg_quality    JPEG quality for the compressed topic
  frame_id        header.frame_id

"synthetic" publishes a moving test pattern so the rest of the pipeline can be
exercised without a camera attached.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class CameraNode(Node):

    def __init__(self) -> None:
        super().__init__("camera_node")
        self.declare_parameter("source", "v4l2")
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("gst_pipeline", "")
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("fourcc", "MJPG")
        self.declare_parameter("flip_code", -2)
        self.declare_parameter("topic", "/camera/front/image")
        self.declare_parameter("publish_raw", False)
        self.declare_parameter("jpeg_quality", 95)   # cv2 default, as agent_simlingo.py::tick re-encodes
        self.declare_parameter("frame_id", "camera_front")

        self._source = str(self.get_parameter("source").value).lower()
        self._flip = int(self.get_parameter("flip_code").value)
        self._quality = int(self.get_parameter("jpeg_quality").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._publish_raw = bool(self.get_parameter("publish_raw").value)
        self._fps = float(self.get_parameter("fps").value)
        self._w = int(self.get_parameter("width").value)
        self._h = int(self.get_parameter("height").value)
        topic = str(self.get_parameter("topic").value)

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=2)
        self._jpeg_pub = self.create_publisher(CompressedImage, topic + "/compressed", qos)
        self._raw_pub = self.create_publisher(Image, topic, qos) if self._publish_raw else None

        self._cap = None
        if self._source != "synthetic":
            self._cap = self._open_capture()

        self._count = 0
        self._t_last_log = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.get_logger().info(f"camera source={self._source} -> {topic}/compressed"
                               + (f" and {topic}" if self._publish_raw else ""))

    def _open_capture(self) -> cv2.VideoCapture:
        if self._source == "gstreamer":
            pipeline = str(self.get_parameter("gst_pipeline").value)
            if not pipeline:
                raise ValueError("source=gstreamer needs a gst_pipeline parameter")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError(
                    "could not open GStreamer pipeline (is this OpenCV built with GStreamer? "
                    "cv2.getBuildInformation() must list GStreamer: YES)")
            return cap
        dev = str(self.get_parameter("device").value)
        handle = int(dev) if dev.isdigit() else dev
        cap = cv2.VideoCapture(handle, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"could not open V4L2 device {dev}")
        fourcc = str(self.get_parameter("fourcc").value)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always the newest frame
        self.get_logger().info(
            f"opened {dev}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
        return cap

    def _synthetic_frame(self) -> np.ndarray:
        t = time.time()
        img = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, self._w, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(0, 255, self._h, dtype=np.uint8)[:, None]
        cx = int((0.5 + 0.4 * np.sin(t)) * self._w)
        cv2.circle(img, (cx, self._h // 2), self._h // 8, (255, 255, 255), -1)
        cv2.putText(img, f"synthetic {t:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img

    def _loop(self) -> None:
        period = 1.0 / max(self._fps, 1e-3)
        while not self._stop.is_set() and rclpy.ok():
            t0 = time.time()
            if self._cap is None:
                bgr = self._synthetic_frame()
            else:
                ok, bgr = self._cap.read()
                if not ok or bgr is None:
                    self.get_logger().warn("camera read failed", throttle_duration_sec=2.0)
                    time.sleep(0.05)
                    continue
            if self._flip != -2:
                bgr = cv2.flip(bgr, self._flip)
            self._publish(bgr)
            if self._cap is None:                        # synthetic: pace it
                time.sleep(max(0.0, period - (time.time() - t0)))

    def _publish(self, bgr: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = self._frame_id
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            self._jpeg_pub.publish(msg)
        if self._raw_pub is not None:
            raw = Image()
            raw.header.stamp = stamp
            raw.header.frame_id = self._frame_id
            raw.height, raw.width = bgr.shape[:2]
            raw.encoding = "bgr8"
            raw.is_bigendian = 0
            raw.step = bgr.shape[1] * 3
            raw.data = np.ascontiguousarray(bgr).tobytes()
            self._raw_pub.publish(raw)
        self._count += 1
        now = time.time()
        if now - self._t_last_log >= 5.0:
            self.get_logger().info(f"published {self._count} frames "
                                   f"({self._count / (now - self._t_last_log):.1f} Hz)")
            self._count = 0
            self._t_last_log = now

    def destroy_node(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
