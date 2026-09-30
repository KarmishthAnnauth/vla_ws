"""Complementary filter that removes the correction steps from the localisation pose.

slam_toolbox corrects the map -> odom transform every ~0.5 m / 0.5 s of travel, so the raw
map pose runs on wheel odometry and then snaps by up to tens of centimetres. This filter
propagates its output with the odometry motion (smooth, no lag) and pulls it towards the
raw map pose with time constant tau, so each correction is blended in instead of jumping.
Poses are (x, y, yaw) tuples.
"""
import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def compose(p, d):
    """Apply relative motion d (expressed in p's frame) to pose p."""
    c, s = math.cos(p[2]), math.sin(p[2])
    return (p[0] + c * d[0] - s * d[1], p[1] + s * d[0] + c * d[1], wrap(p[2] + d[2]))


def relative(a, b):
    """Motion from pose a to pose b, expressed in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


class PoseSmoother:
    def __init__(self, tau=0.3, reset_distance=1.5):
        self.tau = tau
        self.reset_distance = reset_distance
        self.out = None
        self.last_odom = None
        self.last_t = None

    def update(self, t, raw, odom):
        """t: time (s); raw: map pose from slam_toolbox; odom: pose of the same frame in odom."""
        if self.out is None or self.tau <= 0:
            self.out, self.last_odom, self.last_t = raw, odom, t
            return raw
        pred = compose(self.out, relative(self.last_odom, odom))
        err = (raw[0] - pred[0], raw[1] - pred[1], wrap(raw[2] - pred[2]))
        if math.hypot(err[0], err[1]) > self.reset_distance:
            # far off (e.g. relocalised or new start pose): jump instead of sliding across the map
            self.out = raw
        else:
            dt = max(0.0, t - self.last_t)
            alpha = 1.0 - math.exp(-dt / self.tau)
            self.out = (pred[0] + alpha * err[0], pred[1] + alpha * err[1], wrap(pred[2] + alpha * err[2]))
        self.last_odom, self.last_t = odom, t
        return self.out
