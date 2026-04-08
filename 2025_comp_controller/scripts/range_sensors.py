#!/usr/bin/env python3
"""
range_sensors.py
----------------
Helper functions for the three front range sensors.

Topics:
  /B1/range_front_center  — points forward
  /B1/range_front_left    — points 90° left
  /B1/range_front_right   — points 90° right

Usage:
    import range_sensors as rs
    rs.init()

    dist = rs.read("center")
    rs.wait_until("left", "below", 0.5)
"""

import rospy
from sensor_msgs.msg import Range

# ------------------------------------------------------------------ #
# Sensor topics                                                        #
# ------------------------------------------------------------------ #

TOPICS = {
    "center": "/B1/range_front_center",
    "left":   "/B1/range_front_left",
    "right":  "/B1/range_front_right",
}

_readings = {
    "center": None,
    "left":   None,
    "right":  None,
}

_subs    = {}
_inited  = False


# ------------------------------------------------------------------ #
# Initialisation                                                       #
# ------------------------------------------------------------------ #

def init():
    """Subscribe to all three range sensor topics. Call once after rospy.init_node."""
    global _inited
    if _inited:
        return
    for name, topic in TOPICS.items():
        _subs[name] = rospy.Subscriber(
            topic, Range, _make_callback(name))
    _inited = True
    rospy.loginfo("range_sensors: subscribed to %s", list(TOPICS.values()))


def _make_callback(name):
    def cb(msg):
        _readings[name] = msg.range
    return cb


def _check_sensor(sensor):
    if sensor not in TOPICS:
        raise ValueError(f"Unknown sensor '{sensor}'. Choose from: {list(TOPICS.keys())}")
    if not _inited:
        raise RuntimeError("Call range_sensors.init() before using sensor functions.")


# ------------------------------------------------------------------ #
# Read                                                                 #
# ------------------------------------------------------------------ #

def read(sensor):
    """
    Return the latest distance reading from the sensor (metres).
    Returns None if no reading has been received yet.

    Args:
        sensor : "center", "left", or "right"
    """
    _check_sensor(sensor)
    return _readings[sensor]


def read_all():
    """Return a dict of all current readings: {center, left, right}."""
    if not _inited:
        raise RuntimeError("Call range_sensors.init() first.")
    return dict(_readings)


# ------------------------------------------------------------------ #
# Blocking wait                                                        #
# ------------------------------------------------------------------ #

def wait_until(sensor, condition, value, timeout=10.0):
    """
    Block until the sensor reading satisfies the condition, or timeout.

    Args:
        sensor    : "center", "left", or "right"
        condition : "above" or "below"
        value     : threshold distance in metres
        timeout   : seconds before giving up (default 10)

    Returns:
        True  if condition was met before timeout
        False if timed out

    Example:
        rs.wait_until("center", "below", 0.5)   # wait until closer than 0.5m
        rs.wait_until("left",   "above", 1.0)   # wait until further than 1.0m
    """
    _check_sensor(sensor)
    if condition not in ("above", "below"):
        raise ValueError(f"condition must be 'above' or 'below', got '{condition}'")

    rospy.loginfo("range_sensors: waiting until %s is %s %.3f m (timeout=%.1fs)",
                  sensor, condition, value, timeout)

    deadline = rospy.Time.now() + rospy.Duration(timeout)
    rate     = rospy.Rate(20)

    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        dist = _readings[sensor]
        if dist is not None:
            met = dist < value if condition == "below" else dist > value
            rospy.loginfo("  %s: %.3f m", sensor, dist)
            if met:
                rospy.loginfo("  Condition met.")
                return True
        rate.sleep()

    rospy.logwarn("range_sensors: wait_until timed out (%s %s %.3f)",
                  sensor, condition, value)
    return False


# ------------------------------------------------------------------ #
# Standalone test                                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    rospy.init_node("range_sensors_test", anonymous=True)
    init()
    rospy.sleep(0.5)

    rate = rospy.Rate(2)
    rospy.loginfo("Reading sensors. Ctrl+C to stop.")
    while not rospy.is_shutdown():
        all_readings = read_all()
        rospy.loginfo("center=%-8s  left=%-8s  right=%-8s",
                      f"{all_readings['center']:.3f}m" if all_readings['center'] is not None else "—",
                      f"{all_readings['left']:.3f}m"   if all_readings['left']   is not None else "—",
                      f"{all_readings['right']:.3f}m"  if all_readings['right']  is not None else "—")
        rate.sleep()