#!/usr/bin/env python3
"""
teleop_replay.py
----------------
Replays movements recorded by teleop.py using movement.py's calibrated
go_forward() and turn() primitives.

Recorded values are in real units (metres / degrees from odom), so they
are passed directly to go_forward() and turn(). If the replay doesn't
match the original path, adjust the scaling factors below.

Usage:
    rosrun <your_package> teleop_replay.py
"""

import sys
import os
import importlib
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import movement

TOPIC           = "/B1/cmd_vel"
RECORDINGS_FILE = "recorded_movements"   # importlib will add .py

# ------------------------------------------------------------------ #
# Scaling factors                                                      #
# Increase above 1.0 if the robot falls short during replay.          #
# Decrease below 1.0 if the robot overshoots.                         #
# ------------------------------------------------------------------ #

DISTANCE_SCALE = 1.0   # applied to every go_forward() call  (metres)
ANGLE_SCALE    = 1.0   # applied to every turn() call        (degrees)

# Speeds used during replay (should match teleop.py)
FORWARD_SPEED  = 0.3   # m/s
TURN_SPEED     = 2.0   # rad/s

# Map direction label → sign multiplier for go_forward / turn
DIRECTION_SIGN = {
    "forward":  1,
    "backward": -1,
    "turn_ccw": 1,
    "turn_cw":  -1,
}


def replay(movements):
    for i, (direction, value) in enumerate(movements):
        if rospy.is_shutdown():
            break

        if direction not in DIRECTION_SIGN:
            rospy.logwarn("Unknown direction '%s', skipping.", direction)
            continue

        sign = DIRECTION_SIGN[direction]

        if direction in ("forward", "backward"):
            scaled = value * DISTANCE_SCALE
            rospy.loginfo("Segment %d/%d: %s  %.4f m  (scaled: %.4f m)",
                          i + 1, len(movements), direction, value, scaled)
            movement.go_forward(sign * scaled, speed=FORWARD_SPEED)

        else:
            scaled = value * ANGLE_SCALE
            rospy.loginfo("Segment %d/%d: %s  %.2f deg  (scaled: %.2f deg)",
                          i + 1, len(movements), direction, value, scaled)
            movement.turn(sign * scaled, speed=TURN_SPEED)


def main():
    rospy.init_node("teleop_replay", anonymous=False)
    movement.pub = rospy.Publisher(TOPIC, Twist, queue_size=1)
    rospy.sleep(1.0)

    # Load recorded movements
    try:
        rec = importlib.import_module(RECORDINGS_FILE)
        movements = rec.MOVEMENTS
    except ModuleNotFoundError:
        rospy.logerr("Could not find '%s.py'. Run teleop.py first.",
                     RECORDINGS_FILE)
        return
    except AttributeError:
        rospy.logerr("'%s.py' has no MOVEMENTS list.", RECORDINGS_FILE)
        return

    rospy.loginfo("Loaded %d segments from %s.py",
                  len(movements), RECORDINGS_FILE)
    rospy.loginfo("Scaling factors: distance=%.2f  angle=%.2f",
                  DISTANCE_SCALE, ANGLE_SCALE)

    for i, (d, v) in enumerate(movements):
        unit = "m" if d in ("forward", "backward") else "deg"
        rospy.loginfo("  [%d] %-12s  %.4f %s", i + 1, d, v, unit)

    rospy.loginfo("Starting replay in 2 seconds...")
    rospy.sleep(2.0)

    replay(movements)

    movement.stop()
    rospy.loginfo("Replay complete.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass