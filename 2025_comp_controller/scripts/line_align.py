#!/usr/bin/env python3
"""
line_align.py
-------------
Standalone visual servoing script that aligns the robot to a red line
visible in the top-down camera feed.

Steps:
  1. Coarse rotation  — rotate until the line is horizontal
  2. Translation      — center line horizontally and at target vertical position
  3. Fine rotation    — tighten residual angle with a smaller gain

Camera topic: /B1/robot/camera_top/image_raw

Usage:
    rosrun <your_package> line_align.py
"""

import math
import time
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

TOPIC      = "/B1/cmd_vel"
CAM_TOPIC  = "/B1/robot/camera_top/image_raw"
ALIGN_HZ   = 20    # Hz for visual servoing control loop

# ------------------------------------------------------------------ #
# Tunable parameters                                                   #
# ------------------------------------------------------------------ #

# Target vertical position of the line as a fraction of image height.
# 0.0 = top of image, 1.0 = bottom. 0.4 means 40% from top.
DEFAULT_TARGET_Y_RATIO = 0.4

# Proportional gains
GAIN_ANGLE       = 0.08   # angular.z per degree of line angle (steps 1 & 2)
GAIN_ANGLE_FINE  = 0.03   # smaller gain for step 3 fine rotation
GAIN_FORWARD     = 0.004  # linear speed per pixel of vertical error
GAIN_LATERAL     = 0.004  # linear speed per pixel of horizontal error

# Thresholds — considered done when all errors are below these
THRESH_ANGLE_DEG = 2.0    # degrees
THRESH_PIXEL_Y   = 10     # pixels
THRESH_PIXEL_X   = 15     # pixels

# Crop the top portion of the image to ignore distracting lines above.
# Set to 0.0 to disable, 0.2 to remove top 20%, etc.
CROP_TOP = 0.2

# Max speeds
MAX_ANGULAR = 1.0   # rad/s
MAX_LINEAR  = 0.3   # m/s

bridge       = CvBridge()
latest_image = None


# ------------------------------------------------------------------ #
# Camera                                                               #
# ------------------------------------------------------------------ #

def image_callback(msg):
    global latest_image
    img     = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if CROP_TOP > 0.0:
        h       = rotated.shape[0]
        rotated = rotated[int(h * CROP_TOP):, :]
    latest_image = rotated


def get_image():
    """Block until a fresh image is available (up to 3 s)."""
    global latest_image
    latest_image = None
    deadline = time.time() + 3.0
    while latest_image is None and time.time() < deadline:
        rospy.sleep(0.05)
    return latest_image


# ------------------------------------------------------------------ #
# Red line detection                                                   #
# ------------------------------------------------------------------ #

def detect_red_line(cv_image):
    """
    Detect the red line in cv_image.

    Returns (angle_deg, cx, cy, img_w, img_h) or None if not found.
      angle_deg : angle of line in degrees, [-90, 90], 0 = horizontal
      cx, cy    : centroid of red pixels in image coordinates
    """
    hsv   = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,   120, 100]), np.array([10,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_h, img_w = cv_image.shape[:2]

    M = cv2.moments(mask)
    if M["m00"] < 100:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    lines = cv2.HoughLinesP(mask, 1, np.pi / 180,
                            threshold=40, minLineLength=30, maxLineGap=20)
    if lines is None or len(lines) == 0:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while a >  90: a -= 180
        while a < -90: a += 180
        angles.append(a)

    return float(np.median(angles)), cx, cy, img_w, img_h


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def body_to_world(vx_body, vy_body, heading_deg):
    """Rotate body-frame velocities to world frame for planar_move."""
    h        = math.radians(heading_deg)
    vx_world = vx_body * math.cos(h) - vy_body * math.sin(h)
    vy_world = vx_body * math.sin(h) + vy_body * math.cos(h)
    return vx_world, vy_world


# ------------------------------------------------------------------ #
# Alignment                                                            #
# ------------------------------------------------------------------ #

def align_to_red_line(pub, pose=None,
                      target_y_ratio=DEFAULT_TARGET_Y_RATIO,
                      timeout=30.0):
    """
    Visually servo the robot so the red line is:
      - Horizontal           (step 1: coarse rotation)
      - Correctly positioned (step 2: translation)
      - Fine angle corrected (step 3: fine rotation)

    The robot stops completely between each control pulse to avoid
    overshoot — planar_move holds the last commanded velocity
    indefinitely, so an explicit zero Twist is sent after each pulse.

    Args:
        pub           : rospy.Publisher for /B1/cmd_vel
        pose          : object with .heading_deg attribute for world-frame
                        velocity rotation. If None, heading 0 is assumed.
        target_y_ratio: vertical target position (0.0=top, 1.0=bottom).
        timeout       : total time budget in seconds across all steps.
    """
    rospy.loginfo("align_to_red_line: starting (target_y=%.0f%% from top)",
                  target_y_ratio * 100)

    # ALIGN_HZ controls pulse duration via rospy.sleep(1.0 / ALIGN_HZ)
    deadline    = rospy.Time.now() + rospy.Duration(timeout)
    heading_deg = pose.heading_deg if pose is not None else 0.0

    def publish_stop():
        pub.publish(Twist())

    # ---- Step 1: Coarse rotation ---- #
    rospy.loginfo("Step 1: coarse rotation to make line horizontal...")
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        img = get_image()
        if img is None:
            rospy.logwarn("No image, retrying...")
            continue

        result = detect_red_line(img)
        if result is None:
            rospy.logwarn("Red line not detected, waiting...")
            rospy.sleep(1.0 / ALIGN_HZ)
            continue

        angle_deg, cx, cy, img_w, img_h = result
        rospy.loginfo("  angle=%.1f deg", angle_deg)

        if abs(angle_deg) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 1 done.")
            break

        cmd = Twist()
        cmd.angular.z = -float(np.clip(GAIN_ANGLE * angle_deg,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        pub.publish(cmd)
        rospy.sleep(1.0 / ALIGN_HZ)  # fixed duration — rate.sleep() returns immediately if loop is behind
        publish_stop()  # stop between pulses to prevent overshoot

    publish_stop()
    rospy.sleep(0.3)

    # ---- Step 2: Translation ---- #
    rospy.loginfo("Step 2: translating to position line...")
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        img = get_image()
        if img is None:
            rospy.sleep(1.0 / ALIGN_HZ)
            continue

        result = detect_red_line(img)
        if result is None:
            rospy.logwarn("Red line lost during step 2, stopping.")
            break

        angle_deg, cx, cy, img_w, img_h = result
        target_y = int(target_y_ratio * img_h)
        target_x = img_w // 2
        y_error  = cy - target_y
        x_error  = cx - target_x

        rospy.loginfo("  cx=%d cy=%d  x_err=%d  y_err=%d  angle=%.1f",
                      cx, cy, x_error, y_error, angle_deg)

        if (abs(y_error) < THRESH_PIXEL_Y and
                abs(x_error) < THRESH_PIXEL_X and
                abs(angle_deg) < THRESH_ANGLE_DEG):
            rospy.loginfo("  Step 2 done.")
            break

        vx_body   = float(np.clip(-GAIN_FORWARD * y_error, -MAX_LINEAR, MAX_LINEAR))
        vy_body   = float(np.clip(-GAIN_LATERAL * x_error, -MAX_LINEAR, MAX_LINEAR))
        angular_z = float(np.clip(-GAIN_ANGLE   * angle_deg, -MAX_ANGULAR, MAX_ANGULAR))

        vx_world, vy_world = body_to_world(vx_body, vy_body, heading_deg)

        cmd = Twist()
        cmd.linear.x  = vx_world
        cmd.linear.y  = vy_world
        cmd.angular.z = angular_z
        pub.publish(cmd)
        rospy.sleep(1.0 / ALIGN_HZ)  # fixed duration — rate.sleep() returns immediately if loop is behind
        publish_stop()  # stop between pulses to prevent overshoot

    publish_stop()
    rospy.sleep(0.3)

    # ---- Step 3: Fine rotation ---- #
    rospy.loginfo("Step 3: fine rotation...")
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        img = get_image()
        if img is None:
            rospy.sleep(1.0 / ALIGN_HZ)
            continue

        result = detect_red_line(img)
        if result is None:
            rospy.logwarn("Red line lost during step 3, stopping.")
            break

        angle_deg, cx, cy, img_w, img_h = result
        rospy.loginfo("  angle=%.1f deg (fine)", angle_deg)

        if abs(angle_deg) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 3 done.")
            break

        cmd = Twist()
        cmd.angular.z = -float(np.clip(GAIN_ANGLE_FINE * angle_deg,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        pub.publish(cmd)
        rospy.sleep(1.0 / ALIGN_HZ)  # fixed duration — rate.sleep() returns immediately if loop is behind
        publish_stop()  # stop between pulses to prevent overshoot

    publish_stop()
    rospy.loginfo("align_to_red_line: complete.")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    rospy.init_node("line_align", anonymous=True)
    pub = rospy.Publisher(TOPIC, Twist, queue_size=1)
    rospy.Subscriber(CAM_TOPIC, Image, image_callback)
    rospy.sleep(1.0)

    align_to_red_line(pub, pose=None,
                      target_y_ratio=DEFAULT_TARGET_Y_RATIO)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass