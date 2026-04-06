#!/usr/bin/env python3
"""
movement.py
-----------
All motion and alignment primitives.

  go_forward(distance, speed)     calibrated forward/back
  turn(angle_deg, speed)          calibrated rotation
  align_to_line(pub, color, ...)  visual servo to any colored line
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
PUBLISH_HZ = 10   # for go_forward / turn
ALIGN_HZ   = 20   # for visual servoing

pub = None

# ================================================================== #
# LINE ALIGNMENT DEFAULTS (override per call)                         #
# ================================================================== #

DEFAULT_TARGET_Y_RATIO = 0.4
DEFAULT_CROP_TOP       = 0.2   # fraction to remove from top of image
DEFAULT_CROP_BOTTOM    = 0.0   # fraction to remove from bottom of image

GAIN_ANGLE      = 0.08
GAIN_ANGLE_FINE = 0.03

# Sign alignment uses higher gains (sign edge is a crisper feature)
SIGN_GAIN_ANGLE      = 0.2
SIGN_GAIN_ANGLE_FINE = 0.1
SIGN_SEARCH_SPEED    = 0.05   # m/s strafe when sign not detected
GAIN_FORWARD    = 0.004
GAIN_LATERAL    = 0.004

THRESH_ANGLE_DEG = 2.0
THRESH_PIXEL_Y   = 10
THRESH_PIXEL_X   = 15

MAX_ANGULAR = 1.0
MAX_LINEAR  = 0.3

bridge       = CvBridge()
latest_image = None
_cam_sub     = None


# ================================================================== #
# CAMERA                                                              #
# ================================================================== #

def image_callback(msg):
    global latest_image
    img          = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    latest_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def _ensure_camera():
    global _cam_sub
    if _cam_sub is None:
        _cam_sub = rospy.Subscriber(CAM_TOPIC, Image, image_callback)
        rospy.sleep(0.5)


def get_image(crop_top=0.0, crop_bottom=0.0, crop_left=0.0, crop_right=0.0):
    """
    Block until a fresh image is available, apply crop, and return it.
    All values are applied to the ROTATED image (90 CW from raw camera).

    After 90 CW rotation:
      top/bottom = camera left/right sides
      left/right = far/close to robot

    crop_top    : fraction to remove from top row    (camera right)
    crop_bottom : fraction to remove from bottom row  (camera left)
    crop_left   : fraction to remove from left col   (far from robot)
    crop_right  : fraction to remove from right col  (close to robot)
    """
    global latest_image
    latest_image = None
    deadline = time.time() + 3.0
    while latest_image is None and time.time() < deadline:
        rospy.sleep(0.05)
    if latest_image is None:
        return None
    img = latest_image.copy()
    h, w = img.shape[:2]
    t  = int(h * crop_top)
    b  = h - int(h * crop_bottom)
    l  = int(w * crop_left)
    r  = w - int(w * crop_right)
    return img[t:b, l:r]


# ================================================================== #
# LINE DETECTION (multi-colour)                                       #
# ================================================================== #

# HSV ranges for each supported colour.
# Each entry is a list of (lower, upper) pairs — supports colours that
# wrap around the hue wheel (e.g. red).
COLOR_RANGES = {
    "red": [
        (np.array([0,   120, 100]), np.array([10,  255, 255])),
        (np.array([170, 120, 100]), np.array([180, 255, 255])),
    ],
    "white": [
        (np.array([0,   0,   200]), np.array([180, 40,  255])),
    ],
    "magenta": [
        (np.array([140, 100, 100]), np.array([170, 255, 255])),
    ],
    "blue": [
        # S>=50, V>=150 to catch pale blues like RGB(177,177,255) — the sign top
        (np.array([100, 50, 150]), np.array([130, 255, 255])),
    ],
}


def detect_line(cv_image, color="red"):
    """
    Detect a coloured line in cv_image.

    Returns (angle_deg, cx, cy, img_w, img_h) or None.
      angle_deg : [-90, 90], 0 = horizontal
      cx, cy    : centroid of coloured pixels
    """
    if color not in COLOR_RANGES:
        rospy.logerr("Unknown color '%s'. Choose from: %s",
                     color, list(COLOR_RANGES.keys()))
        return None

    hsv    = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask   = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in COLOR_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

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


# ================================================================== #
# HELPERS                                                             #
# ================================================================== #

def body_to_world(vx_body, vy_body, heading_deg):
    """Convert body-frame velocities to world frame for planar_move."""
    h = math.radians(heading_deg)
    vx_world = vx_body * math.cos(h) - vy_body * math.sin(h)
    vy_world = vx_body * math.sin(h) + vy_body * math.cos(h)
    return vx_world, vy_world


# ================================================================== #
# FORWARD / TURN CALIBRATION                                          #
# ================================================================== #

def forward_error_pct(speed):
    return 4.6 - 3.0 * speed


def forward_corrected_duration(distance, speed):
    scale = 1.0 / (1.0 - forward_error_pct(speed) / 100.0)
    return (distance * scale) / speed


def turn_error_pct(speed):
    return 50.5 - 3.0 * speed


def turn_corrected_duration(angle_rad, speed):
    scale = 1.0 / (1.0 - turn_error_pct(speed) / 100.0)
    return (angle_rad * scale) / speed


# ================================================================== #
# MOTION PRIMITIVES                                                   #
# ================================================================== #

def stop():
    pub.publish(Twist())
    rospy.sleep(0.2)


def go_forward(distance, speed=0.3):
    """Drive forward `distance` m. Negative = backward."""
    if distance < 0:
        speed    = -abs(speed)
        distance =  abs(distance)
    duration = forward_corrected_duration(distance, abs(speed))
    rospy.loginfo("go_forward: %.3f m @ %.2f m/s -> %.3f s", distance, speed, duration)
    cmd = Twist()
    cmd.linear.x = speed
    rate = rospy.Rate(PUBLISH_HZ)
    end  = rospy.Time.now() + rospy.Duration(duration)
    while rospy.Time.now() < end and not rospy.is_shutdown():
        pub.publish(cmd)
        rate.sleep()
    stop()


def turn(angle_deg, speed=2.0):
    """Turn `angle_deg` degrees. Positive = CCW, negative = CW."""
    angle_rad = math.radians(abs(angle_deg))
    direction = 1.0 if angle_deg >= 0 else -1.0
    speed     = abs(speed)
    duration  = turn_corrected_duration(angle_rad, speed)
    rospy.loginfo("turn: %.1f deg @ %.2f rad/s -> %.3f s", angle_deg, speed, duration)
    cmd = Twist()
    cmd.angular.z = direction * speed
    rate = rospy.Rate(PUBLISH_HZ)
    end  = rospy.Time.now() + rospy.Duration(duration)
    while rospy.Time.now() < end and not rospy.is_shutdown():
        pub.publish(cmd)
        rate.sleep()
    stop()


# ================================================================== #
# LINE ALIGNMENT                                                      #
# ================================================================== #

def align_to_line(pub_,
                  heading_deg=0.0,
                  color="red",
                  target_y_ratio=DEFAULT_TARGET_Y_RATIO,
                  crop_top=DEFAULT_CROP_TOP,
                  crop_bottom=DEFAULT_CROP_BOTTOM,
                  crop_left=0.0,
                  crop_right=0.0,
                  timeout=30.0):
    """
    Visually servo the robot to a coloured line.

    Steps:
      1. Coarse rotation  — make line horizontal
      2. Translation      — centre line at target vertical position
      3. Fine rotation    — tighten residual angle

    Args:
        pub_          : rospy.Publisher for cmd_vel
        heading_deg   : robot's current heading in degrees (standard math, CCW positive).
                        Used only for body→world rotation in step 2. Defaults to 0.
                        Pass pose.heading_deg for accurate correction at non-zero headings.
        color         : "red", "white", or "magenta"
        target_y_ratio: vertical target in cropped image (0=bottom, 1=top)
                        e.g. 0.3 = line sits 30% up from the bottom
        crop_top      : fraction to discard from top    (camera right side)
        crop_bottom   : fraction to discard from bottom (camera left side)
        crop_left     : fraction to discard from left   (far from robot)
        crop_right    : fraction to discard from right  (close to robot)
        timeout       : total time budget (seconds)
    """
    _ensure_camera()
    rospy.loginfo("align_to_line: color=%s  target_y=%.0f%%  "
                  "crop_top=%.0f%%  crop_bottom=%.0f%%",
                  color, target_y_ratio * 100,
                  crop_top * 100, crop_bottom * 100)

    deadline = time.time() + timeout  # wall clock — avoids sim-time sync issues

    def img():
        return get_image(crop_top=crop_top, crop_bottom=crop_bottom,
                            crop_left=crop_left, crop_right=crop_right)

    def detect(frame):
        return detect_line(frame, color=color)

    def pulse(cmd):
        pub_.publish(cmd)
        rospy.sleep(1.0 / ALIGN_HZ)
        pub_.publish(Twist())

    # ---- Step 1: Coarse rotation ---- #
    rospy.loginfo("  Step 1: coarse rotation...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.logwarn("  No image, retrying...")
            continue
        result = detect(frame)
        if result is None:
            rospy.logwarn("  Line not detected, waiting...")
            rospy.sleep(1.0 / ALIGN_HZ)
            continue
        angle_deg, cx, cy, img_w, img_h = result
        rospy.loginfo("  angle=%.1f", angle_deg)
        if abs(angle_deg) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 1 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(GAIN_ANGLE * angle_deg,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub_.publish(Twist())
    rospy.sleep(0.3)

    # ---- Step 2: Translation ---- #
    rospy.loginfo("  Step 2: translating...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ)
            continue
        result = detect(frame)
        if result is None:
            rospy.logwarn("  Line lost during step 2.")
            break
        angle_deg, cx, cy, img_w, img_h = result
        # Reconstruct full image dimensions from cropped + crop params,
        # so target_y_ratio is always a fraction of the FULL image.
        # This prevents crop_top from shifting the effective target position.
        img_h_full = img_h / (1.0 - crop_top - crop_bottom) if (crop_top + crop_bottom) < 1.0 else img_h
        img_w_full = img_w / (1.0 - crop_left - crop_right) if (crop_left + crop_right) < 1.0 else img_w
        # Convert full-image target to cropped-image coordinates
        target_y_full = target_y_ratio * img_h_full
        target_y      = int(target_y_full - crop_top * img_h_full)
        target_x_full = img_w_full / 2.0
        target_x      = int(target_x_full - crop_left * img_w_full)
        y_error  = cy - target_y
        x_error  = cx - target_x
        rospy.loginfo("  x_err=%d  y_err=%d  angle=%.1f", x_error, y_error, angle_deg)
        if (abs(y_error) < THRESH_PIXEL_Y and
                abs(x_error) < THRESH_PIXEL_X and
                abs(angle_deg) < THRESH_ANGLE_DEG):
            rospy.loginfo("  Step 2 done.")
            break
        # Convert body-frame corrections to world frame using current heading
        vx_body = float(np.clip(-GAIN_FORWARD * y_error, -MAX_LINEAR, MAX_LINEAR))
        vy_body = float(np.clip(-GAIN_LATERAL * x_error, -MAX_LINEAR, MAX_LINEAR))
        vx_world, vy_world = body_to_world(vx_body, vy_body, heading_deg)
        cmd = Twist()
        cmd.linear.x  = vx_world
        cmd.linear.y  = vy_world
        cmd.angular.z = float(np.clip(-GAIN_ANGLE * angle_deg, -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub_.publish(Twist())
    rospy.sleep(0.3)

    # ---- Step 3: Fine rotation ---- #
    rospy.loginfo("  Step 3: fine rotation...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ)
            continue
        result = detect(frame)
        if result is None:
            rospy.logwarn("  Line lost during step 3.")
            break
        angle_deg, cx, cy, img_w, img_h = result
        rospy.loginfo("  angle=%.1f (fine)", angle_deg)
        if abs(angle_deg) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 3 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(GAIN_ANGLE_FINE * angle_deg,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub_.publish(Twist())
    rospy.loginfo("  align_to_line complete.")


# ================================================================== #
# SIGN ALIGNMENT                                                      #
# ================================================================== #

def detect_sign_top_edge(cv_image):
    """
    Detect the top edge of the blue sign in cv_image.

    Strategy: find all blue pixels, isolate the topmost row(s)
    (within TOP_EDGE_BAND pixels of the minimum y), then fit a line
    to those pixels to get angle and centroid.

    Returns (angle_deg, cx, cy, img_w, img_h) or None.
      angle_deg : tilt of top edge, [-90, 90], 0 = horizontal
      cx, cy    : centroid of top-edge pixels
    """
    TOP_EDGE_BAND = 5   # pixels — how many rows below topmost to include

    hsv   = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv,
                        np.array([100, 50, 150]),
                        np.array([130, 255, 255]))

    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_h, img_w = cv_image.shape[:2]

    # Find all blue pixel coordinates
    ys, xs = np.where(mask > 0)
    if len(ys) < 10:
        return None

    # Isolate top-edge pixels
    y_min   = ys.min()
    top_idx = ys <= (y_min + TOP_EDGE_BAND)
    top_xs  = xs[top_idx]
    top_ys  = ys[top_idx]

    if len(top_xs) < 2:
        return None

    # Centroid of top-edge pixels
    cx = int(np.mean(top_xs))
    cy = int(np.mean(top_ys))

    # Angle of top edge via linear regression on top pixels
    if len(top_xs) >= 2:
        coeffs    = np.polyfit(top_xs, top_ys, 1)   # slope = dy/dx
        angle_deg = math.degrees(math.atan(coeffs[0]))
    else:
        angle_deg = 0.0

    return angle_deg, cx, cy, img_w, img_h


def align_to_sign(pub_,
                  target_x_ratio=0.5,
                  target_y_ratio=0.4,
                  crop_top=0.0,
                  crop_bottom=0.0,
                  crop_left=0.0,
                  crop_right=0.0,
                  timeout=30.0):
    """
    Align the robot using the blue sign's top edge as a reference.

    Steps:
      1. Rotate until the top edge of the sign is horizontal
      2. Strafe/translate until the top edge centroid is at
         (target_x_ratio, target_y_ratio) in the image
      3. Fine rotation to correct any residual angle

    Args:
        pub_          : rospy.Publisher for cmd_vel
        target_x_ratio: horizontal target (0=left, 1=right, 0.5=centre)
        target_y_ratio: vertical target   (0=bottom, 1=top of full image)
        crop_*        : fractions to discard from each edge
        timeout       : total time budget (seconds)
    """
    _ensure_camera()
    rospy.loginfo("align_to_sign: target=(%.0f%%, %.0f%%)",
                  target_x_ratio * 100, target_y_ratio * 100)

    deadline = time.time() + timeout

    def img():
        return get_image(crop_top=crop_top, crop_bottom=crop_bottom,
                         crop_left=crop_left, crop_right=crop_right)

    def detect(frame):
        return detect_sign_top_edge(frame)

    def pulse(cmd):
        pub_.publish(cmd)
        rospy.sleep(1.0 / ALIGN_HZ)
        pub_.publish(Twist())

    # ---- Step 1: Coarse rotation — top edge horizontal ---- #
    rospy.loginfo("  Step 1: rotating until sign top edge is horizontal...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.logwarn("  No image, retrying...")
            continue
        result = detect(frame)
        if result is None:
            rospy.logwarn("  Sign not detected — strafing to search...")
            search = Twist()
            search.linear.y = -SIGN_SEARCH_SPEED if target_x_ratio >= 0.5 \
                               else SIGN_SEARCH_SPEED
            pulse(search)
            continue
        angle_deg, cx, cy, img_w, img_h = result
        rospy.loginfo("  top-edge angle=%.1f deg  centroid=(%d, %d)",
                      angle_deg, cx, cy)
        if abs(angle_deg) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 1 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(SIGN_GAIN_ANGLE * angle_deg,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub_.publish(Twist())
    rospy.sleep(0.3)

    # ---- Step 2: Translation — move centroid to target ---- #
    rospy.loginfo("  Step 2: translating to position sign centroid...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ)
            continue
        result = detect(frame)
        if result is None:
            rospy.logwarn("  Sign lost during step 2 — strafing to search...")
            search = Twist()
            search.linear.y = -SIGN_SEARCH_SPEED if target_x_ratio >= 0.5 \
                               else SIGN_SEARCH_SPEED
            pulse(search)
            continue
        angle_deg, cx, cy, img_w, img_h = result

        # Compute targets in cropped image coordinates
        img_h_full = img_h / (1.0 - crop_top - crop_bottom)                      if (crop_top + crop_bottom) < 1.0 else img_h
        img_w_full = img_w / (1.0 - crop_left - crop_right)                      if (crop_left + crop_right) < 1.0 else img_w

        target_y_full = target_y_ratio * img_h_full
        target_x_full = target_x_ratio * img_w_full
        target_y_crop = int(target_y_full - crop_top  * img_h_full)
        target_x_crop = int(target_x_full - crop_left * img_w_full)

        y_error = cy - target_y_crop
        x_error = cx - target_x_crop

        rospy.loginfo("  cx=%d cy=%d  x_err=%d  y_err=%d  angle=%.1f",
                      cx, cy, x_error, y_error, angle_deg)

        if (abs(y_error) < THRESH_PIXEL_Y and
                abs(x_error) < THRESH_PIXEL_X and
                abs(angle_deg) < THRESH_ANGLE_DEG):
            rospy.loginfo("  Step 2 done.")
            break

        # Image axes = robot body axes (top-down fixed camera) — no transform needed
        cmd = Twist()
        cmd.linear.x  = float(np.clip(-GAIN_FORWARD * y_error, -MAX_LINEAR, MAX_LINEAR))
        cmd.linear.y  = float(np.clip(-GAIN_LATERAL * x_error, -MAX_LINEAR, MAX_LINEAR))
        cmd.angular.z = float(np.clip(-SIGN_GAIN_ANGLE * angle_deg,
                                      -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub_.publish(Twist())
    rospy.sleep(0.3)

    # ---- Step 3: Fine rotation ---- #
    rospy.loginfo("  Step 3: fine rotation...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ)
            continue
        result = detect(frame)
        if result is None:
            rospy.logwarn("  Sign lost during step 3 — strafing to search...")
            search = Twist()
            search.linear.y = -SIGN_SEARCH_SPEED if target_x_ratio >= 0.5 \
                               else SIGN_SEARCH_SPEED
            pulse(search)
            continue
        angle_deg, cx, cy, img_w, img_h = result
        rospy.loginfo("  angle=%.1f deg (fine)", angle_deg)
        if abs(angle_deg) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 3 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(SIGN_GAIN_ANGLE_FINE * angle_deg,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub_.publish(Twist())
    rospy.loginfo("  align_to_sign complete.")


# ================================================================== #
# MAIN (standalone test)                                              #
# ================================================================== #

def main():
    global pub
    rospy.init_node("movement", anonymous=True)
    pub = rospy.Publisher(TOPIC, Twist, queue_size=1)
    rospy.Subscriber(CAM_TOPIC, Image, image_callback)
    rospy.sleep(1.0)

    go_forward(1.0, speed=0.3)
    turn(-90.0, speed=2.0)
    align_to_line(pub, color="red")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass