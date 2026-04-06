#!/usr/bin/env python3
"""
adjustment.py
-------------
Visual alignment using rotation and forward/back movement.
Lateral strafing is disabled by default but can be enabled.

align_to_line(pub, color, ...)  — align to a coloured line
align_to_sign(pub, ...)         — align to the blue sign top edge

Both functions:
  Step 1: Rotate until feature is horizontal
  Step 2: Forward/back (+ optional strafe) to reach target position
  Step 3: Fine rotation

When the feature is not detected the robot creeps forward by default.
Set search_backward=True to creep backward instead.
"""

import math
import time
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

CAM_TOPIC = "/B1/robot/camera_top/image_raw"

# ------------------------------------------------------------------ #
# Parameters                                                           #
# ------------------------------------------------------------------ #

ALIGN_HZ          = 20

GAIN_ANGLE        = 0.6
GAIN_ANGLE_FINE   = 0.3
GAIN_FORWARD      = 0.004
GAIN_LATERAL      = 0.004

SIGN_GAIN_ANGLE       = 0.6
SIGN_GAIN_ANGLE_FINE  = 0.3
SIGN_GAIN_FORWARD     = 0.004
SIGN_GAIN_LATERAL     = 0.004

SEARCH_SPEED      = 0.05   # m/s creep when feature not detected

THRESH_ANGLE_DEG  = 0.5
THRESH_PIXEL_Y    = 10
THRESH_PIXEL_X    = 15
MAX_ANGULAR       = 1.0
MAX_LINEAR        = 0.3
MIN_ANGULAR       = 0.4   # minimum angular speed to overcome static friction
MIN_LINEAR        = 0.05   # minimum linear speed to overcome static friction

CROP_TOP_DEFAULT  = 0.0
DEFAULT_TARGET_X  = 0.5
DEFAULT_TARGET_Y  = 0.4

TOP_EDGE_BAND     = 5

COLOR_RANGES = {
    "red": [
        (np.array([0,   120, 100]), np.array([10,  255, 255])),
        (np.array([170, 120, 100]), np.array([180, 255, 255])),
    ],
    "white": [
        (np.array([0,   0,   200]), np.array([180, 40,  255])),
    ],
    "magenta": [
        (np.array([140, 200, 200]), np.array([170, 255, 255])),
    ],
    "blue": [
        (np.array([100,  50, 150]), np.array([130, 255, 255])),
    ],
}

# ------------------------------------------------------------------ #
# Camera                                                               #
# ------------------------------------------------------------------ #

bridge       = CvBridge()
latest_image = None
_cam_sub     = None


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
    global latest_image
    latest_image = None
    deadline = time.time() + 3.0
    while latest_image is None and time.time() < deadline:
        rospy.sleep(0.05)
    if latest_image is None:
        return None
    img  = latest_image.copy()
    h, w = img.shape[:2]
    t = int(h * crop_top);    b = h - int(h * crop_bottom)
    l = int(w * crop_left);   r = w - int(w * crop_right)
    return img[t:b, l:r]


# ------------------------------------------------------------------ #
# Detection                                                            #
# ------------------------------------------------------------------ #

def detect_line(cv_image, color="red"):
    if color not in COLOR_RANGES:
        rospy.logerr("Unknown color '%s'", color)
        return None
    hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
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


def detect_sign_top_edge(cv_image):
    hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 50, 150]), np.array([130, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    img_h, img_w = cv_image.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) < 10:
        return None
    y_min   = ys.min()
    top_idx = ys <= (y_min + TOP_EDGE_BAND)
    top_xs  = xs[top_idx];  top_ys = ys[top_idx]
    if len(top_xs) < 2:
        return None
    cx        = int(np.mean(top_xs));  cy = int(np.mean(top_ys))
    coeffs    = np.polyfit(top_xs, top_ys, 1)
    angle_deg = math.degrees(math.atan(coeffs[0]))
    return angle_deg, cx, cy, img_w, img_h


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _apply_min_speeds(cmd):
    """Ensure non-zero commands meet minimum speeds to overcome static friction."""
    if cmd.angular.z != 0.0:
        if 0 < abs(cmd.angular.z) < MIN_ANGULAR:
            cmd.angular.z = math.copysign(MIN_ANGULAR, cmd.angular.z)
    if cmd.linear.x != 0.0:
        if 0 < abs(cmd.linear.x) < MIN_LINEAR:
            cmd.linear.x = math.copysign(MIN_LINEAR, cmd.linear.x)
    if cmd.linear.y != 0.0:
        if 0 < abs(cmd.linear.y) < MIN_LINEAR:
            cmd.linear.y = math.copysign(MIN_LINEAR, cmd.linear.y)
    return cmd


def _pulse(pub, cmd):
    pub.publish(_apply_min_speeds(cmd))
    rospy.sleep(1.0 / ALIGN_HZ)
    pub.publish(Twist())


def _target_y_px(target_y_ratio, img_h, crop_top, crop_bottom):
    img_h_full = img_h / (1.0 - crop_top - crop_bottom) \
                 if (crop_top + crop_bottom) < 1.0 else img_h
    return int(target_y_ratio * img_h_full - crop_top * img_h_full)


def _target_x_px(target_x_ratio, img_w, crop_left, crop_right):
    img_w_full = img_w / (1.0 - crop_left - crop_right) \
                 if (crop_left + crop_right) < 1.0 else img_w
    return int(target_x_ratio * img_w_full - crop_left * img_w_full)


# ------------------------------------------------------------------ #
# Core alignment                                                       #
# ------------------------------------------------------------------ #

def _align(pub, detect_fn,
           gain_angle, gain_angle_fine, gain_fwd, gain_lat,
           target_x_ratio, target_y_ratio,
           crop_top, crop_bottom, crop_left, crop_right,
           enable_strafe, strafe_left_when_positive,
           search_backward, timeout, label,
           target_vertical=False):
    """
    3-step alignment:
      1. Rotate until feature is horizontal (or vertical if target_vertical=True)
      2. Forward/back (+ optional strafe) to reach target position
      3. Fine rotation
    """
    deadline = time.time() + timeout

    def img():
        return get_image(crop_top=crop_top, crop_bottom=crop_bottom,
                         crop_left=crop_left, crop_right=crop_right)

    def not_detected():
        rospy.logwarn("  %s not detected — %s",
                      label, "backing up" if search_backward else "creeping forward")
        cmd = Twist()
        cmd.linear.x = -SEARCH_SPEED if search_backward else SEARCH_SPEED
        _pulse(pub, cmd)

    # ---- Step 1: Rotate → horizontal or vertical ---- #
    orient_label = "vertical" if target_vertical else "horizontal"
    rospy.loginfo("  Step 1: rotating until %s is %s...", label, orient_label)
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None: continue
        result = detect_fn(frame)
        if result is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        angle_deg, cx, cy, img_w, img_h = result
        # For vertical target, error is distance from ±90°
        if target_vertical:
            # Pick the closer of +90 and -90
            err_pos = angle_deg - 90.0
            err_neg = angle_deg + 90.0
            error = err_pos if abs(err_pos) < abs(err_neg) else err_neg
        else:
            error = angle_deg
        rospy.loginfo("  angle=%.1f  error=%.1f", angle_deg, error)
        if abs(error) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 1 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(gain_angle * error, -MAX_ANGULAR, MAX_ANGULAR))
        _pulse(pub, cmd)

    pub.publish(Twist());  rospy.sleep(0.3)

    # ---- Step 2: Position (forward/back + optional strafe) ---- #
    rospy.loginfo("  Step 2: positioning (strafe=%s)...", enable_strafe)
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        result = detect_fn(frame)
        if result is None:
            not_detected(); continue
        angle_deg, cx, cy, img_w, img_h = result
        target_y = _target_y_px(target_y_ratio, img_h, crop_top, crop_bottom)
        target_x = _target_x_px(target_x_ratio, img_w, crop_left, crop_right)
        y_error  = cy - target_y
        x_error  = cx - target_x

        # Use same angle error as steps 1 & 3
        if target_vertical:
            err_pos = angle_deg - 90.0
            err_neg = angle_deg + 90.0
            angle_error = err_pos if abs(err_pos) < abs(err_neg) else err_neg
        else:
            angle_error = angle_deg

        done_y = abs(y_error) < THRESH_PIXEL_Y
        done_x = abs(x_error) < THRESH_PIXEL_X or not enable_strafe
        done_a = abs(angle_error) < THRESH_ANGLE_DEG

        rospy.loginfo("  y_err=%d  x_err=%d  angle_err=%.1f", y_error, x_error, angle_error)

        if done_y and done_x and done_a:
            rospy.loginfo("  Step 2 done.")
            break

        cmd = Twist()
        cmd.linear.x  = float(np.clip(-gain_fwd * y_error, -MAX_LINEAR, MAX_LINEAR))
        cmd.angular.z = float(np.clip(-gain_angle * angle_error, -MAX_ANGULAR, MAX_ANGULAR))
        if enable_strafe:
            # strafe_left_when_positive=True: positive x_error → strafe left (+vy)
            sign = 1.0 if strafe_left_when_positive else -1.0
            cmd.linear.y = float(np.clip(-sign * gain_lat * x_error,
                                         -MAX_LINEAR, MAX_LINEAR))
        _pulse(pub, cmd)

    pub.publish(Twist());  rospy.sleep(0.3)

    # ---- Step 3: Fine rotation ---- #
    rospy.loginfo("  Step 3: fine rotation...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        result = detect_fn(frame)
        if result is None:
            break
        angle_deg, cx, cy, img_w, img_h = result
        if target_vertical:
            err_pos = angle_deg - 90.0
            err_neg = angle_deg + 90.0
            error = err_pos if abs(err_pos) < abs(err_neg) else err_neg
        else:
            error = angle_deg
        rospy.loginfo("  angle=%.1f  error=%.1f (fine)", angle_deg, error)
        if abs(error) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 3 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(gain_angle_fine * error,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        _pulse(pub, cmd)

    pub.publish(Twist())
    rospy.loginfo("  %s alignment complete.", label)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def align_to_line(pub,
                  color="red",
                  target_x_ratio=DEFAULT_TARGET_X,
                  target_y_ratio=DEFAULT_TARGET_Y,
                  crop_top=CROP_TOP_DEFAULT,
                  crop_bottom=0.0,
                  crop_left=0.0,
                  crop_right=0.0,
                  enable_strafe=False,
                  strafe_left_when_positive=True,
                  search_backward=False,
                  target_vertical=False,
                  timeout=30.0):
    """
    Align to a coloured line.

    Args:
        pub                    : rospy.Publisher for cmd_vel
        color                  : "red", "white", or "magenta"
        target_x_ratio         : horizontal target — only used if enable_strafe=True
        target_y_ratio         : vertical target (0=bottom, 1=top)
        crop_*                 : image crop fractions
        enable_strafe          : if True, also correct lateral (x) position
        strafe_left_when_positive: strafe direction when x_error > 0
        search_backward        : if True, reverse when line not detected
        target_vertical        : if True, rotate until line is vertical instead of horizontal
        timeout                : total time budget (seconds)
    """
    _ensure_camera()
    orient = "vertical" if target_vertical else "horizontal"
    rospy.loginfo("align_to_line: color=%s  y=%.0f%%  target=%s  strafe=%s",
                  color, target_y_ratio * 100, orient, enable_strafe)

    def detect(frame):
        return detect_line(frame, color)

    _align(pub, detect,
           GAIN_ANGLE, GAIN_ANGLE_FINE, GAIN_FORWARD, GAIN_LATERAL,
           target_x_ratio, target_y_ratio,
           crop_top, crop_bottom, crop_left, crop_right,
           enable_strafe, strafe_left_when_positive,
           search_backward, timeout, f"{color} line",
           target_vertical=target_vertical)


def align_to_sign(pub,
                  target_x_ratio=DEFAULT_TARGET_X,
                  target_y_ratio=DEFAULT_TARGET_Y,
                  crop_top=0.0,
                  crop_bottom=0.0,
                  crop_left=0.0,
                  crop_right=0.0,
                  enable_strafe=False,
                  strafe_left_when_positive=True,
                  search_backward=False,
                  timeout=30.0):
    """
    Align to the blue sign top edge.

    Args:
        pub                    : rospy.Publisher for cmd_vel
        target_x_ratio         : horizontal target (0=left, 1=right) — only used if enable_strafe=True
        target_y_ratio         : vertical target (0=bottom, 1=top)
        crop_*                 : image crop fractions
        enable_strafe          : if True, also correct lateral (x) position
        strafe_left_when_positive: if True, positive x_error → strafe left
        search_backward        : if True, reverse when sign not detected
        timeout                : total time budget (seconds)
    """
    _ensure_camera()
    rospy.loginfo("align_to_sign: y=%.0f%%  strafe=%s",
                  target_y_ratio * 100, enable_strafe)

    _align(pub, detect_sign_top_edge,
           SIGN_GAIN_ANGLE, SIGN_GAIN_ANGLE_FINE, SIGN_GAIN_FORWARD, SIGN_GAIN_LATERAL,
           target_x_ratio, target_y_ratio,
           crop_top, crop_bottom, crop_left, crop_right,
           enable_strafe, strafe_left_when_positive,
           search_backward, timeout, "blue sign")


# ================================================================== #
# COMPLEX LINE ALIGNMENT (horizontal + vertical discrimination)       #
# ================================================================== #

def _detect_line_orientation(cv_image, color="red"):
    """
    Detects a coloured line and returns its angle, centroid, and
    a stable 'true_angle' in [-180, 180] that correctly distinguishes
    horizontal (~0°) from vertical (~90° or ~-90°).

    Instead of relying solely on HoughLinesP (which wraps at ±90°),
    we use the principal axis of the pixel distribution via PCA.
    PCA gives a true orientation vector — we then project it to
    a signed angle in [-180, 180].

    Returns (true_angle_deg, cx, cy, img_w, img_h) or None.
    """
    if color not in COLOR_RANGES:
        return None
    hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_h, img_w = cv_image.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return None

    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    # PCA on pixel coordinates to find principal axis
    pts   = np.column_stack([xs - cx, ys - cy]).astype(np.float32)
    cov   = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    # Principal eigenvector (largest eigenvalue)
    principal = evecs[:, np.argmax(evals)]
    dx, dy    = principal

    # atan2 gives angle of the principal axis in [-180, 180]
    # dx = x-component, dy = y-component (image coords: y down)
    true_angle_deg = math.degrees(math.atan2(dy, dx))

    # Normalise to [-90, 90] for consistency with Hough convention
    # (a line at 100° is the same as -80°)
    while true_angle_deg >  90: true_angle_deg -= 180
    while true_angle_deg < -90: true_angle_deg += 180

    return true_angle_deg, cx, cy, img_w, img_h


def _align_complex(pub, color,
                   gain_angle, gain_angle_fine, gain_fwd,
                   target_x_ratio, target_y_ratio,
                   crop_top, crop_bottom, crop_left, crop_right,
                   enable_strafe, strafe_left_when_positive,
                   search_backward, timeout):
    """
    3-step line alignment using PCA-based orientation (stable for both
    horizontal and vertical lines):

      Step 1: Forward/back → y centroid at target_y_ratio
      Step 2: Rotate       → line horizontal (angle → 0°)
      Step 3: Forward/back → x centroid at target_x_ratio
                             (+ fine rotation, + optional strafe)
    """
    deadline = time.time() + timeout
    label    = f"{color} line (complex)"

    def img():
        return get_image(crop_top=crop_top, crop_bottom=crop_bottom,
                         crop_left=crop_left, crop_right=crop_right)

    def detect(frame):
        return _detect_line_orientation(frame, color)

    def not_detected():
        rospy.logwarn("  %s not detected — %s",
                      label, "backing up" if search_backward else "creeping forward")
        cmd = Twist()
        cmd.linear.x = -SEARCH_SPEED if search_backward else SEARCH_SPEED
        _pulse(pub, cmd)

    # ---- Step 1: Forward/back → y centroid ---- #
    rospy.loginfo("  [complex] Step 1: forward/back to align y (target_y=%.0f%%)...",
                  target_y_ratio * 100)
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        result = detect(frame)
        if result is None:
            not_detected(); continue
        angle_deg, cx, cy, img_w, img_h = result
        target_y = _target_y_px(target_y_ratio, img_h, crop_top, crop_bottom)
        y_error  = cy - target_y
        rospy.loginfo("  cy=%d  target=%d  y_err=%d  angle=%.1f",
                      cy, target_y, y_error, angle_deg)
        if abs(y_error) < THRESH_PIXEL_Y:
            rospy.loginfo("  Step 1 done.")
            break
        cmd = Twist()
        cmd.linear.x = float(np.clip(-gain_fwd * y_error, -MAX_LINEAR, MAX_LINEAR))
        _pulse(pub, cmd)

    pub.publish(Twist());  rospy.sleep(0.3)

    # ---- Step 2: Rotate → horizontal (angle → 0°) ---- #
    rospy.loginfo("  [complex] Step 2: rotating until line is horizontal...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None: continue
        result = detect(frame)
        if result is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        angle_deg, cx, cy, img_w, img_h = result
        rospy.loginfo("  angle=%.1f (PCA)", angle_deg)
        if abs(angle_deg) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Step 2 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(gain_angle * angle_deg,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        _pulse(pub, cmd)

    pub.publish(Twist());  rospy.sleep(0.3)

    # ---- Step 3: Forward/back → x centroid (+ fine rotation) ---- #
    rospy.loginfo("  [complex] Step 3: x positioning (strafe=%s)...", enable_strafe)
    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        result = detect(frame)
        if result is None:
            not_detected(); continue
        angle_deg, cx, cy, img_w, img_h = result
        target_x = _target_x_px(target_x_ratio, img_w, crop_left, crop_right)
        x_error  = cx - target_x

        done_x = abs(x_error) < THRESH_PIXEL_X or not enable_strafe
        done_a = abs(angle_deg) < THRESH_ANGLE_DEG

        rospy.loginfo("  cx=%d  x_err=%d  angle=%.1f", cx, x_error, angle_deg)

        if done_x and done_a:
            rospy.loginfo("  Step 3 done.")
            break

        cmd = Twist()
        cmd.linear.x  = float(np.clip(-gain_fwd          * x_error,   -MAX_LINEAR,  MAX_LINEAR))
        cmd.angular.z = float(np.clip(-gain_angle_fine    * angle_deg, -MAX_ANGULAR, MAX_ANGULAR))
        if enable_strafe:
            sign = 1.0 if strafe_left_when_positive else -1.0
            cmd.linear.y = float(np.clip(-sign * GAIN_LATERAL * x_error,
                                         -MAX_LINEAR, MAX_LINEAR))
        _pulse(pub, cmd)

    pub.publish(Twist())
    rospy.loginfo("  %s alignment complete.", label)


def align_to_line_complex(pub,
                          color="red",
                          target_x_ratio=DEFAULT_TARGET_X,
                          target_y_ratio=DEFAULT_TARGET_Y,
                          crop_top=CROP_TOP_DEFAULT,
                          crop_bottom=0.0,
                          crop_left=0.0,
                          crop_right=0.0,
                          enable_strafe=False,
                          strafe_left_when_positive=True,
                          search_backward=False,
                          timeout=30.0):
    """
    Complex line alignment using PCA-based orientation detection.
    More robust than align_to_line for distinguishing horizontal
    from vertical lines.

    Steps:
      1. Forward/back → y centroid at target_y_ratio
      2. Rotate → line horizontal
      3. Forward/back → x centroid at target_x_ratio (+ fine rotation)

    Args:
        pub                    : rospy.Publisher for cmd_vel
        color                  : "red", "white", or "magenta"
        target_x_ratio         : horizontal target (used if enable_strafe=True)
        target_y_ratio         : vertical target (0=bottom, 1=top)
        crop_*                 : image crop fractions
        enable_strafe          : if True, correct x position laterally
        strafe_left_when_positive: strafe direction when x_error > 0
        search_backward        : creep backward when line not detected
        timeout                : total time budget (seconds)
    """
    _ensure_camera()
    rospy.loginfo("align_to_line_complex: color=%s  y=%.0f%%  strafe=%s",
                  color, target_y_ratio * 100, enable_strafe)

    _align_complex(pub, color,
                   GAIN_ANGLE, GAIN_ANGLE_FINE, GAIN_FORWARD,
                   target_x_ratio, target_y_ratio,
                   crop_top, crop_bottom, crop_left, crop_right,
                   enable_strafe, strafe_left_when_positive,
                   search_backward, timeout)


# ================================================================== #
# WATER-BASED CENTERING                                               #
# ================================================================== #

# HSV range for the water (blue-ish, less saturated than the sign)
# Tune these if the water colour doesn't match
WATER_HSV_LO = np.array([109,  34,  80])
WATER_HSV_HI = np.array([125,  64, 220])

# How many rows from the bottom of the image to sample
WATER_BOTTOM_BAND = 0   # 0 = use full image; >0 = restrict to bottom N rows


def detect_water_center(cv_image):
    """
    Finds the midpoint between the two bodies of water using the
    bottom WATER_BOTTOM_BAND rows where the water colour is most consistent.

    Strategy:
      1. Mask water pixels
      2. Restrict to the bottom N rows
      3. Find the rightmost water pixel on the LEFT side of the image
         and the leftmost water pixel on the RIGHT side
      4. The road centre = midpoint between those two edges

    Returns (road_cx, img_w) or None if fewer than two water regions found.
    """
    hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WATER_HSV_LO, WATER_HSV_HI)

    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_h, img_w = cv_image.shape[:2]

    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None

    # Optionally restrict to bottom band for consistency.
    # WATER_BOTTOM_BAND=0 means use all rows.
    if WATER_BOTTOM_BAND > 0:
        band_start = max(0, img_h - WATER_BOTTOM_BAND)
        band_mask  = ys >= band_start
        ys = ys[band_mask]
        xs = xs[band_mask]
        if len(xs) < 10:
            # Fall back to full image if bottom band has too few pixels
            ys_all, xs_all = np.where(mask > 0)
            ys, xs = ys_all, xs_all
            if len(xs) < 10:
                return None

    mid = img_w // 2

    # Left water body: pixels left of centre — take the rightmost edge
    left_xs = xs[xs < mid]
    # Right water body: pixels right of centre — take the leftmost edge
    right_xs = xs[xs >= mid]

    if len(left_xs) == 0 or len(right_xs) == 0:
        return None

    left_edge  = int(np.max(left_xs))    # rightmost pixel of left water
    right_edge = int(np.min(right_xs))   # leftmost pixel of right water

    road_cx = (left_edge + right_edge) // 2
    road_cy = int(np.mean(ys))   # vertical position of water pixels
    return road_cx, road_cy, img_w, img_h


def align_between_water(pub,
                        target_y_ratio=0.5,
                        crop_top=0.0,
                        crop_bottom=0.0,
                        crop_left=0.0,
                        crop_right=0.0,
                        search_backward=False,
                        timeout=30.0):
    """
    Position the robot relative to two bodies of water using forward/back
    movement only (no strafe, no rotation).

    Drives until the vertical centroid of the water pixels reaches
    target_y_ratio in the image. A higher ratio places the water
    boundary further up in the frame (robot closer to water).

    Args:
        pub             : rospy.Publisher for cmd_vel
        target_y_ratio  : vertical target (0=bottom, 1=top of full image)
        crop_*          : image crop fractions
        search_backward : creep backward when water not detected
        timeout         : total time budget (seconds)
    """
    _ensure_camera()
    rospy.loginfo("align_between_water: target_y=%.0f%%", target_y_ratio * 100)

    deadline = time.time() + timeout

    def img():
        return get_image(crop_top=crop_top, crop_bottom=crop_bottom,
                         crop_left=crop_left, crop_right=crop_right)

    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ)
            continue

        result = detect_water_center(frame)
        if result is None:
            rospy.logwarn("  Water not detected — %s",
                          "backing up" if search_backward else "creeping forward")
            cmd = Twist()
            cmd.linear.x = -SEARCH_SPEED if search_backward else SEARCH_SPEED
            _pulse(pub, cmd)
            continue

        road_cx, road_cy, img_w, img_h = result
        target_y = _target_y_px(target_y_ratio, img_h, crop_top, crop_bottom)
        y_error  = road_cy - target_y

        rospy.loginfo("  road_cy=%d  target_y=%d  y_err=%d", road_cy, target_y, y_error)

        if abs(y_error) < THRESH_PIXEL_Y:
            rospy.loginfo("  Water at target position. Done.")
            break

        cmd = Twist()
        cmd.linear.x = float(np.clip(-GAIN_FORWARD * y_error, -MAX_LINEAR, MAX_LINEAR))
        _pulse(pub, cmd)

    pub.publish(Twist())
    rospy.loginfo("align_between_water complete.")


# ================================================================== #
# WATER LINE ALIGNMENT                                                #
# ================================================================== #

def detect_water_line_angle(cv_image):
    """
    Detects the angle of the water boundary by finding the bottom edge
    of the water in each column (the water-land boundary), then fitting
    a line to those edge points.

    This avoids the PCA pitfall of spanning between two separate water
    bodies instead of measuring each individual edge.

    Returns the angle in degrees [-90, 90] where 0 = horizontal, or None.
    """
    hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WATER_HSV_LO, WATER_HSV_HI)

    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_h, img_w = mask.shape

    # For each column, find the bottommost water pixel (highest y value)
    edge_xs = []
    edge_ys = []
    for x in range(img_w):
        col = mask[:, x]
        water_rows = np.where(col > 0)[0]
        if len(water_rows) > 0:
            edge_xs.append(x)
            edge_ys.append(int(np.max(water_rows)))  # bottommost water pixel

    if len(edge_xs) < 10:
        return None

    edge_xs = np.array(edge_xs, dtype=np.float32)
    edge_ys = np.array(edge_ys, dtype=np.float32)

    # Fit a line: y = slope * x + intercept
    coeffs    = np.polyfit(edge_xs, edge_ys, 1)
    angle_deg = math.degrees(math.atan(coeffs[0]))

    # Fold to [-90, 90]
    while angle_deg >  90: angle_deg -= 180
    while angle_deg < -90: angle_deg += 180

    return angle_deg


def align_water_horizontal(pub,
                           crop_top=0.0,
                           crop_bottom=0.0,
                           crop_left=0.0,
                           crop_right=0.0,
                           search_backward=False,
                           timeout=30.0):
    """
    Rotate the robot until the water boundary line is horizontal.
    Uses PCA on water pixels to find the dominant orientation.

    Args:
        pub             : rospy.Publisher for cmd_vel
        crop_*          : image crop fractions
        search_backward : creep backward when water not detected
        timeout         : total time budget (seconds)
    """
    _ensure_camera()
    rospy.loginfo("align_water_horizontal: rotating until water line is horizontal...")

    deadline = time.time() + timeout

    def img():
        return get_image(crop_top=crop_top, crop_bottom=crop_bottom,
                         crop_left=crop_left, crop_right=crop_right)

    while time.time() < deadline and not rospy.is_shutdown():
        frame = img()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ)
            continue

        angle_deg = detect_water_line_angle(frame)

        if angle_deg is None:
            rospy.logwarn("  Water not detected — %s",
                          "backing up" if search_backward else "creeping forward")
            cmd = Twist()
            cmd.linear.x = -SEARCH_SPEED if search_backward else SEARCH_SPEED
            _pulse(pub, cmd)
            continue

        rospy.loginfo("  water line angle=%.1f deg", angle_deg)

        if abs(angle_deg) < THRESH_ANGLE_DEG:
            rospy.loginfo("  Water line is horizontal. Done.")
            break

        cmd = Twist()
        cmd.angular.z = -float(np.clip(GAIN_ANGLE * angle_deg,
                                       -MAX_ANGULAR, MAX_ANGULAR))
        _pulse(pub, cmd)

    pub.publish(Twist())
    rospy.loginfo("align_water_horizontal complete.")