#!/usr/bin/env python3
"""
sign_debug.py
-------------
Continuously prints blue sign top-edge angle and centroid error.
Press Enter to trigger the full alignment sequence.

Usage:
    rosrun <your_package> sign_debug.py
"""

import math
import time
import threading
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

CAM_TOPIC = "/B1/robot/camera_top/image_raw"
CMD_TOPIC = "/B1/cmd_vel"

# ------------------------------------------------------------------ #
# Configuration — match these to align_to_sign() call               #
# ------------------------------------------------------------------ #
TARGET_X_RATIO = 0.1
TARGET_Y_RATIO = 0.2

CROP_TOP    = 0.0
CROP_BOTTOM = 0.0
CROP_LEFT   = 0.0
CROP_RIGHT  = 0.0

TOP_EDGE_BAND = 5

GAIN_ANGLE      = 0.20
GAIN_ANGLE_FINE = 0.10
GAIN_FORWARD    = 0.004
GAIN_LATERAL    = 0.004
THRESH_ANGLE    = 2.0
THRESH_PIXEL_Y  = 10
THRESH_PIXEL_X  = 15
MAX_ANGULAR     = 1.0
MAX_LINEAR      = 0.3
ALIGN_HZ        = 20

# ------------------------------------------------------------------ #
bridge       = CvBridge()
latest_image = None
pub          = None
aligning     = False


def image_callback(msg):
    global latest_image
    img          = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    latest_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def get_image():
    if latest_image is None:
        return None
    img  = latest_image.copy()
    h, w = img.shape[:2]
    t    = int(h * CROP_TOP);    b = h - int(h * CROP_BOTTOM)
    l    = int(w * CROP_LEFT);   r = w - int(w * CROP_RIGHT)
    return img[t:b, l:r]


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


def target_in_crop(img_w, img_h):
    """Return (target_x, target_y) in cropped image coordinates."""
    img_h_full = img_h / (1.0 - CROP_TOP - CROP_BOTTOM) \
                 if (CROP_TOP + CROP_BOTTOM) < 1.0 else img_h
    img_w_full = img_w / (1.0 - CROP_LEFT - CROP_RIGHT) \
                 if (CROP_LEFT + CROP_RIGHT) < 1.0 else img_w
    ty = int(TARGET_Y_RATIO * img_h_full - CROP_TOP  * img_h_full)
    tx = int(TARGET_X_RATIO * img_w_full - CROP_LEFT * img_w_full)
    return tx, ty


def pulse(cmd):
    pub.publish(cmd)
    rospy.sleep(1.0 / ALIGN_HZ)
    pub.publish(Twist())


def run_alignment():
    global aligning
    aligning = True
    deadline = time.time() + 30.0
    print("\n>>> Alignment started <<<")

    # ---- Step 1: Coarse rotation ---- #
    print("  Step 1: coarse rotation...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = get_image()
        if frame is None: continue
        result = detect_sign_top_edge(frame)
        if result is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        angle_deg, cx, cy, img_w, img_h = result
        if abs(angle_deg) < THRESH_ANGLE:
            print("  Step 1 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(GAIN_ANGLE * angle_deg, -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub.publish(Twist())
    rospy.sleep(0.3)

    # ---- Step 2: Translation ---- #
    print("  Step 2: translating...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = get_image()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        result = detect_sign_top_edge(frame)
        if result is None:
            print("  Sign lost."); break
        angle_deg, cx, cy, img_w, img_h = result
        tx, ty  = target_in_crop(img_w, img_h)
        x_err   = cx - tx
        y_err   = cy - ty
        if (abs(y_err) < THRESH_PIXEL_Y and
                abs(x_err) < THRESH_PIXEL_X and
                abs(angle_deg) < THRESH_ANGLE):
            print("  Step 2 done.")
            break
        cmd = Twist()
        cmd.linear.x  = float(np.clip(-GAIN_FORWARD * y_err, -MAX_LINEAR, MAX_LINEAR))
        cmd.linear.y  = float(np.clip(-GAIN_LATERAL * x_err, -MAX_LINEAR, MAX_LINEAR))
        cmd.angular.z = float(np.clip(-GAIN_ANGLE   * angle_deg, -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub.publish(Twist())
    rospy.sleep(0.3)

    # ---- Step 3: Fine rotation ---- #
    print("  Step 3: fine rotation...")
    while time.time() < deadline and not rospy.is_shutdown():
        frame = get_image()
        if frame is None:
            rospy.sleep(1.0 / ALIGN_HZ); continue
        result = detect_sign_top_edge(frame)
        if result is None:
            print("  Sign lost."); break
        angle_deg, cx, cy, img_w, img_h = result
        if abs(angle_deg) < THRESH_ANGLE:
            print("  Step 3 done.")
            break
        cmd = Twist()
        cmd.angular.z = -float(np.clip(GAIN_ANGLE_FINE * angle_deg, -MAX_ANGULAR, MAX_ANGULAR))
        pulse(cmd)

    pub.publish(Twist())
    print(">>> Alignment complete <<<\n")
    aligning = False


def input_thread():
    """Wait for Enter key to trigger alignment."""
    while not rospy.is_shutdown():
        input()
        if not aligning:
            threading.Thread(target=run_alignment, daemon=True).start()
        else:
            print("  (alignment already running)")


def main():
    global pub
    rospy.init_node("sign_debug", anonymous=True)
    pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
    rospy.Subscriber(CAM_TOPIC, Image, image_callback)
    rospy.sleep(1.0)

    print(f"Target: x_ratio={TARGET_X_RATIO}  y_ratio={TARGET_Y_RATIO}")
    print(f"Crop:   top={CROP_TOP}  bottom={CROP_BOTTOM}  "
          f"left={CROP_LEFT}  right={CROP_RIGHT}")
    print("Press Enter to align. Ctrl+C to quit.")
    print("-" * 62)
    print(f"{'angle_deg':>10}  {'cx':>6}  {'cy':>6}  {'x_err':>7}  {'y_err':>7}")
    print("-" * 62)

    threading.Thread(target=input_thread, daemon=True).start()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if not aligning:
            frame = get_image()
            if frame is None:
                print("  (no image)")
            else:
                result = detect_sign_top_edge(frame)
                if result is None:
                    print("  (sign not detected)")
                else:
                    angle_deg, cx, cy, img_w, img_h = result
                    tx, ty  = target_in_crop(img_w, img_h)
                    x_err   = cx - tx
                    y_err   = cy - ty
                    print(f"{angle_deg:>+10.2f}  {cx:>6}  {cy:>6}  "
                          f"{x_err:>+7}  {y_err:>+7}")
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass