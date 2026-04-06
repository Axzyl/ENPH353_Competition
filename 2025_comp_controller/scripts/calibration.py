#!/usr/bin/env python3
"""
calibrate.py
------------
Automatically finds the conversion from distance/angle to time using
odom feedback and binary search.

For each calibration target (e.g. move 0.5m, 1.0m, 1.5m), the script:
  1. Respawns the robot at the start pose
  2. Drives for a trial duration
  3. Measures actual displacement from /B1/odom
  4. Adjusts duration via binary search until odom matches the target
  5. Records (target, calibrated_duration) pair

After all targets are done, fits a linear model:
    time = SLOPE * distance + INTERCEPT
    time = SLOPE * angle    + INTERCEPT  (for turns)

and prints the result + a ready-to-use go_forward() / turn() function.

Usage:
    rosrun <your_package> calibrate.py
"""

import math
import time
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

# ================================================================== #
# CONFIGURATION                                                        #
# ================================================================== #

CMD_TOPIC  = "/B1/cmd_vel"
ODOM_TOPIC = "/B1/odom"
MODEL_NAME = "B1"

# Fixed speeds (match what you use in your controller)
FORWARD_SPEED = 0.5   # m/s
TURN_SPEED    = 2.0   # rad/s

# Calibration targets
# Linear: distances in metres to calibrate against
LINEAR_TARGETS_M = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# Angular: angles in degrees to calibrate against
ANGULAR_TARGETS_DEG = [45, 90, 135, 180, 270]

# Binary search parameters
BINARY_SEARCH_TOL   = 0.005   # metres (or degrees) — stop when this close
BINARY_SEARCH_ITERS = 20      # max iterations per target
TIME_MIN            = 0.05    # seconds — minimum trial duration
TIME_MAX            = 30.0    # seconds — maximum trial duration

PUBLISH_HZ = 10
SETTLE_TIME = 0.3   # seconds to wait after stopping before reading odom

# Start pose for respawn (update to match your spawn point)
START_X   = 0
START_Y   = 2.3
START_Z   = 0.04
START_QW  = 1.0


 
# ================================================================== #
# ODOM                                                                 #
# ================================================================== #
 
odom_x   = None
odom_y   = None
odom_yaw = None
 
 
def quaternion_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)
 
 
def odom_callback(msg):
    global odom_x, odom_y, odom_yaw
    odom_x   = msg.pose.pose.position.x
    odom_y   = msg.pose.pose.position.y
    odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
 
 
def wait_odom():
    while (odom_x is None) and not rospy.is_shutdown():
        rospy.sleep(0.05)
 
 
def snapshot():
    return odom_x, odom_y, odom_yaw
 
 
# ================================================================== #
# GAZEBO RESPAWN                                                       #
# ================================================================== #
 
def respawn():
    msg = ModelState()
    msg.model_name          = MODEL_NAME
    msg.pose.position.x     = START_X
    msg.pose.position.y     = START_Y
    msg.pose.position.z     = START_Z
    msg.pose.orientation.w  = START_QW
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        svc = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        svc(msg)
    except rospy.ServiceException as e:
        rospy.logerr("Respawn failed: %s", e)
    rospy.sleep(0.5)   # let physics settle
 
 
# ================================================================== #
# MOVEMENT                                                             #
# ================================================================== #
 
def drive(pub, lin_x, ang_z, duration):
    """Drive for exactly `duration` seconds then stop."""
    cmd = Twist()
    cmd.linear.x  = lin_x
    cmd.angular.z = ang_z
    rate     = rospy.Rate(PUBLISH_HZ)
    deadline = rospy.Time.now() + rospy.Duration(duration)
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        pub.publish(cmd)
        rate.sleep()
    pub.publish(Twist())
    rospy.sleep(SETTLE_TIME)
 
 
# ================================================================== #
# BINARY SEARCH CALIBRATION                                            #
# ================================================================== #
 
def calibrate_linear(pub, target_m):
    """
    Binary search for the time that makes the robot travel exactly
    `target_m` metres forward.
    Returns calibrated duration in seconds.
    """
    naive      = target_m / FORWARD_SPEED
    lo         = max(TIME_MIN, naive * 0.5)
    hi         = min(TIME_MAX, naive * 2.0)
    best_time  = naive
    best_error = float('inf')
 
    for iteration in range(BINARY_SEARCH_ITERS):
        trial_time = (lo + hi) / 2.0
 
        respawn()
        wait_odom()
        x0, y0, _ = snapshot()
 
        drive(pub, FORWARD_SPEED, 0.0, trial_time)
 
        x1, y1, _ = snapshot()
        actual_m   = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        error      = actual_m - target_m
 
        rospy.loginfo("  [linear] target=%.3fm  trial=%.4fs  actual=%.4fm  err=%+.4fm",
                      target_m, trial_time, actual_m, error)
 
        if abs(error) < abs(best_error):
            best_error = error
            best_time  = trial_time
 
        if abs(error) < BINARY_SEARCH_TOL:
            rospy.loginfo("  Converged in %d iterations.", iteration + 1)
            break
 
        if error < 0:
            lo = trial_time   # went too short → increase time
        else:
            hi = trial_time   # went too far  → decrease time
 
    return best_time
 
 
def calibrate_angular(pub, target_deg):
    """
    Binary search for the time that makes the robot turn exactly
    `target_deg` degrees CCW.
    Returns calibrated duration in seconds.
    """
    target_rad = math.radians(target_deg)
    naive      = target_rad / TURN_SPEED
    lo         = max(TIME_MIN, naive * 0.5)
    hi         = min(TIME_MAX, naive * 2.0)
    best_time  = naive
    best_error = float('inf')
 
    for iteration in range(BINARY_SEARCH_ITERS):
        trial_time = (lo + hi) / 2.0
 
        respawn()
        wait_odom()
        _, _, yaw0 = snapshot()
 
        drive(pub, 0.0, TURN_SPEED, trial_time)
 
        _, _, yaw1  = snapshot()
        actual_rad  = yaw1 - yaw0
        # Normalise to [-pi, pi]
        while actual_rad >  math.pi: actual_rad -= 2 * math.pi
        while actual_rad < -math.pi: actual_rad += 2 * math.pi
        actual_deg = math.degrees(actual_rad)
        error      = actual_deg - target_deg
 
        rospy.loginfo("  [angular] target=%.1fdeg  trial=%.4fs  actual=%.2fdeg  err=%+.2fdeg",
                      target_deg, trial_time, actual_deg, error)
 
        if abs(error) < abs(best_error):
            best_error = error
            best_time  = trial_time
 
        if abs(error) < BINARY_SEARCH_TOL:
            rospy.loginfo("  Converged in %d iterations.", iteration + 1)
            break
 
        if error < 0:
            lo = trial_time
        else:
            hi = trial_time
 
    return best_time
 
 
# ================================================================== #
# LINEAR FIT                                                           #
# ================================================================== #
 
def linear_fit(xs, ys):
    """Fit y = slope * x + intercept via least squares."""
    n     = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom  = n * sum_xx - sum_x ** 2
    slope     = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    # R²
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2
 
 
# ================================================================== #
# OUTPUT                                                               #
# ================================================================== #
 
def print_results(linear_data, angular_data,
                  lin_slope, lin_intercept, lin_r2,
                  ang_slope, ang_intercept, ang_r2):
    print("\n" + "=" * 60)
    print("  CALIBRATION RESULTS")
    print("=" * 60)
 
    print("\nLinear (forward) calibration:")
    print(f"  {'Target (m)':<14} {'Calibrated time (s)'}")
    for target, t in linear_data:
        naive = target / FORWARD_SPEED
        print(f"  {target:<14.3f} {t:.4f}s  (naive={naive:.4f}s)")
    print(f"\n  Fit:  time = {lin_slope:.6f} * distance + {lin_intercept:.6f}")
    print(f"  R² = {lin_r2:.6f}")
 
    print("\nAngular (turn CCW) calibration:")
    print(f"  {'Target (deg)':<14} {'Calibrated time (s)'}")
    for target, t in angular_data:
        naive = math.radians(target) / TURN_SPEED
        print(f"  {target:<14.1f} {t:.4f}s  (naive={naive:.4f}s)")
    print(f"\n  Fit:  time = {ang_slope:.6f} * angle_deg + {ang_intercept:.6f}")
    print(f"  R² = {ang_r2:.6f}")
 
    print("\n" + "=" * 60)
    print("  READY-TO-USE FUNCTIONS (copy into movement.py)")
    print("=" * 60)
    print(f"""
# Auto-calibrated at FORWARD_SPEED={FORWARD_SPEED} m/s, TURN_SPEED={TURN_SPEED} rad/s
# Linear fit R²={lin_r2:.4f},  Angular fit R²={ang_r2:.4f}
 
LIN_SLOPE     = {lin_slope:.6f}
LIN_INTERCEPT = {lin_intercept:.6f}
ANG_SLOPE     = {ang_slope:.6f}
ANG_INTERCEPT = {ang_intercept:.6f}
 
def go_forward(distance, speed={FORWARD_SPEED}):
    \"\"\"Drive forward by `distance` metres (negative = backward).\"\"\"
    sign     = 1 if distance >= 0 else -1
    duration = LIN_SLOPE * abs(distance) + LIN_INTERCEPT
    cmd = Twist()
    cmd.linear.x = sign * speed
    rate = rospy.Rate(10)
    end  = rospy.Time.now() + rospy.Duration(duration)
    while rospy.Time.now() < end and not rospy.is_shutdown():
        pub.publish(cmd)
        rate.sleep()
    pub.publish(Twist())
 
def turn(angle_deg, speed={TURN_SPEED}):
    \"\"\"Turn by `angle_deg` degrees (positive=CCW, negative=CW).\"\"\"
    sign     = 1 if angle_deg >= 0 else -1
    duration = ANG_SLOPE * abs(angle_deg) + ANG_INTERCEPT
    cmd = Twist()
    cmd.angular.z = sign * speed
    rate = rospy.Rate(10)
    end  = rospy.Time.now() + rospy.Duration(duration)
    while rospy.Time.now() < end and not rospy.is_shutdown():
        pub.publish(cmd)
        rate.sleep()
    pub.publish(Twist())
""")
    print("=" * 60)
 
 
# ================================================================== #
# MAIN                                                                 #
# ================================================================== #
 
def main():
    rospy.init_node("calibrate", anonymous=True)
    pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
    rospy.Subscriber(ODOM_TOPIC, Odometry, odom_callback)
    rospy.sleep(1.0)
    wait_odom()
 
    rospy.loginfo("Starting calibration.")
    rospy.loginfo("Linear targets: %s m", LINEAR_TARGETS_M)
    rospy.loginfo("Angular targets: %s deg", ANGULAR_TARGETS_DEG)
 
    # ---- Linear calibration ---- #
    linear_data = []
    for target in LINEAR_TARGETS_M:
        rospy.loginfo("Calibrating linear: %.3f m …", target)
        t = calibrate_linear(pub, target)
        linear_data.append((target, t))
        rospy.loginfo("  Result: %.3f m → %.4f s", target, t)
 
    # ---- Angular calibration ---- #
    angular_data = []
    for target in ANGULAR_TARGETS_DEG:
        rospy.loginfo("Calibrating angular: %.1f deg …", target)
        t = calibrate_angular(pub, target)
        angular_data.append((target, t))
        rospy.loginfo("  Result: %.1f deg → %.4f s", target, t)
 
    # ---- Fit linear models ---- #
    lin_xs = [d[0] for d in linear_data]
    lin_ys = [d[1] for d in linear_data]
    lin_slope, lin_intercept, lin_r2 = linear_fit(lin_xs, lin_ys)
 
    ang_xs = [d[0] for d in angular_data]
    ang_ys = [d[1] for d in angular_data]
    ang_slope, ang_intercept, ang_r2 = linear_fit(ang_xs, ang_ys)
 
    print_results(linear_data, angular_data,
                  lin_slope, lin_intercept, lin_r2,
                  ang_slope, ang_intercept, ang_r2)
 
 
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
 