#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import adjustment
import range_sensors as rs
import sign_reader
import sign_ui
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import numpy as np
 
TOPIC        = "/B1/cmd_vel"
CLUE_TOPIC   = "/score_tracker"
FWD_SPEED    = 0.5   # m/s
TURN_SPEED   = 2.0   # rad/s
 
pub      = None
clue_pub = None
clue_id  = 0

def spawn(x, y, yaw_rad):
    half = yaw_rad / 2.0
    msg  = ModelState()
    msg.model_name          = "B1"
    msg.pose.position.x     = x
    msg.pose.position.y     = y
    msg.pose.position.z     = 0.05
    msg.pose.orientation.z  = math.sin(half)
    msg.pose.orientation.w  = math.cos(half)
    rospy.wait_for_service("/gazebo/set_model_state")
    rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)(msg)
    rospy.sleep(0.5)
 
 
def go_forward(duration, speed_factor=1.0):
    cmd = Twist()
    cmd.linear.x = FWD_SPEED*speed_factor
    end  = rospy.Time.now() + rospy.Duration(duration) / speed_factor
    rate = rospy.Rate(20)
    while rospy.Time.now() < end:
        pub.publish(cmd)
        rate.sleep()
    pub.publish(Twist())
 
 
def turn(duration, clockwise=False, speed_factor=1.0):
    cmd = Twist()
    cmd.angular.z = -TURN_SPEED * speed_factor if clockwise else TURN_SPEED * speed_factor
    end  = rospy.Time.now() + rospy.Duration(duration) / speed_factor
    rate = rospy.Rate(20)
    while rospy.Time.now() < end:
        pub.publish(cmd)
        rate.sleep()
    pub.publish(Twist())
 
 
def go_forward_until(sensor, condition, value,
                     speed=None, timeout=30.0):
    """
    Drive forward until a range sensor satisfies a condition, then stop.
 
    Args:
        sensor    : "center", "left", or "right"
        condition : "above" or "below"
        value     : threshold distance in metres
        speed     : forward speed (default FWD_SPEED)
        timeout   : max seconds before giving up (default 30)
 
    Returns:
        True  if condition met, False if timed out
    """
    if condition not in ("above", "below"):
        raise ValueError(f"condition must be 'above' or 'below', got '{condition}'")
    if speed is None:
        speed = FWD_SPEED
 
    rospy.loginfo("go_forward_until: %s %s %.3fm @ %.2fm/s (timeout=%.1fs)",
                  sensor, condition, value, speed, timeout)
 
    cmd = Twist()
    cmd.linear.x = speed
 
    rate     = rospy.Rate(20)
    deadline = rospy.Time.now() + rospy.Duration(timeout)
 
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        pub.publish(cmd)
        dist = rs.read(sensor)
        if dist is not None:
            met = dist < value if condition == "below" else dist > value
            rospy.loginfo("  %s: %.3f m", sensor, dist)
            if met:
                rospy.loginfo("  Condition met — stopping.")
                pub.publish(Twist())
                return True
        rate.sleep()
 
    pub.publish(Twist())
    rospy.logwarn("go_forward_until: timed out.")
    return False

# ------------------------------------------------------------------ #
# Wall alignment via range sensor minimum                              #
# ------------------------------------------------------------------ #
 
def align_to_wall(sensor="left", right=False, turn_speed=None, tolerance=0.002, adjust_tolerance=0.001, timeout=30.0):
    """
    Turn the robot until the given range sensor reads its minimum value,
    indicating the robot is perpendicular to the nearest wall.
 
    Algorithm:
      1. Turn until the sensor value starts increasing (just passed minimum)
      2. Record the minimum value seen
      3. Turn back (opposite direction) until sensor reads within
         `tolerance` metres of that minimum
 
    Args:
        sensor      : "left", "center", or "right"
        turn_speed  : angular speed (default TURN_SPEED)
        tolerance   : how close to the minimum to stop (metres, default 0.02)
        timeout     : max seconds (default 30)
 
    Returns:
        True if aligned, False if timed out
    """
    if turn_speed is None:
        turn_speed = TURN_SPEED if not right else -TURN_SPEED
 
    rospy.loginfo("align_to_wall: sensor=%s  tolerance=%.3fm", sensor, tolerance)
 
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    rate     = rospy.Rate(40)
 
    # ---- Step 1: Turn until sensor value starts increasing ---- #
    rospy.loginfo("  Step 1: turning to find minimum...")
 
    # Initial turn direction — CCW
    cmd_turn = Twist()
    cmd_turn.angular.z = turn_speed
 
    prev_dist  = rs.read(sensor)
    while prev_dist is None and rospy.Time.now() < deadline:
        rate.sleep()
        prev_dist = rs.read(sensor)
 
    min_dist   = prev_dist
    increasing = False
 
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        pub.publish(cmd_turn)
        rate.sleep()
 
        dist = rs.read(sensor)
        if dist is None:
            continue
 
        rospy.loginfo("  %s: %.3fm  (min=%.3fm)", sensor, dist, min_dist)
 
        if dist < min_dist:
            min_dist = dist
 
        # Detect when value has started increasing past the minimum
        if dist > min_dist + tolerance:
            rospy.loginfo("  Minimum passed. min=%.3fm  current=%.3fm",
                          min_dist, dist)
            increasing = True
            break
 
        prev_dist = dist
 
    pub.publish(Twist())
    rospy.sleep(0.2)
 
    if not increasing:
        rospy.logwarn("align_to_wall: timed out in step 1.")
        return False
 
    # ---- Step 2: Turn back until sensor is within tolerance of minimum ---- #
    rospy.loginfo("  Step 2: turning back to min=%.3fm...", min_dist)
 
    cmd_back = Twist()
    cmd_back.angular.z = -turn_speed / 2   # opposite direction
 
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        pub.publish(cmd_back)
        rate.sleep()
 
        dist = rs.read(sensor)
        if dist is None:
            continue
 
        rospy.loginfo("  %s: %.3fm  (target=%.3fm)", sensor, dist, min_dist)
 
        if dist <= min_dist + tolerance:
            rospy.loginfo("  Aligned. dist=%.3fm", dist)
            break
 
    pub.publish(Twist())
    rospy.loginfo("align_to_wall complete.")
    return True

# ------------------------------------------------------------------ #
# Wall following                                                        #
# ------------------------------------------------------------------ #
 
def follow_wall(follow_sensor="right",
                stop_sensor="left",
                target_dist=0.5,
                stop_dist_min=0.3,
                stop_dist_max=0.8,
                fwd_speed=None,
                turn_gain=10,
                timeout=60.0):
    """
    Drive forward while keeping `follow_sensor` at target_dist from
    the wall, correcting heading with proportional angular control.
    Stops when `stop_sensor` reads within [stop_dist_min, stop_dist_max].
 
    Args:
        follow_sensor : sensor used for wall following ("left", "center", "right")
        stop_sensor   : sensor used for stop condition ("left", "center", "right")
        target_dist   : desired distance from wall (metres)
        stop_dist_min : stop sensor lower bound to trigger stop (metres)
        stop_dist_max : stop sensor upper bound to trigger stop (metres)
        fwd_speed     : forward speed (default FWD_SPEED)
        turn_gain     : proportional gain for angular correction
        timeout       : max seconds (default 60)
 
    Returns:
        True  if stopped because stop sensor detected something
        False if timed out
    """
    if fwd_speed is None:
        fwd_speed = FWD_SPEED
 
    rospy.loginfo("follow_wall: follow=%s  stop=%s  target=%.2fm  stop=[%.2f, %.2f]m",
                  follow_sensor, stop_sensor, target_dist, stop_dist_min, stop_dist_max)
 
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    rate     = rospy.Rate(20)
 
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        follow_dist = rs.read(follow_sensor)
        stop_dist   = rs.read(stop_sensor)
 
        # Check stop condition
        if stop_dist is not None and stop_dist_min <= stop_dist <= stop_dist_max:
            pub.publish(Twist())
            rospy.loginfo("follow_wall: %s=%.3fm — stopping.", stop_sensor, stop_dist)
            return True
 
        # Wall following correction
        cmd = Twist()
        cmd.linear.x = fwd_speed
 
        if follow_dist is not None:
            error = follow_dist - target_dist
            # Positive error: too far from wall → turn toward wall
            # Negative error: too close to wall → turn away from wall
            # For right sensor: toward wall = CW = negative z
            # For left sensor:  toward wall = CCW = positive z
            sign = -1.0 if follow_sensor == "right" else 1.0
            cmd.angular.z = float(np.clip(sign * turn_gain * error,
                                          -TURN_SPEED, TURN_SPEED))
            rospy.loginfo("  %s=%.3fm  err=%+.3f  wz=%+.2f  %s=%s",
                          follow_sensor, follow_dist, error, cmd.angular.z,
                          stop_sensor,
                          f"{stop_dist:.3f}m" if stop_dist else "—")
        else:
            rospy.logwarn("  %s sensor unavailable", follow_sensor)
 
        pub.publish(cmd)
        rate.sleep()
 
    pub.publish(Twist())
    rospy.logwarn("follow_wall: timed out.")
    return False

# ------------------------------------------------------------------ #
# Single-shot sign reading                                             #
# ------------------------------------------------------------------ #
 
_bridge = CvBridge()
 
CAMERA_TOPICS = {
    "left":   "/B1/robot/camera_fl/image_raw",
    "center": "/B1/rrbot/camera1/image_raw",
    "right":  "/B1/robot/camera_fr/image_raw",
}
 
 
def read_sign(camera="center", timeout=5.0):
    """
    Capture a single frame from the given camera, run it through
    sign_reader, and return the detected text.
 
    Args:
        camera  : "left", "center", or "right"
        timeout : seconds to wait for a frame (default 5.0)
 
    Returns:
        (top_text, bot_text) — both empty strings if no sign found
    """
    global clue_id, clue_pub
 
    if camera not in CAMERA_TOPICS:
        raise ValueError(f"Unknown camera '{camera}'. Choose from: {list(CAMERA_TOPICS)}")
 
    frame_holder = [None]
    frame_count  = [0]
 
    def _cb(msg):
        frame_count[0] += 1
        # Skip the first frame — it may be stale/buffered before we subscribed
        if frame_count[0] > 4:
            frame_holder[0] = _bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
 
    topic = CAMERA_TOPICS[camera]
    sub   = rospy.Subscriber(topic, RosImage, _cb)
 
    # Wait for a fresh frame (second one received)
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    rate     = rospy.Rate(10)
    while frame_holder[0] is None and rospy.Time.now() < deadline:
        rate.sleep()
 
    sub.unregister()
 
    if frame_holder[0] is None:
        rospy.logwarn("read_sign: no frame received from %s within %.1fs", camera, timeout)
        return "", ""
 
    _, sign_roi, top_crop, bot_crop, top_text, bot_text = sign_reader.process_frame(frame_holder[0])
    rospy.loginfo("read_sign [%s]: top='%s'  bot='%s'", camera, top_text, bot_text)
 
    # Push to sign UI
    sign_ui.push(frame_holder[0], sign_roi, top_crop, bot_crop, top_text, bot_text)
 
    clue_id += 1
    clue_pub.publish(f"Team 3,67,{clue_id},{bot_text}")
 
    return top_text, bot_text

# ------------------------------------------------------------------ #
# Pedestrian detection via grass pixel change                          #
# ------------------------------------------------------------------ #

# HSV range for the dark green grass
GRASS_HSV_LO = np.array([35,  60,  30])
GRASS_HSV_HI = np.array([85, 255, 120])

# Fraction of the left half of the image to sample (0.0-1.0)
# Only the left portion is used since the pedestrian comes from the left
GRASS_SAMPLE_X = 0.5

# A change larger than this fraction of the baseline pixel count
# is considered a significant pedestrian event
GRASS_CHANGE_THRESH = 0.03   # 15% change triggers detection

# Number of frames to average for the baseline
GRASS_BASELINE_FRAMES = 10


def wait_for_pedestrian_clear(timeout=30.0):
    """
    Block until the pedestrian has crossed and the left camera view
    is clear (i.e. grass pixel count returns close to baseline).

    Strategy:
      1. Capture a baseline grass pixel count over several frames
      2. Wait until the count drops significantly (pedestrian entered)
      3. Wait until the count recovers (pedestrian has passed)
      4. Return so the robot can proceed

    Args:
        timeout : maximum seconds to wait (default 30)

    Returns:
        True  if pedestrian passed and view is clear
        False if timed out
    """
    import cv2
    from cv_bridge import CvBridge as _CvBridge

    bridge_local = _CvBridge()
    topic        = CAMERA_TOPICS["left"]
    latest       = [None]
    frame_count  = [0]

    def _cb(msg):
        
        frame_count[0] += 1
        # Skip the first frame — it may be stale/buffered before we subscribed
        if frame_count[0] > 4:
            latest[0] = bridge_local.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    sub = rospy.Subscriber(topic, RosImage, _cb)
    rate = rospy.Rate(20)

    def count_grass(frame):
        """Count dark-green grass pixels in the left half of the frame."""
        h, w = frame.shape[:2]
        roi  = frame[:, :int(w * GRASS_SAMPLE_X)]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                            np.array(GRASS_HSV_LO),
                            np.array(GRASS_HSV_HI))
        return int(np.sum(mask > 0))

    # ---- Step 1: Establish baseline ---- #
    rospy.loginfo("wait_for_pedestrian_clear: building baseline...")
    baseline_counts = []
    deadline = rospy.Time.now() + rospy.Duration(timeout)

    while len(baseline_counts) < GRASS_BASELINE_FRAMES:
        if rospy.Time.now() > deadline or rospy.is_shutdown():
            sub.unregister()
            rospy.logwarn("wait_for_pedestrian_clear: timed out during baseline.")
            return False
        if latest[0] is not None:
            baseline_counts.append(count_grass(latest[0]))
            latest[0] = None
        rate.sleep()

    baseline = sum(baseline_counts) / len(baseline_counts)
    threshold = baseline * GRASS_CHANGE_THRESH
    rospy.loginfo("  baseline=%.0f px  change_threshold=%.0f px",
                  baseline, threshold)

    # ---- Step 2: Wait for pedestrian to enter (count drops) ---- #
    rospy.loginfo("  Waiting for pedestrian to enter...")
    pedestrian_seen = False
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        if latest[0] is not None:
            count = count_grass(latest[0])
            latest[0] = None
            rospy.loginfo("  grass px: %d  (baseline %.0f)", count, baseline)
            if abs(count - baseline) > threshold:
                rospy.loginfo("  Pedestrian detected (count=%d).", count)
                pedestrian_seen = True
                break
        rate.sleep()

    if not pedestrian_seen:
        sub.unregister()
        rospy.logwarn("wait_for_pedestrian_clear: timed out waiting for pedestrian.")
        return False

    # ---- Step 3: Wait for count to recover (pedestrian has passed) ---- #
    rospy.loginfo("  Waiting for pedestrian to clear...")
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        if latest[0] is not None:
            count = count_grass(latest[0])
            latest[0] = None
            rospy.loginfo("  grass px: %d  (baseline %.0f)", count, baseline)
            if abs(count - baseline) <= threshold:
                rospy.loginfo("  Pedestrian cleared. Safe to proceed.")
                sub.unregister()
                return True
        rate.sleep()

    sub.unregister()
    rospy.logwarn("wait_for_pedestrian_clear: timed out waiting for pedestrian to clear.")
    return False


def main():
    global pub, clue_pub, clue_id
    rospy.init_node("temp_move", anonymous=True)
    pub = rospy.Publisher(TOPIC, Twist, queue_size=1)
    clue_pub = rospy.Publisher(CLUE_TOPIC, String, queue_size=1)
    clue_id = 0
    adjustment._ensure_camera()
    rs.init()
    sign_ui.init()
    rospy.sleep(1.0)
    pub.publish(Twist())
    rs.read_all()
    
    starting_section = 1
    
    if starting_section == 1:
        spawn(5.5, 2.5, -1.57)
        clue_id = 0
    if starting_section == 2:
        spawn(0.4913, -0.0903, 1.56)
        clue_id = 4
    if starting_section == 3:
        clue_id = 6
        spawn(-3.904708, 0.598573, -3.110783)
    if starting_section == 4:
        clue_id = 7
        spawn(-4.278303, -2.352247, 0)
    if starting_section == 5:
        align_to_wall(sensor="center")
        return
    
    section = starting_section

    rospy.sleep(1)
    
    clue_pub.publish("Team 3,67,0,idk")

    if section == 1:
        
        go_forward(1, speed_factor=2.0)
        
        adjustment.align_to_sign(pub, target_x_ratio=0.2, target_y_ratio=0.1)
        
        top, bot = read_sign("left")
        print(f"{top}: {bot}")
                
        go_forward(1)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        
        go_forward(0.2)
        
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)  
        go_forward(0.62, speed_factor=2.0)
        
        # go_forward(0.1)
        
        adjustment.align_to_line(pub, color="red", target_y_ratio=0.5, crop_top=0.2)
        
        wait_for_pedestrian_clear(timeout=8)
        rospy.sleep(0.5)
        
        go_forward_until("center", "below", 2.225, speed=1.0)
        go_forward_until("center", "above", 2.225, speed=-0.2)
        go_forward_until("center", "below", 2.225, speed=0.2)
        
        turn(1, clockwise=False)
        align_to_wall(sensor="center")
        
        go_forward_until("center", "below", 0.45, speed=1.0)
        go_forward_until("center", "above", 0.45, speed=-0.2)
        go_forward_until("center", "below", 0.45, speed=0.2)
        
        turn(1.45, clockwise=True)
        
        go_forward(1, speed_factor=2)

        # go_forward(1.8, speed_factor=2.0)
        
        # go_forward(0.5)       
        # turn(0.7, clockwise=False)             
        # go_forward(0.52)       
        # turn(0.7, clockwise=False)  
        # go_forward(0.62)
        
        # go_forward(0.1)
        
        # go_forward(0.5)       
        # turn(0.7, clockwise=True)             
        # go_forward(0.52)       
        # turn(0.7, clockwise=True)  
        # go_forward(0.62, speed_factor=2.0)
        
        # go_forward(0.3, speed_factor=2.0)
        
        adjustment.align_to_sign(pub, target_y_ratio=0.1)
        top, bot = read_sign("right")
        print(f"{top}: {bot}")
                
        go_forward_until("center", "below", 0.38, speed=1.0)
        go_forward_until("center", "above", 0.38, speed=-0.2)
        go_forward_until("center", "below", 0.38, speed=0.2)
        turn(1.4, clockwise=True)
        go_forward(2.1)
        turn(1.4, clockwise=False)
        go_forward_until("center", "above", 1.24, speed=-0.5)
        go_forward_until("center", "below", 1.24, speed=0.2)
        go_forward_until("center", "above", 1.24, speed=-0.2)
        turn(1.4, clockwise=True)
        
        # go_forward(0.5)       
        # turn(0.7, clockwise=True)             
        # go_forward(0.52)       
        # turn(0.7, clockwise=True)  
        # go_forward(0.62)
        # go_forward(0.6)       
        # turn(0.7, clockwise=True)             
        # go_forward(0.52)       
        # turn(0.7, clockwise=True)  
        # go_forward(0.62)
        # go_forward(0.6)       
        # turn(0.7, clockwise=False)             
        # go_forward(0.52)       
        # turn(0.7, clockwise=False)

        adjustment.align_to_sign(pub, target_y_ratio=0.1)
        top, bot = read_sign("left")
        print(f"{top}: {bot}")
        
        go_forward(0.62)
        
        rs.wait_until("center", "below", 0.5, timeout=20)
        rospy.sleep(1)
        
        go_forward(1.2)
        turn(1.4, clockwise=False)
        go_forward(0.9) 
        
        go_forward(0.5)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        
        go_forward(3)
        turn(1.4, clockwise=False)
        go_forward_until("center", "above", 1.24, speed=-0.5)
        go_forward_until("center", "below", 1.24, speed=0.2)
        go_forward_until("center", "above", 1.24, speed=-0.2)
        turn(1, clockwise=True)
        align_to_wall(sensor="center", right=True)
        
        # go_forward(0.5)       
        # turn(0.7, clockwise=True)             
        # go_forward(0.52)       
        # turn(0.7, clockwise=True)  
        # go_forward(0.62)  
        # go_forward(0.9)
        # turn(1.35, clockwise=False)
        
        
        go_forward(1, speed_factor=2.0)  
        
        go_forward_until("center", "below", 0.525)     
        go_forward_until("center", "above", 0.525, speed=-0.2)   
        go_forward_until("center", "below", 0.525, speed=0.2)    
        turn(1.4, clockwise=True)             
        
        go_forward(0.8, speed_factor=2.0)
        
        adjustment.align_to_sign(pub, target_y_ratio=0.1)
        top, bot = read_sign("right")
        print(f"{top}: {bot}")
        
        go_forward(1.67)
        adjustment.align_to_line(pub, color="magenta", target_y_ratio=0.5, crop_top=0.0)
        
        section += 1
        
    if(section == 2):
        
        go_forward(0.2)
        go_forward(0.5)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        go_forward(2.2, speed_factor=2.0)
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.75, clockwise=False)  
        go_forward(0.57)
        go_forward(2.0, speed_factor=2.0)
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)  
        go_forward(0.57)
        
        # go_forward(2.1, speed_factor=2.0)
        follow_wall(target_dist=0.4, stop_dist_min=0.6, stop_dist_max=0.9,turn_gain=12)
        
        go_forward(0.6)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.65, clockwise=False)  
        go_forward(0.62, speed_factor=2.0)
        
        adjustment.align_to_sign(pub, target_y_ratio=0.1)
        top, bot = read_sign("left")
        print(f"{top}: {bot}")
        
        go_forward(0.1)
        go_forward(0.5) 
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)
        go_forward(0.72)
                
        # Bridge
        adjustment.align_water_horizontal(pub)
        adjustment.align_between_water(pub, target_y_ratio=0.4)
        
        turn(0.3, clockwise=True)  
        go_forward(0.8)
        turn(0.3, clockwise=True)  
        go_forward(1.2)
        turn(0.95, clockwise=False) 
        go_forward(3.2)
        turn(0.4, clockwise=True)
        go_forward(1.2)
        adjustment.align_to_sign(pub, target_y_ratio=0.1, crop_bottom=0.3)
        top, bot = read_sign("right")
        print(f"{top}: {bot}")
        
        go_forward(0.2) 
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)  
        go_forward(0.62)
        
        # adjustment.align_to_line(pub, color="magenta", target_y_ratio=0.45, target_vertical=True)
        go_forward_until("center", "below", 0.3)
        turn(1.4, clockwise=True)  
        go_forward(0.5)
        adjustment.align_to_line(pub, color="magenta", target_y_ratio=0.5, crop_top=0.0)
        
        section += 1

    if section == 3:
        
        turn(0.5, clockwise=False)
        rs.wait_until("center", "below", 1.5, timeout=30)
        rospy.sleep(5)
        
        # turn(0.1, clockwise=False)
                
        go_forward(3)
        turn(0.8, clockwise=False)
        go_forward(1.5)
        turn(1.1, clockwise=False)
        go_forward(2.5)
        turn(1, clockwise=True)
        go_forward_until("center", "below", 0.375)
        turn(1.4, clockwise=False)
        
        adjustment.align_to_line(pub, color="magenta", target_y_ratio=0.3, crop_top=0.0)
        top, bot = read_sign("left")
        print(f"{top}: {bot}")
        
        section += 1
        
    if section == 4:
        
        go_forward(4, speed_factor=2.0)
        go_forward_until("left", "above", 0.7, speed=1.0)
        go_forward(0.5)
        turn(1.4, clockwise=False)
        go_forward(2, speed_factor=2.0)
        go_forward_until("left", "above", 0.5, speed=1.0)
        go_forward(0.7)
        turn(1.45, clockwise=False)
        go_forward(2, speed_factor=2.0)
        go_forward_until("left", "above", 0.5, speed=0.75, timeout=1)
        go_forward(0.7)
        turn(1.4, clockwise=False)
        go_forward(1, speed_factor=2.0)
        go_forward_until("left", "above", 1)
        go_forward(0.9)
        turn(1, clockwise=False)
        align_to_wall(sensor="center")
        go_forward_until("center", "below", 0.5, speed=0.75, timeout=1.8)
        # go_forward(1.8, speed_factor=2)
   
        pub.publish(Twist())
        
        top, bot = read_sign("center")
        print(f"{top}: {bot}")
        rospy.sleep(2)
        
    clue_pub.publish("Team 3,67,-1,idk")
     


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass