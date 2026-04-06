#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import adjustment

TOPIC        = "/B1/cmd_vel"
FWD_SPEED    = 0.5   # m/s
TURN_SPEED   = 2.0   # rad/s
 
pub = None
 
 
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
 
 
def go_forward(duration):
    cmd = Twist()
    cmd.linear.x = FWD_SPEED
    end  = rospy.Time.now() + rospy.Duration(duration)
    rate = rospy.Rate(20)
    while rospy.Time.now() < end:
        pub.publish(cmd)
        rate.sleep()
    pub.publish(Twist())
 
 
def turn(duration, clockwise=False):
    cmd = Twist()
    cmd.angular.z = -TURN_SPEED if clockwise else TURN_SPEED
    end  = rospy.Time.now() + rospy.Duration(duration)
    rate = rospy.Rate(20)
    while rospy.Time.now() < end:
        pub.publish(cmd)
        rate.sleep()
    pub.publish(Twist())
 
 
def main():
    global pub
    rospy.init_node("temp_move", anonymous=True)
    pub = rospy.Publisher(TOPIC, Twist, queue_size=1)
    rospy.sleep(1.0)
    
    starting_section = 1
    
    if starting_section == 1:
        spawn(5.5, 2.5, -1.57)
    if starting_section == 2:
        spawn(0.4913, -0.0903, 1.56)
    if starting_section == 3:
        spawn(-3.904708, 0.598573, -3.110783)
    if starting_section == 4:
        spawn(-4.278303, -2.352247, 0)
    
    section = starting_section

    rospy.sleep(1)

    if section == 1:
        
        go_forward(1)
        
        adjustment.align_to_sign(pub, target_x_ratio=0.2, target_y_ratio=0.1)
        
        rospy.sleep(2)
        
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
        go_forward(0.62)
        
        go_forward(0.1)
        
        adjustment.align_to_line(pub, color="red", target_y_ratio=0.5, crop_top=0.2)

        go_forward(1.8)
        
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)  
        go_forward(0.62)
        go_forward(0.1)
        go_forward(0.5)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        
        go_forward(0.3)
        
        adjustment.align_to_sign(pub, target_y_ratio=0.1)
        rospy.sleep(2)
        
        go_forward(1.8)
        
        go_forward(0.5)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        go_forward(0.6)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        go_forward(0.4)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)

        
        adjustment.align_to_sign(pub, target_y_ratio=0.1)
        rospy.sleep(2)
        go_forward(0.62)
        
        # Sense Car
        
        go_forward(1.2)
        turn(1.4, clockwise=False)
        go_forward(0.9) 
        
        go_forward(0.5)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        
        go_forward(2.2)
        
        go_forward(0.5)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)  
        
        go_forward(1)
        turn(1.35, clockwise=False)
        go_forward(1)  
        
        go_forward(0.5)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        
        go_forward(0.4)
        
        adjustment.align_to_sign(pub, target_y_ratio=0.1)
        
        go_forward(1)
        adjustment.align_to_line(pub, color="magenta", target_y_ratio=0.5, crop_top=0.0)
        
        section += 1
    
    # Dirt Section (section 2)
    
    if(section == 2):
        
        go_forward(0.2)
        go_forward(0.5)       
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)  
        go_forward(0.62)
        go_forward(2.2)
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)  
        go_forward(0.62)
        go_forward(1.8)
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)  
        go_forward(0.62)
        go_forward(2.2)
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)  
        go_forward(0.62)
        
        adjustment.align_to_sign(pub, target_y_ratio=0.1)
        go_forward(0.1)
        go_forward(0.5) 
        turn(0.7, clockwise=True)             
        go_forward(0.52)       
        turn(0.7, clockwise=True)
        go_forward(0.62)
                
        # Bridge
        adjustment.align_water_horizontal(pub)
        adjustment.align_between_water(pub, target_y_ratio=0.4)
        
        turn(0.3, clockwise=True)  
        go_forward(0.8)
        turn(0.3, clockwise=True)  
        go_forward(1.2)
        turn(0.95, clockwise=False) 
        go_forward(3)
        turn(0.4, clockwise=True)
        go_forward(1.2)
        adjustment.align_to_sign(pub, target_y_ratio=0.1, crop_bottom=0.3)
        
        go_forward(0.2) 
        go_forward(0.5)       
        turn(0.7, clockwise=False)             
        go_forward(0.52)       
        turn(0.7, clockwise=False)  
        go_forward(0.62)
        
        adjustment.align_to_line(pub, color="magenta", target_y_ratio=0.45, target_vertical=True)
        turn(1.4, clockwise=True)  
        adjustment.align_to_line(pub, color="magenta", target_y_ratio=0.5, crop_top=0.0)
        
        section += 1

    if section == 3:
        
        turn(0.5, clockwise=False)
        go_forward(3)
        turn(0.8, clockwise=False)
        go_forward(1.5)
        turn(1.1, clockwise=False)
        go_forward(2.5)
        turn(1, clockwise=True)
        go_forward(1.8)
        turn(1.4, clockwise=False)
        
        adjustment.align_to_line(pub, color="magenta", target_y_ratio=0.3, crop_top=0.0)
        
        section += 1
        
    if section == 4:
        
        
        go_forward(8)
        turn(1.4, clockwise=False)
        go_forward(6)
        turn(1.4, clockwise=False)
        go_forward(4)
        turn(1.4, clockwise=False)
        go_forward(4)
        turn(1.4, clockwise=False)
        go_forward(1.5)
        turn(0.2, clockwise=False)
        go_forward(0.75)
        
        
    
    # go_forward(0.5)       
    # turn(0.7, clockwise=True)             
    # go_forward(0.52)       
    # turn(0.7, clockwise=True)  
    # go_forward(0.62)
     


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass