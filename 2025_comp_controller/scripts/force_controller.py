#!/usr/bin/env python3
"""
force_controller.py
-------------------
Simple ROS node that moves the robot by publishing geometry_msgs/Wrench
messages to /B1/cmd_force (libgazebo_ros_force.so).

Forces  are in the chassis body frame (metres/s²  × mass).
Torques are in the chassis body frame (rad/s²    × inertia).

Axes:
  +X  forward
  +Y  left
  +Z  up

Usage:
  rosrun <your_package> force_controller.py
Then type a command at the prompt, e.g.:
  forward, back, left, right, up, down, yaw_left, yaw_right, stop
"""

import rospy
from geometry_msgs.msg import Wrench


# ------------------------------------------------------------------ #
# Tuneable parameters                                                  #
# ------------------------------------------------------------------ #
TOPIC       = "/B1/cmd_force"
PUBLISH_HZ  = 50          # how often we resend the current command
FORCE_MAG   = 2.0         # Newtons  – linear thrust
TORQUE_MAG  = 0.5         # N·m      – yaw torque


# Map command name → (fx, fy, fz, tx, ty, tz)
COMMANDS = {
    "forward"  : ( FORCE_MAG,  0,           9.81*20,  0, 0,  0),
    "back"     : (-FORCE_MAG,  0,           9.81*20,  0, 0,  0),
    "left"     : ( 0,          FORCE_MAG,   9.81*20,  0, 0,  0),
    "right"    : ( 0,         -FORCE_MAG,   9.81*20,  0, 0,  0),
    "up"       : ( 0,          0,   FORCE_MAG + 9.81*20,  0, 0,  0),
    "down"     : ( 0,          0,  -FORCE_MAG + 9.81*20,  0, 0,  0),
    "yaw_left" : ( 0,          0,           9.81*20,  0, 0,  TORQUE_MAG),
    "yaw_right": ( 0,          0,           9.81*20,  0, 0, -TORQUE_MAG),
    "stop"     : ( 0,          0,           9.81*20,  0, 0,  0),
}

HELP = (
    "\nCommands:\n"
    "  forward  / back      – thrust along X\n"
    "  left     / right     – thrust along Y\n"
    "  up       / down      – thrust along Z\n"
    "  yaw_left / yaw_right – rotate around Z\n"
    "  stop                 – zero all forces\n"
    "  quit / exit          – shut down\n"
)


def make_wrench(fx, fy, fz, tx, ty, tz):
    w = Wrench()
    w.force.x  = fx;  w.force.y  = fy;  w.force.z  = fz
    w.torque.x = tx;  w.torque.y = ty;  w.torque.z = tz
    return w


def main():
    rospy.init_node("force_controller", anonymous=True)
    pub  = rospy.Publisher(TOPIC, Wrench, queue_size=1)
    rate = rospy.Rate(PUBLISH_HZ)

    current_wrench = make_wrench(0, 0, 0, 0, 0, 0)

    rospy.loginfo("Force controller ready.  Publishing on %s", TOPIC)
    print(HELP)

    import threading

    def input_loop():
        nonlocal current_wrench
        while not rospy.is_shutdown():
            try:
                cmd = input("command> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                rospy.signal_shutdown("user quit")
                break

            if cmd in ("quit", "exit", "q"):
                rospy.signal_shutdown("user quit")
                break
            elif cmd in COMMANDS:
                current_wrench = make_wrench(*COMMANDS[cmd])
                rospy.loginfo("Applying: %s  %s", cmd, COMMANDS[cmd])
            elif cmd == "help":
                print(HELP)
            elif cmd == "":
                pass
            else:
                print(f"  Unknown command '{cmd}'.  Type 'help' for options.")

    t = threading.Thread(target=input_loop, daemon=True)
    t.start()

    while not rospy.is_shutdown():
        pub.publish(current_wrench)
        rate.sleep()


if __name__ == "__main__":
    main()