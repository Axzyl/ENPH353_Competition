#!/usr/bin/env python3
"""
extract_waypoints.py
--------------------
Extracts the positions of temporary waypoint objects placed in Gazebo
(named "beer", "beer_clone", "beer_clone_0", "beer_clone_1", ...) and
saves them in order to waypoints.txt.

Order:
  1. beer
  2. beer_clone
  3. beer_clone_0
  4. beer_clone_1
  ...

Usage:
    rosrun <your_package> extract_waypoints.py
"""

import rospy
from gazebo_msgs.srv import GetModelState


OUTPUT_FILE = "waypoints.txt"


def get_position(svc, model_name):
    """
    Returns (x, y) for the given Gazebo model name,
    or None if the model doesn't exist.
    """
    try:
        resp = svc(model_name, "world")
        if resp.success:
            return resp.pose.position.x, resp.pose.position.y
        return None
    except Exception:
        return None


def build_name_list():
    """
    Builds the ordered list of model names to query:
      beer, beer_clone, beer_clone_0, beer_clone_1, ...
    Keeps going until a name isn't found in Gazebo.
    """
    rospy.wait_for_service("/gazebo/get_model_state")
    svc = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    names = ["beer", "beer_clone"]
    index = 0
    while True:
        names.append(f"beer_clone_{index}")
        index += 1
        # Stop building the list once we have enough headroom;
        # we'll trim missing ones in the next step.
        if index > 100:
            break

    # Query each name and keep only the ones that exist, in order
    waypoints = []
    for name in names:
        pos = get_position(svc, name)
        if pos is not None:
            waypoints.append((name, pos[0], pos[1]))
            rospy.loginfo("  %-20s  x=%.4f  y=%.4f", name, pos[0], pos[1])
        else:
            # Once we hit a missing clone index, stop — they're sequential
            if name.startswith("beer_clone_"):
                rospy.loginfo("  %-20s  not found, stopping.", name)
                break

    return waypoints


def main():
    rospy.init_node("extract_waypoints", anonymous=True)
    rospy.loginfo("Querying Gazebo for waypoint objects...")

    waypoints = build_name_list()

    if not waypoints:
        rospy.logerr("No waypoint objects found. Is Gazebo running?")
        return

    rospy.loginfo("Found %d waypoints. Writing to %s...", len(waypoints), OUTPUT_FILE)

    with open(OUTPUT_FILE, "w") as f:
        f.write("# Waypoints extracted from Gazebo\n")
        f.write("# index, name, x, y\n")
        for i, (name, x, y) in enumerate(waypoints):
            f.write(f"{i}, {name}, {x:.6f}, {y:.6f}\n")

    rospy.loginfo("Done. %s written with %d waypoints.", OUTPUT_FILE, len(waypoints))

    # Print a summary table
    print("\n{:<6} {:<20} {:>10} {:>10}".format("Index", "Name", "X", "Y"))
    print("-" * 50)
    for i, (name, x, y) in enumerate(waypoints):
        print("{:<6} {:<20} {:>10.4f} {:>10.4f}".format(i, name, x, y))

    # Print as a 2D list of positions only
    positions = [[x, y] for _, x, y in waypoints]
    print("\nPositions as 2D list:")
    print(positions)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass