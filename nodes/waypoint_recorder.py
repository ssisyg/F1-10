#!/usr/bin/env python3
"""Record the car's pose to a CSV file while it is driven around the track.

Drive the car manually (or with any other controller) while this node runs.
The resulting CSV (columns: x, y) is replayed by racing_line_publisher.py as
the global racing line.

Note: The map built by SLAM has its origin at the car start pose, so the map
frame origin is the car's starting pose. Record and replay with the car
starting from the same place and heading.
"""
import csv
import math
import os

import rospy
from geometry_msgs.msg import PoseStamped


class WaypointRecorder:
    def __init__(self):
        rospy.init_node('waypoint_recorder')

        self.min_spacing = rospy.get_param('~min_spacing', 0.1)  # m between recorded points
        output = rospy.get_param('~output', '~/racing_line.csv')
        self.output = os.path.expanduser(output)

        self.last_xy = None
        self.count = 0

        self.file = open(self.output, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['x', 'y'])

        pose_topic = rospy.get_param('~pose_topic', '/pose')  # must be geometry_msgs/PoseStamped
        rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback, queue_size=10)
        rospy.on_shutdown(self.close)
        rospy.loginfo("Recording waypoints to %s", self.output)

    def pose_callback(self, msg):
        x, y = msg.pose.position.x, msg.pose.position.y
        if self.last_xy is not None and math.hypot(x - self.last_xy[0], y - self.last_xy[1]) < self.min_spacing:
            return
        self.writer.writerow([x, y])
        self.file.flush()
        self.last_xy = (x, y)
        self.count += 1

    def close(self):
        rospy.loginfo("Recorded %d waypoints to %s", self.count, self.output)
        self.file.close()


if __name__ == '__main__':
    try:
        WaypointRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
