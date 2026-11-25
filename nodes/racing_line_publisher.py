#!/usr/bin/env python3
"""Publish a racing line CSV (columns x, y and optionally v) as the global path.

Publishes nav_msgs/Path on /racing_line (latched). The speed v of every point is stored in
pose.position.z, which the pure pursuit controller uses as the target speed. A CSV with only
x and y (for example from waypoint_recorder.py) is also accepted: speeds are then 0, and the
controller falls back to its default speed.

The file is made by tools/plan_racing_line.py (from a map) or waypoint_recorder.py (driven by hand).
"""
import csv
import math
import os

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


def load_waypoints(path):
    """Read a CSV with columns x, y[, v]. Header and malformed rows are skipped."""
    points = []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            try:
                x, y = float(row[0]), float(row[1])
                v = float(row[2]) if len(row) > 2 else 0.0
            except (ValueError, IndexError):
                continue
            points.append((x, y, v))
    return points


def build_path(points, frame_id, stamp):
    path = Path()
    path.header.stamp = stamp
    path.header.frame_id = frame_id
    n = len(points)
    for i, (x, y, v) in enumerate(points):
        nx, ny, _ = points[(i + 1) % n]
        yaw = math.atan2(ny - y, nx - x)
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, v
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        path.poses.append(pose)
    return path


if __name__ == '__main__':
    rospy.init_node('racing_line_publisher')
    csv_path = os.path.expanduser(rospy.get_param('~file', '~/racing_line.csv'))
    frame_id = rospy.get_param('~map_frame', 'map')

    waypoints = load_waypoints(csv_path)
    if len(waypoints) < 2:
        rospy.logfatal("Need at least 2 waypoints in %s (found %d)", csv_path, len(waypoints))
        raise SystemExit(1)

    pub = rospy.Publisher('/racing_line', Path, queue_size=1, latch=True)
    path_msg = build_path(waypoints, frame_id, rospy.Time.now())
    pub.publish(path_msg)
    rospy.loginfo("Publishing %d waypoints from %s on /racing_line", len(waypoints), csv_path)

    # Re-publish periodically so nodes started later also receive the path
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        path_msg.header.stamp = rospy.Time.now()
        pub.publish(path_msg)
        rate.sleep()
