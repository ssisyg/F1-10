#!/usr/bin/env python3
"""Follow the Gap: a reactive driver that only needs the LiDAR.

It is used to drive the mapping lap (no map needed) and works as a simple baseline
for driving around obstacles.

Algorithm:
  1. Keep the beams in the forward field of view and clip their range.
  2. Find the closest point and clear a "bubble" around it, so the car does not clip it.
  3. Find the largest gap of free beams and aim at its middle, biased to the farthest points.
  4. Steer towards that point; slow down in turns and when the way ahead is short.

Publishes geometry_msgs/Twist on /cmd_vel (linear.x = m/s, angular.z = steering angle in radians).
"""
import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


def largest_gap(free):
    """Start and end index (inclusive) of the longest run of True values in a boolean array."""
    best = (0, -1)
    start = None
    for i, f in enumerate(free):
        if f and start is None:
            start = i
        if (not f or i == len(free) - 1) and start is not None:
            end = i if f else i - 1
            if end - start > best[1] - best[0]:
                best = (start, end)
            start = None
    return best


def compute_command(ranges, angle_min, angle_increment, p):
    """Return (speed, steering_rad) for one scan. `p` is a dict of parameters."""
    ranges = np.nan_to_num(np.asarray(ranges, dtype=float), nan=0.0, posinf=p['max_range'], neginf=0.0)
    angles = angle_min + np.arange(len(ranges)) * angle_increment

    fov = (angles >= -p['fov']) & (angles <= p['fov'])
    r = np.clip(ranges[fov], 0.0, p['max_range'])
    a = angles[fov]
    if len(r) == 0:
        return 0.0, 0.0
    k = max(1, int(p['smooth_window']))
    r = np.convolve(r, np.ones(k) / k, mode='same')

    # Stop if something is right in front of the car
    front = np.abs(a) < math.radians(15)
    if np.min(r[front]) < p['stop_distance']:
        return 0.0, 0.0

    # Bubble around the closest point
    closest = int(np.argmin(r))
    half = math.atan2(p['bubble_radius'], max(r[closest], 1e-3)) / angle_increment
    lo, hi = max(0, int(closest - half)), min(len(r), int(closest + half) + 1)
    r = r.copy()
    r[lo:hi] = 0.0

    # Largest gap of free beams and the best point inside it
    free = r > p['gap_threshold']
    start, end = largest_gap(free)
    if end < start:
        return 0.0, 0.0
    gap = slice(start, end + 1)
    # aim between the middle of the gap and its farthest point
    mid = (start + end) // 2
    far = start + int(np.argmax(r[gap]))
    best = int(round(p['far_bias'] * far + (1.0 - p['far_bias']) * mid))
    steering = float(np.clip(a[best], -p['max_steer'], p['max_steer']))

    # Speed: slower in turns and when the way ahead is short
    turn = abs(steering) / p['max_steer']
    ahead = min(1.0, r[best] / p['max_range'])
    speed = p['max_speed'] * (1.0 - 0.6 * turn) * (0.4 + 0.6 * ahead)
    return float(np.clip(speed, p['min_speed'], p['max_speed'])), steering


class FollowTheGap:
    def __init__(self):
        rospy.init_node('follow_the_gap')
        self.p = {
            'fov': math.radians(rospy.get_param('~fov_deg', 90.0)),
            'max_range': rospy.get_param('~max_range', 3.0),
            'smooth_window': rospy.get_param('~smooth_window', 5),
            'bubble_radius': rospy.get_param('~bubble_radius', 0.4),
            'gap_threshold': rospy.get_param('~gap_threshold', 0.8),
            'far_bias': rospy.get_param('~far_bias', 0.5),
            'stop_distance': rospy.get_param('~stop_distance', 0.3),
            'max_steer': math.radians(rospy.get_param('~max_steer_deg', 30.0)),
            'max_speed': rospy.get_param('~max_speed', 0.8),
            'min_speed': rospy.get_param('~min_speed', 0.3),
        }
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

    def scan_callback(self, msg):
        speed, steering = compute_command(msg.ranges, msg.angle_min, msg.angle_increment, self.p)
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = steering
        self.pub.publish(cmd)


if __name__ == '__main__':
    try:
        FollowTheGap()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
