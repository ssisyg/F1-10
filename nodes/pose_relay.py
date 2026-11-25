#!/usr/bin/env python3
"""Publish the car pose in the map frame as geometry_msgs/PoseStamped on /pose.

The pose is read from tf (map -> base_link), so it works with any localization that
provides the map frame (AMCL, slam_toolbox, Hector SLAM). Between two AMCL updates tf
is interpolated with the odometry, so the pose is smooth at the rate set by ~rate.
If the transform is not available the node publishes nothing, and the controller
stops the car when the pose gets too old.
"""
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped


if __name__ == '__main__':
    rospy.init_node('pose_relay')
    map_frame = rospy.get_param('~map_frame', 'map')
    base_frame = rospy.get_param('~base_frame', 'base_link')
    rate = rospy.Rate(rospy.get_param('~rate', 30.0))

    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)
    pub = rospy.Publisher('/pose', PoseStamped, queue_size=1)

    while not rospy.is_shutdown():
        try:
            t = buf.lookup_transform(map_frame, base_frame, rospy.Time(0))
            msg = PoseStamped()
            msg.header.stamp = t.header.stamp
            msg.header.frame_id = map_frame
            msg.pose.position.x = t.transform.translation.x
            msg.pose.position.y = t.transform.translation.y
            msg.pose.position.z = t.transform.translation.z
            msg.pose.orientation = t.transform.rotation
            pub.publish(msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn_throttle(5.0, "No transform %s -> %s yet", map_frame, base_frame)
        rate.sleep()
