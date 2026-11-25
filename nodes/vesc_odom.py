#!/usr/bin/env python3
"""Wheel odometry for AMCL and SLAM, from the VESC speed and the commanded steering.

The car has no wheel encoders, so the motor speed (ERPM reported by the VESC on
/sensors/core) and the commanded servo position are integrated with a kinematic
bicycle model. It drifts, which is fine: AMCL and SLAM correct it with the LiDAR.

Publishes nav_msgs/Odometry on /odom and the transform odom -> base_link.
The steering parameters must match the ones of cmdvel_to_vesc.py.
"""
import math

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped


def clamp(value, low, high):
    return max(low, min(high, value))


def servo_to_steering(servo, servo_center, max_steer_rad, inverted):
    """Inverse of steering_to_servo() in cmdvel_to_vesc.py."""
    offset = (servo - servo_center) * 2.0 * max_steer_rad
    return -offset if inverted else offset


def bicycle_step(x, y, yaw, speed, steering, wheelbase, dt):
    """One integration step of the kinematic bicycle model."""
    x += speed * math.cos(yaw) * dt
    y += speed * math.sin(yaw) * dt
    yaw += speed * math.tan(steering) / wheelbase * dt
    return x, y, yaw


class VescOdom:
    def __init__(self):
        rospy.init_node('vesc_odom')

        self.gain = rospy.get_param('~speed_to_erpm_gain', 4614.0)    # ERPM per m/s (F1TENTH default)
        self.offset = rospy.get_param('~speed_to_erpm_offset', 0.0)
        self.wheelbase = rospy.get_param('~wheelbase', 0.325)
        self.servo_center = rospy.get_param('~servo_center', 0.5)
        self.max_steer_rad = rospy.get_param('~max_steer_rad', math.radians(30.0))
        self.steering_inverted = rospy.get_param('~steering_inverted', False)
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.publish_tf = rospy.get_param('~publish_tf', True)

        self.x = self.y = self.yaw = 0.0
        self.servo = self.servo_center
        self.last_stamp = None

        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        rospy.Subscriber('/commands/servo/position', Float64, self.servo_callback, queue_size=1)
        rospy.Subscriber('/sensors/core', VescStateStamped, self.core_callback, queue_size=10)

    def servo_callback(self, msg):
        self.servo = msg.data

    def core_callback(self, msg):
        stamp = msg.header.stamp
        if self.last_stamp is None:
            self.last_stamp = stamp
            return
        dt = (stamp - self.last_stamp).to_sec()
        self.last_stamp = stamp
        if dt <= 0.0 or dt > 0.5:
            return

        speed = (msg.state.speed - self.offset) / self.gain
        steering = servo_to_steering(self.servo, self.servo_center, self.max_steer_rad, self.steering_inverted)
        self.x, self.y, self.yaw = bicycle_step(self.x, self.y, self.yaw, speed, steering, self.wheelbase, dt)
        self.publish(stamp, speed, speed * math.tan(steering) / self.wheelbase)

    def publish(self, stamp, speed, yaw_rate):
        qz, qw = math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.twist.twist.linear.x = speed
        odom.twist.twist.angular.z = yaw_rate
        self.odom_pub.publish(odom)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x, t.transform.translation.y = self.x, self.y
            t.transform.rotation.z, t.transform.rotation.w = qz, qw
            self.tf_pub.sendTransform(t)


if __name__ == '__main__':
    try:
        VescOdom()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
