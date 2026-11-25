#!/usr/bin/env python3
"""Bridge between /cmd_vel and the VESC driver.

The pure pursuit controller publishes geometry_msgs/Twist on /cmd_vel:
  linear.x  = desired speed in m/s
  angular.z = steering angle in radians (positive = left, ROS convention)

The VESC driver (f1tenth/vesc) expects:
  /commands/motor/duty_cycle    std_msgs/Float64   motor duty cycle in [-1, 1]
  /commands/servo/position      std_msgs/Float64   servo position in [0, 1], 0.5 = straight

The speed-to-duty mapping below is a simple linear approximation. It is NOT
calibrated for any specific car: measure your car and tune the parameters.
A watchdog stops the motor when /cmd_vel stops arriving.
"""
import math

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


def clamp(value, low, high):
    return max(low, min(high, value))


def speed_to_duty(speed, speed_to_duty_gain, max_duty):
    """Linear speed (m/s) -> motor duty cycle, limited to +/- max_duty."""
    return clamp(speed * speed_to_duty_gain, -max_duty, max_duty)


def steering_to_servo(steering_rad, servo_center, max_steer_rad, inverted):
    """Steering angle (rad) -> servo position in [0, 1].

    +/- max_steer_rad maps to the two ends of the servo range around servo_center.
    """
    steering_rad = clamp(steering_rad, -max_steer_rad, max_steer_rad)
    offset = 0.5 * steering_rad / max_steer_rad
    if inverted:
        offset = -offset
    return clamp(servo_center + offset, 0.0, 1.0)


class CmdVelToVesc:
    def __init__(self):
        rospy.init_node('cmdvel_to_vesc')

        # --- Parameters (calibrate on your car) ---
        self.max_duty = rospy.get_param('~max_duty', 0.2)                 # hard limit on motor duty cycle
        self.speed_to_duty_gain = rospy.get_param('~speed_to_duty', 0.1)  # duty cycle per m/s (uncalibrated)
        self.servo_center = rospy.get_param('~servo_center', 0.5)         # servo value for driving straight
        self.max_steer_rad = rospy.get_param('~max_steer_rad', math.radians(30.0))
        self.steering_inverted = rospy.get_param('~steering_inverted', False)
        self.cmd_timeout = rospy.get_param('~cmd_timeout', 0.5)           # s, stop if no command for this long

        self.last_cmd_time = rospy.Time(0)

        self.motor_pub = rospy.Publisher('/commands/motor/duty_cycle', Float64, queue_size=1)
        self.servo_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)

        # Watchdog: runs at 10 Hz
        rospy.Timer(rospy.Duration(0.1), self.watchdog)
        rospy.on_shutdown(self.stop)

    def cmd_vel_callback(self, msg):
        self.last_cmd_time = rospy.Time.now()

        duty = speed_to_duty(msg.linear.x, self.speed_to_duty_gain, self.max_duty)
        servo = steering_to_servo(msg.angular.z, self.servo_center,
                                  self.max_steer_rad, self.steering_inverted)

        rospy.loginfo_throttle(1.0, "speed=%.2f m/s -> duty=%.3f, steer=%.2f rad -> servo=%.3f",
                               msg.linear.x, duty, msg.angular.z, servo)
        self.motor_pub.publish(Float64(data=duty))
        self.servo_pub.publish(Float64(data=servo))

    def watchdog(self, event):
        if (rospy.Time.now() - self.last_cmd_time).to_sec() > self.cmd_timeout:
            self.motor_pub.publish(Float64(data=0.0))

    def stop(self):
        self.motor_pub.publish(Float64(data=0.0))
        self.servo_pub.publish(Float64(data=self.servo_center))


if __name__ == '__main__':
    try:
        CmdVelToVesc()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
