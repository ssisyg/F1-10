#!/usr/bin/env python3
"""Pure pursuit path follower.

Subscribes to a target path (nav_msgs/Path) and the car pose (geometry_msgs/PoseStamped)
and publishes geometry_msgs/Twist on /cmd_vel:
    linear.x  = speed in m/s
    angular.z = steering angle in radians (positive = left)

Speed: if the path points carry a speed in pose.position.z (see plan_racing_line.py),
the speed at the lookahead point is used; otherwise `default_speed`. The result is
limited to [min_speed, max_speed] and reduced in sharp turns.
The car is stopped when the pose or the path is missing or too old.
"""
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path


def is_closed(points, tol):
    """A path is a closed loop when its last point is close to its first one."""
    return len(points) > 10 and np.hypot(*(points[0, :2] - points[-1, :2])) < tol


def find_lookahead_index(points, position, lookahead, closed):
    """Index of the first path point at least `lookahead` meters from `position`.

    The search starts at the point closest to the car and moves forward, wrapping
    around when the path is a closed loop. Falls back to the last point.
    """
    dists = np.hypot(points[:, 0] - position[0], points[:, 1] - position[1])
    closest = int(np.argmin(dists))
    n = len(points)
    for k in range(n):
        i = closest + k
        if i >= n:
            if not closed:
                break
            i -= n
        if dists[i] >= lookahead:
            return i
    return n - 1


def pure_pursuit_steering(yaw, position, target, wheelbase, lookahead_distance):
    alpha = np.arctan2(target[1] - position[1], target[0] - position[0]) - yaw
    # wrap to [-pi, pi]
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
    return float(np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead_distance))


class PurePursuitController:
    def __init__(self):
        rospy.init_node('pure_pursuit_controller')

        # --- Parameters (CRITICAL TO TUNE!) ---
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 1.5)  # m
        self.max_speed = rospy.get_param('~max_speed', 3.0)                    # m/s
        self.min_speed = rospy.get_param('~min_speed', 1.0)                    # m/s
        self.default_speed = rospy.get_param('~default_speed', 1.0)            # m/s, used when the path has no speed
        self.wheelbase = rospy.get_param('~wheelbase', 0.325)                  # m
        self.k_p_speed = rospy.get_param('~k_p_speed', 0.5)                    # speed reduction per rad of steering
        self.pose_timeout = rospy.get_param('~pose_timeout', 0.5)              # s, stop if the pose is older than this
        self.closed_tol = rospy.get_param('~closed_path_tolerance', 0.5)       # m, first/last point distance of a loop

        # --- Internal Variables ---
        self.target_path = None
        self.path_points = None
        self.path_closed = False
        self.current_pose = None
        self.pose_time = None

        # --- ROS Subscribers/Publishers ---
        rospy.Subscriber('/target_path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseStamped, self.pose_callback, queue_size=1)  # remapped in the launch files
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Main control loop runs at 50 Hz
        rospy.Timer(rospy.Duration(0.02), self.control_loop)

    def path_callback(self, msg):
        if not msg.poses:
            return
        self.target_path = msg
        self.path_points = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses])
        self.path_closed = is_closed(self.path_points, self.closed_tol)

    def pose_callback(self, msg):
        self.current_pose = msg.pose
        self.pose_time = rospy.Time.now()

    def control_loop(self, event):
        if self.path_points is None or self.current_pose is None:
            self.publish_stop_command()
            return
        if (rospy.Time.now() - self.pose_time).to_sec() > self.pose_timeout:
            rospy.logwarn_throttle(2.0, "Pose is older than %.1f s: stopping" % self.pose_timeout)
            self.publish_stop_command()
            return

        pos = np.array([self.current_pose.position.x, self.current_pose.position.y])
        q = self.current_pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        idx = find_lookahead_index(self.path_points, pos, self.lookahead_distance, self.path_closed)
        target = self.path_points[idx]

        steering = pure_pursuit_steering(yaw, pos, target, self.wheelbase, self.lookahead_distance)

        # Speed from the path if it has one, then reduce it in sharp turns
        wanted = target[2] if target[2] > 0.05 else self.default_speed
        speed = min(wanted, self.max_speed - abs(steering) * self.k_p_speed)
        speed = float(np.clip(speed, self.min_speed, self.max_speed))

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = steering  # radians; cmdvel_to_vesc.py maps it to the servo
        self.cmd_vel_pub.publish(cmd)

    def publish_stop_command(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


if __name__ == '__main__':
    try:
        PurePursuitController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
