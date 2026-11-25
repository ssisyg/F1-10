#!/usr/bin/env python3
"""TEST ONLY: a fake car for dry runs without the real car.

It replaces the Hokuyo LiDAR, the VESC and the car itself:
  * moves a kinematic bicycle model with the commands on /cmd_vel
  * ray casts the occupancy map (plus optional obstacles) and publishes /scan
  * publishes /odom and the transform odom -> base_link, with a speed scale error and a
    yaw bias so the odometry drifts like the real one (AMCL has to correct it)
  * publishes the true pose on /ground_truth_pose, and /fake_car/clearance (distance in
    meters between the car body and the nearest wall or obstacle; negative = collision)

It is a simple model, not a physics simulation: use it to check that the software chain
works, not to tune speeds or steering for the real car.
"""
import math
import os

import numpy as np
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry
from PIL import Image
from scipy import ndimage
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class OccupancyMap:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img = meta['image']
        if not os.path.isabs(img):
            img = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img)
        arr = np.flipud(np.array(Image.open(img).convert('L'), dtype=float))
        p = arr / 255.0 if meta.get('negate', 0) else (255.0 - arr) / 255.0
        self.res = float(meta['resolution'])
        self.origin = np.array(meta['origin'][:2], dtype=float)
        self.occupied = p > float(meta.get('occupied_thresh', 0.65))
        self.dist = ndimage.distance_transform_edt(~self.occupied) * self.res  # meters to the nearest wall

    def cell(self, pts):
        ij = np.floor((pts - self.origin) / self.res).astype(int)
        h, w = self.occupied.shape
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < w) & (ij[..., 1] >= 0) & (ij[..., 1] < h)
        return ij, inside

    def is_occupied(self, pts):
        ij, inside = self.cell(pts)
        out = np.ones(pts.shape[:-1], dtype=bool)  # outside the map counts as a wall
        out[inside] = self.occupied[ij[inside, 1], ij[inside, 0]]
        return out

    def distance(self, pts):
        ij, inside = self.cell(pts)
        out = np.zeros(pts.shape[:-1])
        out[inside] = self.dist[ij[inside, 1], ij[inside, 0]]
        return out


def cast_rays(grid, origin, yaw, angles, circles, range_min, range_max, step):
    """Range of every beam: first wall hit (marching) or circle intersection (analytic)."""
    th = yaw + angles
    d = np.stack([np.cos(th), np.sin(th)], axis=1)                       # (nb, 2)
    t = np.arange(range_min, range_max, step)                            # (nt,)
    pts = origin[None, None, :] + d[:, None, :] * t[None, :, None]       # (nb, nt, 2)
    hit = grid.is_occupied(pts)
    first = np.argmax(hit, axis=1)
    ranges = np.where(hit.any(axis=1), t[first], np.inf)
    for cx, cy, r in circles:
        oc = np.array([cx, cy]) - origin
        proj = d @ oc
        perp2 = oc @ oc - proj ** 2
        ok = (proj > 0) & (perp2 <= r * r)
        tc = np.where(ok, proj - np.sqrt(np.maximum(r * r - perp2, 0.0)), np.inf)
        ranges = np.minimum(ranges, tc)
    ranges[(ranges < range_min) | (ranges > range_max)] = np.inf
    return ranges


class MovingObstacle:
    """A disc driving along a closed path (CSV with columns x, y[, v])."""

    def __init__(self, csv_path, speed, radius, start_s):
        pts = np.loadtxt(os.path.expanduser(csv_path), delimiter=',', skiprows=1)[:, :2]
        pts = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.pts, self.s = pts, np.concatenate([[0], np.cumsum(seg)])
        self.length = self.s[-1]
        self.speed, self.radius, self.pos_s = speed, radius, start_s % self.length

    def step(self, dt):
        self.pos_s = (self.pos_s + self.speed * dt) % self.length

    def position(self):
        return np.array([np.interp(self.pos_s, self.s, self.pts[:, 0]), np.interp(self.pos_s, self.s, self.pts[:, 1])])


class FakeCar:
    def __init__(self):
        rospy.init_node('fake_car')
        self.grid = OccupancyMap(os.path.expanduser(rospy.get_param('~map_yaml')))

        self.x = rospy.get_param('~init_x', 0.0)
        self.y = rospy.get_param('~init_y', 0.0)
        self.yaw = rospy.get_param('~init_yaw', 0.0)
        self.wheelbase = rospy.get_param('~wheelbase', 0.325)
        self.max_steer = rospy.get_param('~max_steer', math.radians(30.0))
        self.max_accel = rospy.get_param('~max_accel', 3.0)
        self.steer_tau = rospy.get_param('~steer_tau', 0.08)        # s, servo lag
        self.cmd_timeout = rospy.get_param('~cmd_timeout', 0.5)

        self.v = 0.0
        self.steer = 0.0
        self.cmd = (0.0, 0.0)
        self.cmd_time = rospy.Time(0)

        # Odometry errors, so AMCL has something to correct
        self.speed_scale = 1.0 + rospy.get_param('~odom_speed_scale_error', 0.04)
        self.yaw_bias = rospy.get_param('~odom_yaw_bias', 0.03)    # rad/s
        self.ox, self.oy, self.oyaw = self.x, self.y, self.yaw

        # LiDAR
        self.num_beams = rospy.get_param('~num_beams', 541)
        self.fov = math.radians(rospy.get_param('~fov_deg', 270.0))
        self.range_min = rospy.get_param('~range_min', 0.05)
        self.range_max = rospy.get_param('~range_max', 10.0)
        self.scan_noise = rospy.get_param('~scan_noise', 0.01)
        self.angles = np.linspace(-self.fov / 2, self.fov / 2, self.num_beams)
        self.scan_every = max(1, int(round(rospy.get_param('~sim_rate', 50.0) / rospy.get_param('~scan_rate', 20.0))))
        self.tick = 0

        self.static_obstacles = [tuple(o) for o in rospy.get_param('~obstacles', [])]  # [[x, y, r], ...]
        self.moving = None
        if rospy.has_param('~moving_obstacle_csv'):
            self.moving = MovingObstacle(rospy.get_param('~moving_obstacle_csv'),
                                         rospy.get_param('~moving_obstacle_speed', 0.5),
                                         rospy.get_param('~moving_obstacle_radius', 0.15),
                                         rospy.get_param('~moving_obstacle_start_s', 5.0))
        self.publish_gt_tf = rospy.get_param('~publish_ground_truth_tf', False)
        self.noise_rng = np.random.default_rng(1)

        self.scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=1)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=5)
        self.gt_pub = rospy.Publisher('/ground_truth_pose', PoseStamped, queue_size=5)
        self.clear_pub = rospy.Publisher('/fake_car/clearance', Float32, queue_size=5)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_callback, queue_size=1)

        self.dt = 1.0 / rospy.get_param('~sim_rate', 50.0)
        rospy.Timer(rospy.Duration(self.dt), self.step)

    def cmd_callback(self, msg):
        self.cmd = (msg.linear.x, float(np.clip(msg.angular.z, -self.max_steer, self.max_steer)))
        self.cmd_time = rospy.Time.now()

    def obstacles(self):
        circles = list(self.static_obstacles)
        if self.moving is not None:
            px, py = self.moving.position()
            circles.append((px, py, self.moving.radius))
        return circles

    def step(self, event):
        dt = self.dt
        v_cmd, steer_cmd = self.cmd
        if (rospy.Time.now() - self.cmd_time).to_sec() > self.cmd_timeout:
            v_cmd = 0.0
        dv = float(np.clip(v_cmd - self.v, -self.max_accel * dt, self.max_accel * dt))
        self.v += dv
        self.steer += (steer_cmd - self.steer) * min(1.0, dt / self.steer_tau)

        yaw_rate = self.v * math.tan(self.steer) / self.wheelbase
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += yaw_rate * dt
        if self.moving is not None:
            self.moving.step(dt)

        # Drifting odometry
        ov, oyr = self.v * self.speed_scale, yaw_rate + self.yaw_bias
        self.ox += ov * math.cos(self.oyaw) * dt
        self.oy += ov * math.sin(self.oyaw) * dt
        self.oyaw += oyr * dt

        now = rospy.Time.now()
        self.publish_odom(now, ov, oyr)
        self.publish_truth(now)
        self.publish_clearance()
        self.tick += 1
        if self.tick % self.scan_every == 0:
            self.publish_scan(now)

    def publish_odom(self, stamp, v, yaw_rate):
        qz, qw = math.sin(self.oyaw / 2), math.cos(self.oyaw / 2)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, 'odom', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.ox, self.oy
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = v, yaw_rate
        self.odom_pub.publish(odom)
        if not self.publish_gt_tf:
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, 'odom', 'base_link'
            t.transform.translation.x, t.transform.translation.y = self.ox, self.oy
            t.transform.rotation.z, t.transform.rotation.w = qz, qw
            self.tf_pub.sendTransform(t)

    def publish_truth(self, stamp):
        qz, qw = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = stamp, 'map'
        msg.pose.position.x, msg.pose.position.y = self.x, self.y
        msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
        self.gt_pub.publish(msg)
        if self.publish_gt_tf:  # map -> base_link straight from the truth (no localization)
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, 'map', 'base_link'
            t.transform.translation.x, t.transform.translation.y = self.x, self.y
            t.transform.rotation.z, t.transform.rotation.w = qz, qw
            self.tf_pub.sendTransform(t)

    def publish_clearance(self):
        # Car body approximated by three discs along its length
        offsets = np.array([-0.15, 0.0, 0.15])
        centers = np.stack([self.x + offsets * math.cos(self.yaw), self.y + offsets * math.sin(self.yaw)], axis=1)
        car_r = 0.18
        clearance = float(np.min(self.grid.distance(centers)) - car_r)
        for cx, cy, r in self.obstacles():
            d = np.min(np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)) - r - car_r
            clearance = min(clearance, float(d))
        self.clear_pub.publish(Float32(data=clearance))

    def publish_scan(self, stamp):
        ranges = cast_rays(self.grid, np.array([self.x, self.y]), self.yaw, self.angles, self.obstacles(),
                           self.range_min, self.range_max, 0.03)
        noisy = ranges + self.noise_rng.normal(0.0, self.scan_noise, ranges.shape)
        msg = LaserScan()
        msg.header.stamp, msg.header.frame_id = stamp, 'laser'
        msg.angle_min, msg.angle_max = float(self.angles[0]), float(self.angles[-1])
        msg.angle_increment = float(self.angles[1] - self.angles[0])
        msg.scan_time = 1.0 / 20.0
        msg.range_min, msg.range_max = self.range_min, self.range_max
        msg.ranges = [float(r) for r in noisy]
        self.scan_pub.publish(msg)


if __name__ == '__main__':
    try:
        FakeCar()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
