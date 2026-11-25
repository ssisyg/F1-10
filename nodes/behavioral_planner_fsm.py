#!/usr/bin/env python3
"""Behavioral planner: a finite state machine that decides how to drive around obstacles.

States:
  LANE_FOLLOWING    follow the global racing line (/racing_line)
  AVOID_STATIC      pass a static (or not yet classified) obstacle that blocks the line
  OVERTAKE_DYNAMIC  pass a moving obstacle, using its predicted position

Local paths are built by shifting the racing line sideways: the lateral offset ramps
smoothly from the car's current offset to the offset needed to clear the obstacle,
holds while passing it, and ramps back to the racing line afterwards. Because the
local path is derived from the racing line it follows the shape of the track.
Speeds stored in the path (pose.position.z) are kept, capped at `avoid_speed`.
"""
from enum import Enum

import numpy as np
import rospy
from f1tenth_obstacle_avoidance.msg import TrackedObjectArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from scipy import ndimage
from std_msgs.msg import String


class FSMState(Enum):
    LANE_FOLLOWING = 1
    AVOID_STATIC = 2
    OVERTAKE_DYNAMIC = 3


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class PathGeometry:
    """Arc-length helpers for a path that may be an open path or a closed loop."""

    def __init__(self, points, closed_tol=0.5):
        pts = np.asarray(points, dtype=float)
        self.xy = pts[:, :2]
        self.z = pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts))
        n = len(self.xy)
        self.n = n
        self.closed = n > 10 and np.hypot(*(self.xy[0] - self.xy[-1])) < closed_tol

        nxt = np.roll(self.xy, -1, axis=0)
        seg = np.linalg.norm(nxt - self.xy, axis=1)
        if not self.closed:
            seg[-1] = 0.0
        self.s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        self.length = float(np.sum(seg))

        prev = np.roll(self.xy, 1, axis=0)
        d = nxt - prev
        if not self.closed:
            d[0] = self.xy[1] - self.xy[0]
            d[-1] = self.xy[-1] - self.xy[-2]
        norm = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        self.tangent = d / norm
        self.normal = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)  # left normal

    def closest(self, p):
        return int(np.argmin(np.linalg.norm(self.xy - np.asarray(p), axis=1)))

    def forward_distance(self, i_from, i_to):
        """Signed distance along the path from index i_from to i_to (negative = behind)."""
        d = self.s[i_to] - self.s[i_from]
        if self.closed and self.length > 0:
            d = (d + self.length / 2.0) % self.length - self.length / 2.0
        return float(d)

    def indices_ahead(self, i_from, distance):
        """Indices from i_from forward until `distance` meters of path have been covered."""
        out = []
        for k in range(self.n):
            i = i_from + k
            if i >= self.n:
                if not self.closed:
                    break
                i -= self.n
            if self.forward_distance(i_from, i) > distance and k > 0:
                break
            out.append(i)
        return out


def find_blocking_obstacle(geom, car_xy, obstacles, lookahead, behind, clearance):
    """Return (obstacle, distance_ahead, lateral_offset) of the closest obstacle that blocks the line.

    `obstacles` is a list of dicts with keys x, y, r. An obstacle blocks the line when it is within
    `lookahead` meters ahead (or less than `behind` meters behind the car, so it is not forgotten
    while the car is still next to it) and its edge is closer than `clearance` to the line.
    """
    i_car = geom.closest(car_xy)
    best = None
    for obs in obstacles:
        o = np.array([obs['x'], obs['y']])
        j = geom.closest(o)
        ahead = geom.forward_distance(i_car, j)
        if ahead > lookahead or ahead < -behind:
            continue
        offset = float(np.dot(o - geom.xy[j], geom.normal[j]))  # + = left of the line
        if abs(offset) - obs['r'] > clearance:
            continue  # beside the line, not blocking
        if best is None or ahead < best[1]:
            best = (obs, ahead, offset)
    return best


class WallDistance:
    """Distance (m) to the nearest wall from the static map; unknown cells count as walls."""

    def __init__(self, grid_msg):
        info = grid_msg.info
        data = np.array(grid_msg.data, dtype=np.int16).reshape(info.height, info.width)
        blocked = (data != 0)  # occupied (100) and unknown (-1) both block
        self.dist = ndimage.distance_transform_edt(~blocked) * info.resolution
        self.res = info.resolution
        self.ox, self.oy = info.origin.position.x, info.origin.position.y

    def at(self, xy):
        xy = np.atleast_2d(xy)
        ij = np.floor((xy - [self.ox, self.oy]) / self.res).astype(int)
        h, w = self.dist.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        out = np.zeros(len(xy))
        out[inside] = self.dist[ij[inside, 1], ij[inside, 0]]
        return out


def lateral_room(geom, walls, i_car, s_end, side, wall_clearance, max_offset, step=0.05):
    """Largest sideways shift (m, >= 0) of the racing line towards `side` that keeps
    `wall_clearance` to the walls over the next `s_end` meters. Without a map: max_offset."""
    if walls is None:
        return max_offset
    idx = geom.indices_ahead(i_car, s_end)
    base = geom.xy[idx]
    normals = geom.normal[idx]
    room = 0.0
    while room + step <= max_offset + 1e-9:
        cand = base + side * (room + step) * normals
        if np.min(walls.at(cand)) < wall_clearance:
            break
        room += step
    return room


def choose_side(obstacle_offset, clear, room, default_side):
    """+1 (left) or -1 (right). Prefer the side the obstacle is NOT on; if the map says that side
    has too little room but the other one has more, use the other one."""
    if obstacle_offset > 0.05:
        pref = -1.0
    elif obstacle_offset < -0.05:
        pref = 1.0
    else:
        pref = default_side
    if room is None:
        return pref
    # spare room on each side after clearing the obstacle (negative = does not fit)
    spare = {sd: room[sd] - sd * (obstacle_offset + sd * clear) for sd in (1.0, -1.0)}
    if spare[pref] >= 0.0 or spare[-pref] <= spare[pref]:
        return pref
    return -pref


def build_local_path(geom, car_xy, obstacle_ahead, obstacle_offset, obstacle_radius, params, side=None,
                     room=None):
    """Racing line shifted sideways around an obstacle. Returns (points Nx3, lateral_target, side)."""
    i_car = geom.closest(car_xy)
    clear = obstacle_radius + params['lateral_safety_margin'] + params['car_half_width']
    if side is None:
        side = choose_side(obstacle_offset, clear, room, params['default_side'])
    target = obstacle_offset + side * clear
    limit = params['max_lateral_offset'] if room is None else min(params['max_lateral_offset'], room[side])
    limited = side * limit if side * target > limit else target
    limited = float(np.clip(limited, -params['max_lateral_offset'], params['max_lateral_offset']))

    lam0 = float(np.dot(np.asarray(car_xy) - geom.xy[i_car], geom.normal[i_car]))
    ramp, hold, ramp_out = params['avoid_ramp'], params['avoid_hold'], params['avoid_ramp_out']
    s_in = max(0.5, obstacle_ahead - ramp)
    s_hold_end = max(obstacle_ahead, 0.0) + hold
    s_end = s_hold_end + ramp_out

    pts = []
    for i in geom.indices_ahead(i_car, s_end):
        s = geom.forward_distance(i_car, i)
        if s < s_in:
            lam = lam0 + (limited - lam0) * smoothstep(s / s_in)
        elif s <= s_hold_end:
            lam = limited
        else:
            lam = limited * (1.0 - smoothstep((s - s_hold_end) / ramp_out))
        xy = geom.xy[i] + lam * geom.normal[i]
        z = geom.z[i]
        z = params['avoid_speed'] if z <= 0.05 else min(z, params['avoid_speed'])
        pts.append((xy[0], xy[1], z))
    return np.array(pts), limited, side, target


class BehavioralPlannerFSM:
    def __init__(self):
        rospy.init_node('behavioral_planner_fsm_node')

        # --- Parameters (CRITICAL TO TUNE!) ---
        self.decision_lookahead_dist = rospy.get_param('~decision_lookahead_dist', 4.0)  # m ahead to look for obstacles
        self.overtake_prediction_time = rospy.get_param('~overtake_prediction_time', 1.5)  # s
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.params = {
            'lateral_safety_margin': rospy.get_param('~lateral_safety_margin', 0.3),  # m of free space beside the car
            'car_half_width': rospy.get_param('~car_half_width', 0.15),               # m
            'max_lateral_offset': rospy.get_param('~max_lateral_offset', 0.8),        # m from the racing line
            'default_side': 1.0 if rospy.get_param('~default_side', 'left') == 'left' else -1.0,
            'avoid_ramp': rospy.get_param('~avoid_ramp', 2.0),                        # m to move over before the obstacle
            'avoid_hold': rospy.get_param('~avoid_hold', 0.8),                        # m to stay offset after the obstacle
            'avoid_ramp_out': rospy.get_param('~avoid_ramp_out', 1.5),                # m to return to the line
            'avoid_speed': rospy.get_param('~avoid_speed', 0.8),                      # m/s speed cap while passing
        }
        self.behind_margin = rospy.get_param('~behind_margin', 0.6)                   # m, keep avoiding while next to it
        self.hold_time = rospy.get_param('~avoid_hold_time', 0.4)                     # s, keep last path if the obstacle flickers

        # --- Internal Variables ---
        self.current_state = FSMState.LANE_FOLLOWING
        self.geom = None
        self.global_path = None
        self.current_pose = None
        self.obstacles = []
        self.last_local = None
        self.last_local_time = rospy.Time(0)
        self.side = None
        self.side_obstacle_id = None
        self.walls = None
        self.wall_clearance = rospy.get_param('~wall_clearance', 0.2)  # m between the car's center line and a wall

        # --- ROS Subscribers/Publishers ---
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber('/racing_line', Path, self.global_path_callback, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseStamped, self.pose_callback, queue_size=1)  # remapped in the launch files
        rospy.Subscriber('/lidar_obstacles', TrackedObjectArray, self.obstacles_callback, queue_size=1)
        self.target_path_pub = rospy.Publisher('/target_path', Path, queue_size=1)
        self.state_pub = rospy.Publisher('/fsm_state', String, queue_size=1)

        # Main FSM loop runs at 20 Hz
        rospy.Timer(rospy.Duration(0.05), self.run_fsm)

    def map_callback(self, msg):
        self.walls = WallDistance(msg)
        rospy.loginfo("Static map received: passing sides are checked against the walls")

    def global_path_callback(self, msg):
        if len(msg.poses) < 3:
            return
        self.global_path = msg
        pts = [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses]
        self.geom = PathGeometry(pts)
        rospy.loginfo("Racing line received: %d points, %.1f m, %s", self.geom.n, self.geom.length,
                      'closed loop' if self.geom.closed else 'open path')

    def pose_callback(self, msg):
        self.current_pose = msg.pose

    def obstacles_callback(self, msg):
        self.obstacles = [{
            'id': o.id, 'x': o.pose.position.x, 'y': o.pose.position.y,
            'vx': o.velocity.linear.x, 'vy': o.velocity.linear.y,
            'r': max(o.size.x, o.size.y) / 2.0, 'class': o.class_name,
        } for o in msg.objects]

    def run_fsm(self, event):
        if self.geom is None or self.current_pose is None:
            rospy.loginfo_throttle(5, "Waiting for the racing line and the pose...")
            return

        car = np.array([self.current_pose.position.x, self.current_pose.position.y])
        clearance = self.params['car_half_width'] + self.params['lateral_safety_margin']

        # Dynamic obstacles are judged at their predicted position
        candidates = []
        for o in self.obstacles:
            o = dict(o)
            if o['class'] == 'DYNAMIC':
                o['x'] += o['vx'] * self.overtake_prediction_time
                o['y'] += o['vy'] * self.overtake_prediction_time
            candidates.append(o)

        blocking = find_blocking_obstacle(self.geom, car, candidates, self.decision_lookahead_dist,
                                          self.behind_margin, clearance)

        if blocking is None:
            new_state = FSMState.LANE_FOLLOWING
        elif blocking[0]['class'] == 'DYNAMIC':
            new_state = FSMState.OVERTAKE_DYNAMIC
        else:  # STATIC and UNKNOWN: treat anything not known to be moving as static
            new_state = FSMState.AVOID_STATIC

        if new_state != self.current_state:
            rospy.loginfo("FSM Transition: %s -> %s", self.current_state.name, new_state.name)
            self.current_state = new_state
            if new_state == FSMState.LANE_FOLLOWING:
                self.side = None
        self.state_pub.publish(String(data=self.current_state.name))

        if blocking is None:
            if self.last_local is not None and (rospy.Time.now() - self.last_local_time).to_sec() < self.hold_time:
                self.target_path_pub.publish(self.last_local)  # obstacle flickered: keep the last local path
                return
            self.target_path_pub.publish(self.global_path)
            return

        obs, ahead, offset = blocking
        if obs['id'] != self.side_obstacle_id:
            self.side = None  # a different obstacle: choose the passing side again
        i_car = self.geom.closest(car)
        window = ahead + self.params['avoid_hold'] + self.params['avoid_ramp_out']
        room = None
        if self.walls is not None:
            room = {sd: lateral_room(self.geom, self.walls, i_car, window, sd, self.wall_clearance,
                                     self.params['max_lateral_offset']) for sd in (1.0, -1.0)}
        pts, lateral, side, wanted = build_local_path(self.geom, car, ahead, offset, obs['r'], self.params,
                                                      self.side, room)
        self.side, self.side_obstacle_id = side, obs['id']  # keep passing the same obstacle on the same side
        if side * wanted > side * lateral + 1e-3:
            rospy.logwarn_throttle(2.0, "Not enough room to clear the obstacle (need %.2f m, can shift %.2f m)",
                                   abs(wanted), abs(lateral))

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.map_frame
        for x, y, z in pts:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.last_local, self.last_local_time = path, rospy.Time.now()
        self.target_path_pub.publish(path)


if __name__ == '__main__':
    try:
        BehavioralPlannerFSM()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
