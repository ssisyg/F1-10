#!/usr/bin/env python3
import rospy
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import LaserScan, PointCloud2
from laser_geometry import LaserProjection
from geometry_msgs.msg import Point, Pose, Twist, Vector3
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy import ndimage
from scipy.spatial import distance
from nav_msgs.msg import OccupancyGrid
from f1tenth_obstacle_avoidance.msg import TrackedObject, TrackedObjectArray

class LidarPerceptionNode:
    def __init__(self):
        rospy.init_node('lidar_perception_node')

        # --- Parameters (CRITICAL TO TUNE!) ---
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.lidar_frame = rospy.get_param('~lidar_frame', 'laser') # Your LiDAR's frame
        self.clustering_eps = rospy.get_param('~clustering_eps', 0.25) # DBSCAN neighborhood radius (meters)
        self.clustering_min_samples = rospy.get_param('~clustering_min_samples', 3) # Min points to form a cluster
        self.use_map_filter = rospy.get_param('~use_map_filter', True)       # drop scan points that hit the known map
        self.map_filter_dist = rospy.get_param('~map_filter_dist', 0.2)      # m, points closer than this to a wall are ignored
        self.min_hits = rospy.get_param('~min_hits', 2)                      # detections needed before a track is published
        self.max_obstacle_size = rospy.get_param('~max_obstacle_size', 1.5) # m, clusters larger than this (e.g. walls) are ignored
        self.classification_static_threshold = rospy.get_param('~classification_static_threshold', 0.1) # m/s
        self.classification_dynamic_threshold = rospy.get_param('~classification_dynamic_threshold', 0.2) # m/s
        
        # --- Internal Variables ---
        self.laser_projector = LaserProjection()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_dist = None  # distance (m) to the nearest occupied cell of the static map
        self.tracks = []
        self.next_track_id = 0
        self.last_time = rospy.Time.now()

        # --- ROS Subscribers/Publishers ---
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        self.obstacles_pub = rospy.Publisher('/lidar_obstacles', TrackedObjectArray, queue_size=1)

    def map_callback(self, msg):
        info = msg.info
        occupied = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width) >= 65
        self.map_dist = ndimage.distance_transform_edt(~occupied) * info.resolution
        self.map_res = info.resolution
        self.map_origin = np.array([info.origin.position.x, info.origin.position.y])
        rospy.loginfo("Static map received (%dx%d): scan points on known walls will be ignored", info.width, info.height)

    def filter_with_map(self, points):
        """Remove points that lie on (or next to) occupied cells of the static map."""
        if not self.use_map_filter or self.map_dist is None or len(points) == 0:
            return points
        ij = np.floor((points - self.map_origin) / self.map_res).astype(int)
        h, w = self.map_dist.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        dist = np.full(len(points), np.inf)  # outside the map: unknown, keep the point
        dist[inside] = self.map_dist[ij[inside, 1], ij[inside, 0]]
        return points[dist > self.map_filter_dist]

    def scan_callback(self, msg):
        current_time = msg.header.stamp
        dt = (current_time - self.last_time).to_sec()
        if dt <= 0.01: # Avoid processing too fast or with invalid dt
            return
        self.last_time = current_time

        # 1. Project LaserScan to PointCloud2
        try:
            cloud_msg = self.laser_projector.projectLaser(msg)
        except Exception as e:
            rospy.logwarn(f"Failed to project laser scan: {e}")
            return

        # 2. Transform PointCloud to the map frame
        try:
            transform = self.tf_buffer.lookup_transform(self.map_frame, cloud_msg.header.frame_id, rospy.Time(0), rospy.Duration(0.1))
            cloud_map = do_transform_cloud(cloud_msg, transform)
            points = np.array(list(pc2.read_points(cloud_map, field_names=("x", "y"), skip_nans=True)))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return
        
        points = self.filter_with_map(points)

        if len(points) == 0:
            self.update_tracks([], dt) # Update with no detections to age out old tracks
            self.publish_obstacles()
            return

        # 3. DBSCAN Clustering
        db = DBSCAN(eps=self.clustering_eps, min_samples=self.clustering_min_samples).fit(points)
        labels = db.labels_
        unique_labels = set(labels)
        
        detections = []
        for k in unique_labels:
            if k == -1: # -1 indicates noise points
                continue
            
            class_mask = (labels == k)
            cluster_points = points[class_mask]
            
            # Calculate cluster centroid and size
            centroid = np.mean(cluster_points, axis=0)
            min_pt = np.min(cluster_points, axis=0)
            max_pt = np.max(cluster_points, axis=0)
            size = max_pt - min_pt

            # Skip wall-sized clusters so the track boundary is not treated as an obstacle
            if np.max(size) > self.max_obstacle_size:
                continue

            detections.append({'centroid': centroid, 'size': size})

        # 4. Update tracks with new detections
        self.update_tracks(detections, dt)

        # 5. Publish the final list of tracked obstacles
        self.publish_obstacles()

    def update_tracks(self, detections, dt):
        # --- Predict new positions for all existing tracks ---
        for track in self.tracks:
            track['kf'].predict()
            track['last_seen'] += 1

        # --- Data Association (simple nearest neighbor) ---
        unmatched_detections_indices = list(range(len(detections)))
        matched_track_indices = []

        if self.tracks and detections:
            track_centroids = np.array([track['kf'].x[:2].flatten() for track in self.tracks])
            detection_centroids = np.array([d['centroid'] for d in detections])
            dist_matrix = distance.cdist(track_centroids, detection_centroids)
            
            # Greedily match nearest neighbors
            for i, track in enumerate(self.tracks):
                if not unmatched_detections_indices:
                    break
                if i in matched_track_indices:
                    continue
                
                min_dist_idx = np.argmin(dist_matrix[i, unmatched_detections_indices])
                det_idx = unmatched_detections_indices[min_dist_idx]
                
                if dist_matrix[i, det_idx] < 1.0: # Association distance threshold (meters)
                    track['kf'].update(detections[det_idx]['centroid'])
                    track['size'] = detections[det_idx]['size']
                    track['last_seen'] = 0
                    track['hits'] += 1
                    matched_track_indices.append(i)
                    unmatched_detections_indices.pop(min_dist_idx)

        # --- Create new tracks for unmatched detections ---
        for det_idx in unmatched_detections_indices:
            det = detections[det_idx]
            kf = self.create_kalman_filter(dt)
            kf.x[:2] = det['centroid'].reshape(2, 1)
            
            new_track = {
                'id': self.next_track_id,
                'kf': kf,
                'size': det['size'],
                'last_seen': 0,
                'hits': 1,
                'class_name': "UNKNOWN"
            }
            self.tracks.append(new_track)
            self.next_track_id += 1

        # --- Classify and clean up old tracks ---
        updated_tracks = []
        for track in self.tracks:
            if track['last_seen'] > 5: # Remove track if not seen for 5 frames
                continue

            # Classification logic based on velocity
            velocity = np.linalg.norm(track['kf'].x[2:])
            if velocity < self.classification_static_threshold:
                track['class_name'] = "STATIC"
            elif velocity > self.classification_dynamic_threshold:
                track['class_name'] = "DYNAMIC"
            else:
                track['class_name'] = "UNKNOWN"
            
            updated_tracks.append(track)
        self.tracks = updated_tracks
        
    def create_kalman_filter(self, dt):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.zeros((4, 1))      # state [x, y, vx, vy]
        kf.F = np.array([[1, 0, dt, 0],   # state transition matrix
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],   # measurement function
                         [0, 1, 0, 0]])
        kf.P *= 10.   # covariance matrix
        kf.R = np.diag([0.1, 0.1]) # measurement noise
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1, block_size=2, order_by_dim=False) # process noise for [x, y, vx, vy]
        return kf
        
    def publish_obstacles(self):
        obstacle_array_msg = TrackedObjectArray()
        obstacle_array_msg.header.stamp = self.last_time
        obstacle_array_msg.header.frame_id = self.map_frame

        for track in self.tracks:
            if track['hits'] < self.min_hits:
                continue  # not confirmed yet
            obj = TrackedObject()
            obj.id = track['id']
            obj.class_name = track['class_name']

            pos = track['kf'].x[:2].flatten()
            vel = track['kf'].x[2:].flatten()
            
            obj.pose.position = Point(x=pos[0], y=pos[1], z=0)
            # We don't have orientation data, so leave it as default
            obj.velocity.linear = Vector3(x=vel[0], y=vel[1], z=0)
            obj.size = Vector3(x=max(0.1, track['size'][0]), y=max(0.1, track['size'][1]), z=0.2) # Use a default z and minimum size

            obstacle_array_msg.objects.append(obj)
            
        self.obstacles_pub.publish(obstacle_array_msg)

if __name__ == '__main__':
    try:
        node = LidarPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass