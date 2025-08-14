# F1/10 LiDAR Obstacle Avoidance System

This ROS project implements a complete pipeline for autonomous obstacle avoidance and overtaking for an F1/10 race car, using only 2D LiDAR data. The system can detect, track, and classify obstacles as static or dynamic, enabling intelligent decision-making for complex race scenarios.

## System Architecture

The system is built on a modular, three-node architecture that separates perception, planning, and control. This design makes the system easy to debug, maintain, and extend.

The data flows through the system as follows:

`Raw LiDAR Scan` → **Perception Node** → `Tracked Obstacles List` → **Behavioral Planner (FSM)** → `Local Target Path` → **Path Following Controller** → `Vehicle Commands (/cmd_vel)`

-----

## How It Works

The core logic is divided into three distinct steps:

### 1\. Detection: Euclidean Clustering

First, the system processes the raw LiDAR scan data to group discrete points into individual "objects."

  * **Recommended Algorithm:** Euclidean Cluster Extraction
  * **How it works:** This algorithm iterates through all the laser points. It starts by selecting an arbitrary point and finds all other points within a specified radius, grouping them into a "cluster." It then recursively finds the neighbors of these newly added points until the cluster can no longer expand. This process is repeated until the entire point cloud is segmented into several independent clusters, each representing a potential obstacle.
  * **Input:** Raw `/scan` data (which is first converted to the `PointCloud2` format).
  * **Output:** A list of obstacle clusters. Each cluster is a collection of points, from which we can calculate its centroid, dimensions, and other properties.
  * **Implementation:** The `pcl` (Point Cloud Library) provides highly efficient implementations of this algorithm, which we leverage through Python wrappers.

### 2\. Tracking & Classification: Kalman Filter

Once we have a collection of obstacle clusters, we need to track them over time to understand their motion.

  * **Recommended Algorithm:** Kalman Filter Tracker
  * **How it works:** This is a classic algorithm for multi-object tracking.
      * **Data Association:** For each cluster detected in the current frame, the algorithm attempts to match it with a tracked object from the previous frame (typically based on the nearest neighbor principle).
      * **Prediction:** Based on the previous state (position and velocity) of each tracked object, the Kalman Filter predicts its expected position in the current frame.
      * **Update:** The actual observation from the current frame (the position of the matched cluster) is used to correct the prediction. This results in a smoothed, more accurate estimate of the object's position and velocity.
  * **The Key Role—Classification:** Through tracking, we obtain a stable velocity estimate for each obstacle. Classification then becomes simple:
      * If an object's velocity remains below a threshold (e.g., `0.1 m/s`), it's classified as **STATIC**.
      * If its velocity is above the threshold, it's classified as **DYNAMIC** (i.e., an opponent's vehicle).

### 3\. Decision & Action: Finite State Machine (FSM)

This node acts as the system's brain. Based on the classified obstacles, it decides which behavior to execute.

  * **Recommended Algorithm:** State Machine-based Local Path Planning
  * **State Machine Logic:**
      * **`LANE_FOLLOWING` (Default State):**
          * **Behavior:** Follow the pre-computed global path from `/racing_line`.
          * **Transitions:** Switches to `AVOID_STATIC` or `OVERTAKE_DYNAMIC` when an obstacle is detected on the path ahead.
      * **`AVOID_STATIC` (Static Obstacle Avoidance):**
          * **Behavior:** Dynamically generates a smooth local path (a spline curve) that maneuvers around the obstacle's known position and safely merges back onto the global path.
          * **Transitions:** Switches back to `LANE_FOLLOWING` after the obstacle has been successfully passed.
      * **`OVERTAKE_DYNAMIC` (Dynamic Overtake):**
          * **Behavior:** This is the most critical state. It gets the opponent's current position and estimated velocity to **predict its future trajectory**. It then plans a safe overtake path around this *predicted* trajectory. The system continuously monitors for safety and can abort the overtake if space becomes insufficient.
          * **Transitions:** Switches back to `LANE_FOLLOWING` after the overtake is complete.

-----

## Getting Started

### Prerequisites

  * **ROS Version:** ROS Noetic (Ubuntu 20.04) is recommended. Melodic may work with minor adjustments.
  * **Python Dependencies:** You will need to install several Python libraries.
    ```bash
    pip install numpy scikit-learn filterpy scipy
    ```

### Installation

1.  **Clone the Repository:**
    Clone this repository into the `src` folder of your Catkin workspace.

    ```bash
    cd ~/your_catkin_ws/src
    git clone https://github.com/YourUsername/F1-10.git
    ```

2.  **Custom Messages:**
    This package uses custom ROS messages (`TrackedObject.msg`, `TrackedObjectArray.msg`) located in the `msg/` directory. The `CMakeLists.txt` and `package.xml` are already configured to build them.

3.  **Build the Package:**
    Navigate to the root of your workspace and build the project.

    ```bash
    cd ~/your_catkin_ws
    catkin_make
    ```

4.  **Source the Environment:**
    Source the setup file to make the new package available to ROS.

    ```bash
    source devel/setup.bash
    ```

-----

## Usage

The entire system can be launched with a single command. This will start all three nodes (Perception, FSM, and Controller) with the default parameters.

```bash
roslaunch f1tenth_obstacle_avoidance lidar_planner.launch
```

-----

## Configuration & Tuning

> **Important:** This is a framework, not a plug-and-play product. Performance depends heavily on proper tuning for your specific vehicle and environment.

All key parameters are exposed in `launch/lidar_planner.launch` for easy tuning without modifying the source code.

Key parameters include:

  * **`clustering_eps`**: The radius for DBSCAN clustering. A smaller value separates close objects; a larger value groups them together.
  * **`lookahead_distance`**: The lookahead distance for the Pure Pursuit controller. Affects steering smoothness and aggressiveness.
  * **`lateral_safety_margin`**: How much side clearance the car will try to maintain when avoiding an obstacle.
  * **Topics**: You can easily remap the default topics (`/scan`, `/amcl_pose`, etc.) in the launch file to match your sensor and localization setup.
