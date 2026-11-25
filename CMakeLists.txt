cmake_minimum_required(VERSION 3.0.2)
project(f1tenth_obstacle_avoidance)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  laser_geometry
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  tf2_sensor_msgs
)

add_message_files(
  FILES
  TrackedObject.msg
  TrackedObjectArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs geometry_msgs nav_msgs sensor_msgs tf2_ros tf2_sensor_msgs laser_geometry
)

catkin_install_python(PROGRAMS
  nodes/lidar_perception.py
  nodes/behavioral_planner_fsm.py
  nodes/pursuit_controller.py
  nodes/cmdvel_to_vesc.py
  nodes/waypoint_recorder.py
  nodes/racing_line_publisher.py
  nodes/follow_the_gap.py
  nodes/pose_relay.py
  nodes/vesc_odom.py
  nodes/fake_car.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config maps
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
