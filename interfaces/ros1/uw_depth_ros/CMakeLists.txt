cmake_minimum_required(VERSION 3.0.2)
project(uw_depth_ros)

# Find catkin and other dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

# Python setup
catkin_python_setup()

# Declare a catkin package
catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs
)

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)
