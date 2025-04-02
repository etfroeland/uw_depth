from setuptools import setup

setup(
  name='uw_depth_ros',
  version='0.0.1',
  packages=['uw_depth_ros'],
  package_dir={'': 'src'},
  install_requires=['rospy', 'std_msgs', 'sensor_msgs', 'geometry_msgs', 'message_filters', 'numpy', 'torch', 'cv_bridge', 'torchvision', 'opencv-python'],
)
