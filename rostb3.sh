export ROS_MASTER_URI=http://missgalaxy.local:11311
export ROS_IP=$(ifconfig wlp5s0 | grep 'inet ' | awk '{ print $2}')
export ROS_HOSTNAME=$(ifconfig wlp5s0 | grep 'inet ' | awk '{ print $2}')
