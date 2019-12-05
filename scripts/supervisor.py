#!/usr/bin/env python

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
#from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String
import tf
import Queue
from visualization_msgs.msg import Marker

# Define the objects that we want to be able to detect. Save them in a set for easy lookup
OBJECTS_OF_INTEREST = {'wine_glass', 'airplane', 'banana', 'cake'} 
HOME_LOCATION = 'airplane'

# Statically define the number of locations that the robot should have explored
NUM_LOCATIONS_EXPLORED = len(OBJECTS_OF_INTEREST)

OBJECT_CONFIDENCE_THESH = 0.5
OBJECT_DISTANCE_THESH = 15

class Mode(Enum):
    """State machine modes. Feel free to change."""
    IDLE = 1
    POSE = 2
    STOP = 3
    CROSS = 4
    NAV = 5
    EXPLORE = 6


class SupervisorParams:

    def __init__(self, verbose=False):
        # If sim is True (i.e. using gazebo), we want to subscribe to
        # /gazebo/model_states. Otherwise, we will use a TF lookup.
        self.use_gazebo = rospy.get_param("sim")

        # How is nav_cmd being decided -- human manually setting it, or rviz
        self.rviz = rospy.get_param("rviz")

        # If using gmapping, we will have a map frame. Otherwise, it will be odom frame.
        self.mapping = rospy.get_param("map")

        # Threshold at which we consider the robot at a location
        self.pos_eps = rospy.get_param("~pos_eps", 0.1)
        self.theta_eps = rospy.get_param("~theta_eps", 0.3)

        # Time to stop at a stop sign
        self.stop_time = rospy.get_param("~stop_time", 3.)

        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.5)

        # Time taken to cross an intersection
        self.crossing_time = rospy.get_param("~crossing_time", 3.)

        if verbose:
            print("SupervisorParams:")
            print("    use_gazebo = {}".format(self.use_gazebo))
            print("    rviz = {}".format(self.rviz))
            print("    mapping = {}".format(self.mapping))
            print("    pos_eps, theta_eps = {}, {}".format(self.pos_eps, self.theta_eps))
            print("    stop_time, stop_min_dist, crossing_time = {}, {}, {}".format(self.stop_time, self.stop_min_dist, self.crossing_time))


class Supervisor:

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('turtlebot_supervisor', anonymous=True)
        self.params = SupervisorParams(verbose=True)

        # Current state
        self.x = 0
        self.y = 0
        self.theta = 0

        # Goal state
        self.x_g = 0
        self.y_g = 0
        self.theta_g = 0

        # Current mode
        self.mode = Mode.EXPLORE
        #self.mode = Mode.IDLE
        self.prev_mode = None  # For printing purposes

        self.delivery_locations = {}
        #for testing
        #self.delivery_locations = {'food1': [1, 5, 2], 'food2': [7,8,5], 'food3':[-0.136055752635, -1.08409714699, -0.716856360435], 'food4': [-0.223887324333, -2.57097697258, -0.656349420547], 'food5':[-0.697493612766, -2.98323106766, 0.987384736538], 'food6': [-1.51829814911, -1.35810863972, 0.725559353828]}
        self.requests = []
        self.vendor_marker_ids = {}
        ########## PUBLISHERS ##########

        # Command pose for controller
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)

        # Command for point navigation
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # For rviz markers
        self.vis_pub = rospy.Publisher('/marker_topic', Marker, queue_size=10)
        ########## SUBSCRIBERS ##########

        # Stop sign detector
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)

        # Listen to object detector and save locations of interest
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.detected_objects_callback, queue_size=10)
        
        rospy.Subscriber('/delivery_request', String, self.request_callback)


        # If using gazebo, we have access to perfect state
        #if self.params.use_gazebo:
            #rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        self.trans_listener = tf.TransformListener()

        # If using rviz, we can subscribe to nav goal click
        if self.params.rviz:
            rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        else:
            self.x_g, self.y_g, self.theta_g = 1.5, -4., 0.
            self.mode = Mode.NAV
        print("finished supervisor init")

    ########## SUBSCRIBER CALLBACKS ##########

    def gazebo_callback(self, msg):
        if "turtlebot3_burger" not in msg.name:
            return

        pose = msg.pose[msg.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        origin_frame = "/map" if self.params.mapping else "/odom"
        print("Rviz command received!")

        try:
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
	    print(nav_pose_origin)
            self.x_g = nav_pose_origin.pose.position.x
            self.y_g = nav_pose_origin.pose.position.y
            quaternion = (nav_pose_origin.pose.orientation.x,
                          nav_pose_origin.pose.orientation.y,
                          nav_pose_origin.pose.orientation.z,
                          nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        self.mode = Mode.NAV

    # def nav_pose_callback(self, msg):
    #     self.mode = Mode.NAV

    def detected_objects_callback(self, msg):
        # Iterate through all of the objects found by the detector
        for name,obj in zip(msg.objects, msg.ob_msgs):
            # Check to see if the object has not already been seen and if it is an object of interest
            if name not in self.delivery_locations.keys() and name in OBJECTS_OF_INTEREST:
                # Ensure that the object detected is of high confidence and close to the robot
                if obj.confidence > OBJECT_CONFIDENCE_THESH and obj.distance < OBJECT_DISTANCE_THESH:
                    # Add the object to the robot list
                    currentPose = Pose2D()
                    currentPose.x = self.x
                    currentPose.y = self.y
                    currentPose.theta = self.theta
                    self.delivery_locations[name] = currentPose
                    print(self.delivery_locations)


                    # Once all objects have been found, then start the request cycle
                    if len(self.delivery_locations.keys()) == NUM_LOCATIONS_EXPLORED:
                        rospy.loginfo("SUPERVISOR: Found all delivery locations.")
                        self.mode = Mode.IDLE

    def request_callback(self, msg):
        rospy.loginfo("SUPERVISOR: Receiving request {}".format(msg.data))
        if msg.data == 'clear':
            self.requests = []
            self.mode = Mode.IDLE
            return

        if len(self.requests) == 0:
            for location in msg.data.split(','):
                if location not in self.delivery_locations.keys(): 
                    rospy.loginfo("SUPERVISOR: Location %s invalid. Skipping.", location)
                    continue
                else:
                    rospy.loginfo("SUPERVISOR: Processing request...")
                    self.requests.append(location)

	    if len(self.requests) > 0:
		self.reqeusts.append(HOME_LOCATION)
                self.go_to_next_request()
                self.mode = Mode.NAV


    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance

        # if close enough and in nav mode, stop
        if dist > 0 and dist < self.params.stop_min_dist and self.mode == Mode.NAV:
            self.init_stop_sign()



    ########## STATE MACHINE ACTIONS ##########

    ########## Code starts here ##########
    # Feel free to change the code here. You may or may not find these functions
    # useful. There is no single "correct implementation".
    def go_to_next_request(self):
        goal_pose = self.delivery_locations[self.requests[0]]
        self.x_g = goal_pose.x
        self.y_g = goal_pose.y
        self.theta_g = goal_pose.theta
        #self.x_g = goal_pose[0]
        #self.y_g = goal_pose[1]
        #self.theta_g = goal_pose[2]



    def go_to_pose(self):
        """ sends the current desired pose to the pose controller """

        pose_g_msg = Pose2D()
        pose_g_msg.x = self.x_g
        pose_g_msg.y = self.y_g
        pose_g_msg.theta = self.theta_g

        self.pose_goal_publisher.publish(pose_g_msg)

    def nav_to_pose(self):
        """ sends the current desired pose to the navigator """

        nav_g_msg = Pose2D()
        nav_g_msg.x = self.x_g
        nav_g_msg.y = self.y_g
        nav_g_msg.theta = self.theta_g

        self.nav_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """

        return abs(x - self.x) < self.params.pos_eps and \
               abs(y - self.y) < self.params.pos_eps and \
               abs(theta - self.theta) < self.params.theta_eps

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.params.stop_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.params.crossing_time)



    ########## Code ends here ##########

    ########## RVIZ VISUALIZATION ##########
    
    def publish_vendor_locs(self):

        for name,loc in self.delivery_locations.items():
            # add marker
            marker = Marker()

            marker.header.frame_id = "/my_frame"
            marker.header.stamp = rospy.Time()

            # so we don't create millions of markers over time
            if name in self.vendor_marker_ids.keys():
                marker.id = self.vendor_marker_ids[name]
            else:
                next_avail_id = len(self.vendor_marker_ids) + 1 # robot is 0, so increment from 1
                marker.id = next_avail_id
                self.vendor_marker_ids[name] = next_avail_id

            marker.type = 1 # cube
            marker.pose.position.x = loc.x
            marker.pose.position.y = loc.y
            #marker.pose.position.x = loc[0]
            #marker.pose.position.y = loc[1]
            marker.pose.position.z = 0

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.7
            marker.scale.y = 0.7
            marker.scale.z = 0.7
            if name == 'home':
                marker.color.a = 0.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0 # Don't forget to set the alpha!
                marker.color.r = 0.5
                marker.color.g = 0.0
                marker.color.b = 1.0
            
            self.vis_pub.publish(marker)


    def publish_robot_loc(self):
        marker = Marker()

        marker.header.frame_id = "base_footprint"
        marker.header.stamp = rospy.Time()

        marker.id = 0 # robot marker id is 0

        marker.type = 0 # arrow

        marker.pose.position.x = 0.0 
        marker.pose.position.y = 0.0 
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.4
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.5
        marker.color.g = 1.0
        marker.color.b = 0.5
        self.vis_pub.publish(marker)

    ########## END RVIZ VISUALIZATION ##########


    ########## STATE MACHINE LOOP ##########

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        if not self.params.use_gazebo:
            try:
                origin_frame = "/map" if self.params.mapping else "/odom"
                translation, rotation = self.trans_listener.lookupTransform(origin_frame, '/base_footprint', rospy.Time(0))
                self.x, self.y = translation[0], translation[1]
                self.theta = tf.transformations.euler_from_quaternion(rotation)[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        # logs the current mode
        if self.prev_mode != self.mode:
            rospy.loginfo("SUPERVISOR: Current mode: %s", self.mode)
            self.prev_mode = self.mode

        ########## Code starts here ##########

        # publish vendor and robot locations
        self.publish_vendor_locs()
        self.publish_robot_loc()
    
        if self.mode == Mode.IDLE:
            pass
            # Send zero velocity
            #self.stay_idle()
            #rospy.loginfo("Idling...")

        elif self.mode == Mode.POSE:
            # Moving towards a desired pose
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                self.mode = Mode.IDLE
            else:
                self.go_to_pose()

        elif self.mode == Mode.STOP:
            # Check to see if the robot has stopped long enough
            if (self.has_stopped()):
                self.init_crossing()

        elif self.mode == Mode.CROSS:
            # Crossing an intersection
            if (self.has_crossed()):
                self.nav_to_pose()

        elif self.mode == Mode.NAV:
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                self.requests.pop(0)
                if len(self.requests) == 0:
                    self.mode = Mode.IDLE
                else:
                    self.go_to_next_request()
            else:
                self.nav_to_pose()

        elif self.mode == Mode.EXPLORE:
            pass

        else:
            raise Exception("This mode is not supported: {}".format(str(self.mode)))

        ############ Code ends here ############

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()


if __name__ == '__main__':
    sup = Supervisor()
    sup.run()
