#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
from visualization_msgs.msg import Marker

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig
from asl_turtlebot.msg import DetectedObject, DetectedObjectList

# Define the objects that we want to be able to detect. Save them in a set for easy lookup
#OBJECTS_OF_INTEREST = {'wine_glass', 'airplane', 'banana', 'cake'} 
OBJECTS_OF_INTEREST = {'519', '345'}
HOME_LOCATION = '345'

# Statically define the number of locations that the robot should have explored
NUM_LOCATIONS_EXPLORED = len(OBJECTS_OF_INTEREST)

OBJECT_CONFIDENCE_THESH = 0.5
OBJECT_DISTANCE_THESH = 8.5

# state machine modes, not all implemented

class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    ROLL = 4
    EXPLORE = 5



class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)
        self.mode = Mode.EXPLORE
        self.delivery_locations = {}
        self.requests = []
        self.vendor_marker_ids = {}

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution =  0.1
        self.plan_horizon = 20 # NOTE: Changed this to incraese the plan horizon for testing

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.,0.]
        
        # Robot limits
        self.v_max = 0.2    # maximum velocity
        self.om_max = 0.4   # maximum angular velocity

        self.v_des = 0.12   # desired cruising velocity
        self.theta_start_thresh = 0.05   # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.2     # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.3
        self.at_thresh = 0.03
        #self.at_thresh_theta = 0.05
        self.at_thresh_theta = 0.1

        # trajectory smoothing
        self.spline_alpha = 0.15
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.

        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)
        self.pose_controller = PoseController(0., 0., 0., self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher('/cmd_smoothed_path', Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher('/cmd_smoothed_path_rejected', Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)


        #stop sign parameters
        self.stop_min_dist = 2
        self.stop_sign_roll_start = 0

        self.stop_sign_stop_time = 4
        self.wait_time = 1
        self.first_seen_time = -1


        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
        #For the stop sign
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)

        # For ETA rviz markers
        self.vis_pub = rospy.Publisher('/eta', Marker, queue_size=10)
        # For landmark rviz markers
        self.vis_pub = rospy.Publisher('/marker_topic', Marker, queue_size=10)

        # Listen to object detector and save locations of interest
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.detected_objects_callback, queue_size=10)
        
        rospy.Subscriber('/delivery_request', String, self.request_callback)
        print "finished init"

        
    def dyn_cfg_callback(self, config, level):
        rospy.loginfo("NAVIGATOR: Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config))
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if data.x != self.x_g or data.y != self.y_g or data.theta != self.theta_g:
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            #self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  #self.map_width,
                                                  #self.map_height,
                                                  #self.map_origin[0],
                                                  #self.map_origin[1],
                                                  #8, 
                                                  #self.map_probs)
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  10, # NOTE: Made the window size larger
                                                  self.map_probs) # NOTE: Made the probability lower

            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("NAVIGATOR: replanning because of new map")
                self.replan() # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)


    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance
        rospy.loginfo("Detected stop sign")
        rospy.loginfo(dist)


        if dist > 0 and dist < self.stop_min_dist and self.mode == Mode.TRACK and msg.confidence > OBJECT_CONFIDENCE_THESH:
            if self.first_seen_time == -1:
                self.first_seen_time = rospy.get_rostime()
            elif rospy.get_rostime() - self.first_seen_time > rospy.Duration.from_sec(self.wait_time) and self.mode == Mode.TRACK:
                rospy.loginfo("Initializing roll and changing v_max and v_des")
                self.init_stop_sign()

    def detected_objects_callback(self, msg):
        # Iterate through all of the objects found by the detector
        for name,obj in zip(msg.objects, msg.ob_msgs):
            # Check to see if the object has not already been seen and if it is an object of interest
            if name not in self.delivery_locations.keys() and name in OBJECTS_OF_INTEREST:
                # Ensure that the object detected is of high confidence and close to the robot
                if obj.confidence < OBJECT_CONFIDENCE_THESH and obj.distance < OBJECT_DISTANCE_THESH:
                    # Add the object to the robot list
                    currentPose = Pose2D()
                    currentPose.x = self.x
                    currentPose.y = self.y
                    currentPose.theta = self.theta
                    self.delivery_locations[name] = currentPose
                    print(self.delivery_locations)


                    # Once all objects have been found, then start the request cycle
                    if len(self.delivery_locations.keys()) == NUM_LOCATIONS_EXPLORED:
                        rospy.loginfo("NAVIGATOR: Found all delivery locations.")
                        self.switch_mode(Mode.IDLE)

    def request_callback(self, msg):
        rospy.loginfo("NAVIGATOR: Receiving request {}".format(msg.data))
        if msg.data == 'clear':
            self.requests = []
            self.switch_mode(Mode.IDLE)
            return

        if len(self.requests) == 0:
            for location in msg.data.split(','):
                if location not in self.delivery_locations.keys(): 
                    rospy.loginfo("NAVIGATOR: Location %s invalid. Skipping.", location)
                    continue
                else:
                    rospy.loginfo("NAVIGATOR: Processing request...")
                    self.requests.append(location)

	    if len(self.requests) > 0:
                self.requests.append(HOME_LOCATION)
                self.go_to_next_request()

    def init_stop_sign(self):
        self.stop_sign_roll_start = rospy.get_rostime()
        self.traj_controller.V_max = 0.05
        self.pose_controller.V_max = 0.05
        self.v_des = 0.04
        self.switch_mode(Mode.ROLL)


    def has_rolled(self):
        return (self.mode == Mode.ROLL and rospy.get_rostime() - self.stop_sign_roll_start > rospy.Duration.from_sec(self.stop_sign_stop_time))

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta)

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh)
        
    def close_to_plan_start(self):
        return (abs(self.x - self.plan_start[0]) < self.start_pos_thresh and abs(self.y - self.plan_start[1]) < self.start_pos_thresh)

    def snap_to_grid(self, x):
        return (self.plan_resolution*round(x[0]/self.plan_resolution), self.plan_resolution*round(x[1]/self.plan_resolution))

    def switch_mode(self, new_mode):
        rospy.loginfo("NAVIGATOR: Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i,0]
            pose_st.pose.position.y = traj[i,1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.TRACK or self.mode == Mode.ROLL:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.EXPLORE:
            return
        else:
            V = 0.
            om = 0.

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def go_to_next_request(self):
        goal_pose = self.delivery_locations[self.requests[0]]
        if goal_pose.x != self.x_g or goal_pose.y != self.y_g or goal_pose.theta != self.theta_g:
            rospy.loginfo("NAVIGATOR: Going to a new location")
            self.x_g = goal_pose.x
            self.y_g = goal_pose.y
            self.theta_g = goal_pose.theta
            self.replan()

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("NAVIGATOR: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

        rospy.loginfo("NAVIGATOR: computing navigation plan")
        success =  problem.solve()
        if not success:
            rospy.loginfo("NAVIGATOR: Planning failed")
            return
        rospy.loginfo("NAVIGATOR: Planning Succeeded")

        planned_path = problem.path
        

        # Check whether path is too short
        try:
            if len(planned_path) < 4:
                rospy.loginfo("NAVIGATOR: Path too short to track")
                self.switch_mode(Mode.PARK)
                return
        except:
            rospy.loginfo("NAVIGATOR: len(path_planned) attempt failed. Switching to park. Try a new path.")
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
			t_remaining_curr = self.current_plan_duration - self.get_current_plan_time()

			# Estimate duration of new trajectory
			th_init_new = traj_new[0,2]
			th_err = wrapToPi(th_init_new - self.theta)
			t_init_align = abs(th_err/self.om_max)
			t_remaining_new = t_init_align + t_new[-1]

			if t_remaining_new > t_remaining_curr:
				rospy.loginfo("NAVIGATOR: New plan rejected (longer duration than current plan)")
				self.publish_smoothed_path(traj_new, self.nav_smoothed_path_rej_pub)
				return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0,2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("NAVIGATOR: Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("NAVIGATOR: Ready to track")
        self.switch_mode(Mode.TRACK)

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():

            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                rospy.loginfo("NAVIGATOR: waiting for state info")
                if (self.mode != Mode.EXPLORE):
                    self.switch_mode(Mode.IDLE)
                print e
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("NAVIGATOR: replanning because far from start")
                    self.replan()
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("NAVIGATOR: replanning because out of time")
                    self.replan() # we aren't near the goal but we thought we should have been, so replan
                self.displayETA()
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    # Forget about the current goal
                    self.requests.pop(0)
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)
                    # Check to see if there are more places that must be visited 
                    if len(self.requests) > 0:
                        self.go_to_next_request()
            elif self.mode == Mode.ROLL:
                rospy.loginfo("Rolling...")
                if self.has_rolled():
                    rospy.loginfo("Finishd rolling")
                    self.traj_controller.V_max = 0.2
                    self.pose_controller.V_max = 0.2
                    self.v_des = 0.12
                    self.mode = Mode.TRACK
                    self.first_seen_time = -1

            
            # publish vendor and robot locations
            self.publish_vendor_locs()
            self.publish_robot_loc()


            self.publish_control()
            rate.sleep()

    ########## RVIZ VISUALIZATION ##########
    
    def publish_vendor_locs(self):

        for name,loc in self.delivery_locations.items():
            # add marker
            marker = Marker()

            marker.header.frame_id = "/map"
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
            marker.pose.position.z = 0.1

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            if name == HOME_LOCATION:
                marker.color.a = 1.0
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

    def displayETA(self):
        marker = Marker()

        marker.header.frame_id = "base_footprint"
        marker.header.stamp = rospy.Time()

        marker.id = 1234

        marker.type = Marker.TEXT_VIEW_FACING

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.4
        marker.scale.y = 0.1
        marker.scale.z = 0.2


        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.02
        marker.color.g = 0.84
        marker.color.b = 1.0
        marker.text = "ETA to destination: {}".format(self.current_plan_duration)
        self.vis_pub.publish(marker)

if __name__ == '__main__':    
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
