import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# A 2D state space grid with a set of rectangular obstacles. The grid is fully deterministic
class DetOccupancyGrid2D(object):
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True

    def plot(self, fig_num=0):
        fig = plt.figure(fig_num)
        for obs in self.obstacles:
            ax = fig.add_subplot(111, aspect='equal')
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))

class StochOccupancyGrid2D(object):
    def __init__(self, resolution, width, height, origin_x, origin_y,
                window_size, probs, thresh=0.5):
        self.resolution = resolution 
        self.width = width 
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = probs
        self.window_size = window_size  # 8
        self.thresh = thresh

    def snap_to_grid(self, x):
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    def is_free(self, state):
        # combine the probabilities of each cell by assuming independence
        # of each estimation
        p_total = 1.0
        lower = -int(round((self.window_size-1)/2))
        upper = int(round((self.window_size-1)/2))
        for dx in range(lower,upper+1):
            for dy in range(lower,upper+1):
               
                # original code:
                x, y = self.snap_to_grid([state[0] + dx * self.resolution, state[1] + dy * self.resolution])
                grid_x = int((x - self.origin_x) / self.resolution)
                grid_y = int((y - self.origin_y) / self.resolution)

                if grid_y>0 and grid_x>0 and grid_x<self.width and grid_y<self.height:
                    p_total *= (1.0-max(0.0,float(self.probs[grid_y * self.width + grid_x])/100.0))
                rospy.loginfo("(state_x, state_y)l:  ({}, {})".format(state[0], state[1]))
                rospy.loginfo("(x, y): ({}, {})".format(x,y))
                rospy.loginfo("(grid_x, grid_y): ({}, {})".format(grid_x,grid_y))
                rospy.loginfo("p_total: {}".format(p_total))
                rospy.loginfo("p_total: {}".format(p_total))


                # todo: account for robot's width  
                # robot_width = .15
                # x, y = self.snap_to_grid([state[0] + dx * self.resolution, state[1] + dy * self.resolution])
                # grid_x = int((x - self.origin_x) / self.resolution)
                # grid_y = int((y - self.origin_y) / self.resolution)

                # x_robotcorner1, y_robotcorner1 = self.snap_to_grid([state[0] + robot_width + dx * self.resolution, state[1] + robot_width + dy * self.resolution])
                # grid_x1 = int((x_robotcorner1 - self.origin_x) / self.resolution)
                # grid_y1 = int((y_robotcorner1 - self.origin_y) / self.resolution)
                # corner1_hit = grid_y1>0 and grid_x1>0 and grid_x1<self.width and grid_y1<self.height

                # x_robotcorner2, y_robotcorner2 = self.snap_to_grid([state[0] + robot_width + dx * self.resolution, state[1] - robot_width + dy * self.resolution])
                # grid_x2 = int((x_robotcorner2 - self.origin_x) / self.resolution)
                # grid_y2 = int((y_robotcorner2 - self.origin_y) / self.resolution)
                # corner2_hit = grid_y2>0 and grid_x2>0 and grid_x2<self.width and grid_y2<self.height

                # x_robotcorner3, y_robotcorner3 = self.snap_to_grid([state[0] - robot_width + dx * self.resolution, state[1] + robot_width + dy * self.resolution])
                # grid_x3 = int((x_robotcorner3 - self.origin_x) / self.resolution)
                # grid_y3 = int((y_robotcorner3 - self.origin_y) / self.resolution)
                # corner3_hit = grid_y3>0 and grid_x3>0 and grid_x3<self.width and grid_y3<self.height

                # x_robotcorner2, y_robotcorner2 = self.snap_to_grid([state[0] - robot_width + dx * self.resolution, state[1] - robot_width + dy * self.resolution])
                # grid_x4 = int((x_robotcorner4 - self.origin_x) / self.resolution)
                # grid_y4 = int((y_robotcorner4 - self.origin_y) / self.resolution)
                # corner4_hit = grid_y4>0 and grid_x4>0 and grid_x4<self.width and grid_y4<self.height

                # if corner1_hit or corner2_hit or corner3_hit or corner4_hit:
                #     p_total *= (1.0-max(0.0,float(self.probs[grid_y * self.width + grid_x])/100.0))


        return (1.0-p_total) < self.thresh # the higher p_total is, the more likely cell is free

    def plot(self, fig_num=0):
        fig = plt.figure(fig_num)
        pts = []
        for i in range(len(self.probs)):
            # convert i to (x,y)
            gy = int(i/self.width)
            gx = i % self.width
            x = gx * self.resolution + self.origin_x
            y = gy * self.resolution + self.origin_y
            if not self.is_free((x,y)):
                pts.append((x,y))
        pts_array = np.array(pts)
        plt.scatter(pts_array[:,0],pts_array[:,1],color="red",zorder=15,label='planning resolution')
