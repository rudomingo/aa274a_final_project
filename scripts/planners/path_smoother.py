import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    path = np.array(path)
    x_values = path[:,0]
    y_values = path[:,1]
    times = []
    times.append(0)
    for i in range(path.shape[0]-1):
        distance = np.linalg.norm(path[i+1] - path[i])
        times.append((distance/V_des) + times[-1])

    t_smoothed = np.arange(0, times[-1], dt)

    tck_x = scipy.interpolate.splrep(times, x_values, s=alpha)
    tck_y = scipy.interpolate.splrep(times, y_values, s=alpha)
    x = scipy.interpolate.splev(t_smoothed, tck_x)
    x_dot = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    x_ddot = scipy.interpolate.splev(t_smoothed, tck_x, der=2)
    y = scipy.interpolate.splev(t_smoothed, tck_y)
    y_dot = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    y_ddot = scipy.interpolate.splev(t_smoothed, tck_y, der=2)
    theta = np.arctan2(y_dot, x_dot)

    traj_smoothed = np.zeros((t_smoothed.shape[0],7))
    traj_smoothed[:,0] = x
    traj_smoothed[:,1] = y
    traj_smoothed[:,2] = theta
    traj_smoothed[:,3] = x_dot
    traj_smoothed[:,4] = y_dot
    traj_smoothed[:,5] = x_ddot
    traj_smoothed[:,6] = y_ddot
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
