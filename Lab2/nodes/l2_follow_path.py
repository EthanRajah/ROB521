#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros
from skimage.draw import disk

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils


TRANS_GOAL_TOL = .1  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .3  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.025, 0.13, 0.26]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'path.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
# TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        print(self.map_odom_tf)

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        map = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        self.map_resolution = round(map.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        print(self.map_origin)
        self.map_nonzero_idxes = np.argwhere(self.map_np)
        print(map)


        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, 'path.npy')).T
        # self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)

            print("TO DO: Propogate the trajectory forward, storing the resulting points in local_paths!")
            for i, opt in enumerate(self.all_opts):
                trajectory = self.trajectory_rollout(opt[0], opt[1], local_paths[0, i, 2]) # 3xN trajectory set of waypoints
                # Store in local_paths[1:N, i, :]
                local_paths[:, i, :] = trajectory.T + local_paths[0,i].reshape(1,3)

            # Robot current coords in pixels
            cur_y, cur_x = self.pos_in_map_pix[0], self.pos_in_map_pix[1]

            # check all trajectory points for collisions
            # first find the closest collision point in the map to each local path point
            local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution
            valid_opts = range(self.num_opts)
            local_paths_lowest_collision_dist = np.ones(self.num_opts) * 50

            # Range for comparing current point to surrounding obstacles
            check_radius = 10
            collision_indices = np.where(np.logical_and(abs(self.map_nonzero_idxes[:,0] - cur_x) < check_radius, abs(self.map_nonzero_idxes[:,1] - cur_y) < check_radius))
            collision_range = self.map_nonzero_idxes[collision_indices]

            print("TO DO: Check the points in local_path_pixels for collisions")
            for opt in range(local_paths_pixels.shape[1]):
                for timestep in range(local_paths_pixels.shape[0]):
                    # For each pt in trajectory, check for a collision in a circle around the robot of radius self.collision_radius_pix
                    cur_pixel = local_paths_pixels[timestep, opt]
                    robot_footprint = self.points_to_robot_circle(cur_pixel)
                    if np.any(self.map_nonzero_idxes[robot_footprint[0], robot_footprint[1]]):
                        # This is a collision so remove the option
                        valid_opts.remove(opt)
                        break
                    else:
                        # Get the closest distance to an obstacle
                        dists = cityblock(collision_range.T, cur_pixel)
                        closest_dist = np.min(dists)
                        if closest_dist < local_paths_lowest_collision_dist[opt]:
                            local_paths_lowest_collision_dist[opt] = closest_dist
                        


            # # remove trajectories that were deemed to have collisions
            # print("TO DO: Remove trajectories with collisions!")

            # calculate final cost and choose best option
            print("TO DO: Calculate the final cost and choose the best control option!")
            # Initialize to some high cost
            final_cost = np.zeros(valid_opts)
            for opt in valid_opts:
                if (abs(local_paths[0][opt][0] - self.cur_goal[0]) < MIN_TRANS_DIST_TO_USE_ROT) and (abs(local_paths[0][opt][1] - self.cur_goal[1]) < MIN_TRANS_DIST_TO_USE_ROT):
                    final_cost[opt] = (local_paths[-1][opt][0] - self.cur_goal[0])**2 + (local_paths[75][opt][1] - self.cur_goal[1])**2 + (local_paths[-1][opt][2] - self.cur_goal[2])**2
                else:
                    final_cost[opt] = (local_paths[-1][opt][0] - self.cur_goal[0])**2 + (local_paths[-1][opt][1] - self.cur_goal[1])**2 


            if final_cost.size == 0:  # hardcoded recovery if all options have collision
                control = [-.1, 0]
            else:
                best_opt = valid_opts[final_cost.argmin()]
                control = self.all_opts[best_opt]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_opt], 'map'))

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            # print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
            #     control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")

    def trajectory_rollout(self, vel, rot_vel, theta_i):
            # Given your chosen velocities determine the trajectory of the robot for your given timestep
            # The returned trajectory should be a series of points to check for collisions
            """Implement a way to rollout the controls chosen"""
            trajectory = np.array([[],[],[]])                          # initialize array
            t = np.linspace(0.1, CONTROL_HORIZON, self.horizon_timesteps+1)
            # If the robot is not rotating, use the unicycle model with zero angle moves 
            if rot_vel == 0:
                # Cant use exact equations, approximate with first order differential
                x_I = [np.around((vel*t*np.cos(theta_i)),2)]
                y_I = [np.around((vel*t*np.sin(theta_i)),2)]
                theta_I = [np.zeros(self.horizon_timesteps+1)]
            else:
                # Integrate xdot = v*costheta and ydot = v*sintheta: 
                # x = (v/w)*sin(wt + theta_i) - (v/w)*sin(theta_i)
                x_I = [np.around((vel/rot_vel)*(np.sin(rot_vel*t + theta_i)-np.sin(theta_i)), 4)]
                # y = -(v/w)*cos(wt + theta_i) + (v/w)*cos(theta_i)
                y_I = [np.around((vel/rot_vel)*(np.cos(theta_i)-np.cos(rot_vel*t + theta_i)), 4)]
                # theta = rot_vel*t
                theta_I = [np.around(rot_vel*t, 4)]

            trajectory = np.vstack((x_I, y_I, theta_I))
            # Return 3xN trajectory set of waypoints
            return trajectory

    def point_to_cell(self, point):
            #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
            #point is a 2 by N matrix of points of interest
            """Implement a method to get the map cell the robot is currently occupying"""
            # Use self.bounds to map from point frame to occupancy map (pixel coordinates) frame (lower left corner)
            offset = np.array(self.bounds[:, 0]).reshape(2, 1)
            resolution = self.map_resolution # self.map_settings_dict["resolution"]
            adjusted_pts = point - offset
            # Since map y pixel coordinates are wrt the upper left corner, need to adjust y point coordinates (currently wrt lower left corner) based on map height
            adjusted_pts[1, :] = self.map_np.shape[1] * resolution - adjusted_pts[1, :] 
            # Transform the point to map frame, then use resolution to get cell indices
            cell = (adjusted_pts / resolution).astype(int)
            # Return new 2xN matrix of cell indices
            return cell

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        """Implement a method to get the pixel locations of the robot path"""
        # Get the robot radius in cells (pixel units)
        robot_radius_cells = self.collision_radius_pix # self.robot_radius / self.map_settings_dict["resolution"]
        # Convert points to cells
        cells = self.point_to_cell(points)
        # Use cell indices to create a circle around the robot for determining the cells that it occupies
        robot_footprint = [[], []]
        for cell in cells.T:
            rr, cc = disk(cell, robot_radius_cells, shape=self.map_np.shape) # Pixel coordinates of robot footprint
            robot_footprint = np.hstack((robot_footprint, np.vstack((rr, cc))))
        # Convert float64 robot_footprint to int
        robot_footprint = robot_footprint.astype(int)
        return robot_footprint

if __name__ == '__main__':#
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass