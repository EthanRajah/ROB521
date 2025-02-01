#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag


def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist, map_filename)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        """Sample point to drive towards"""
        # Sample a point within the bounds of the map
        x = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])
        y = np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])
        return np.array([x, y])
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        """Check that nodes are not duplicates"""
        for node in self.nodes:
            if np.array_equal(node.point, point):
                return True
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        """Implement a method to get the closest node to a sapled point"""
        min_dist = np.inf
        closest_node_id = -1
        for i, node in enumerate(self.nodes):
            dist = np.linalg.norm(node.point - point)
            if dist < min_dist:
                min_dist = dist
                closest_node_id = i
        return closest_node_id
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        """Implment a method to simulate a trajectory given a sampled point"""
        current_node = node_i
        iteration_counter = 1

        # Do one iteration of the controller to get the initial trajectory
        vel, rot_vel = self.robot_controller(current_node, point_s)
        robot_traj = self.trajectory_rollout(vel, rot_vel)
        current_node = robot_traj[:, -1]

        while (np.linalg.norm(current_node[0:2] - point_s) > self.stopping_dist):
            vel, rot_vel = self.robot_controller(current_node, point_s)
            current_traj = self.trajectory_rollout(vel, rot_vel) + current_node.reshape(3, 1)
            print("Current node: ", current_node)
            robot_traj = np.hstack((robot_traj, current_traj))
            current_node = robot_traj[:, -1]
            iteration_counter += 1
        print("Number of iterations in simulation: ", iteration_counter)
        # Return 3xN trajectory
        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        """Implement a control scheme to drive you towards the sampled point"""
        # Get the current robot state
        x, y, theta = node_i[0], node_i[1], node_i[2]
        # Get the desired robot state
        x_s, y_s = point_s[0], point_s[1]
        # Compute the error between the current and desired robot (x,y) states
        dx = x_s - x
        dy = y_s - y
        target_dist = np.sqrt(dx**2 + dy**2)
        # Compute the angle error between the current and desired robot angle states
        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - theta
        # Compute control velocities
        vel_gain = 0.4
        rot_gain = 0.5
        vel = min(target_dist * vel_gain, self.vel_max)
        rot_vel = min(self.rot_vel_max, max(-self.rot_vel_max, rot_gain * angle_error))
        return vel, rot_vel
    
    def trajectory_rollout(self, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        """Implement a way to rollout the controls chosen"""
        dt = self.timestep # Trajectory over one substep time
        trajectory = np.zeros((3, self.num_substeps))
        x, y, theta = 0, 0, 0
        for i in range(self.num_substeps):
            # Update state using unicycle model equations:
            # dx/dt = vcos(theta)
            # dy/dt = vsin(theta)
            # dtheta/dt = omega
            x = x + vel * np.cos(theta) * dt
            y = y + vel * np.sin(theta) * dt
            theta = theta + rot_vel * dt
            trajectory[0, i] = x
            trajectory[1, i] = y
            trajectory[2, i] = theta
        return trajectory
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        """Implement a method to get the map cell the robot is currently occupying"""
        # Use self.bounds to map from point frame to occupancy map (pixel coordinates) frame (lower left corner)
        offset = np.array(self.bounds[:, 0]).reshape(2, 1)
        resolution = self.map_settings_dict["resolution"]
        adjusted_pts = point - offset
        # Since map y pixel coordinates are wrt the upper left corner, need to adjust y point coordinates (currently wrt lower left corner) based on map height
        adjusted_pts[1, :] = self.map_shape[1] * resolution - adjusted_pts[1, :] 
        # Transform the point to map frame, then use resolution to get cell indices
        cell = (adjusted_pts / resolution).astype(int)
        # Return new 2xN matrix of cell indices
        return cell

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        """Implement a method to get the pixel locations of the robot path"""
        # Get the robot radius in cells (pixel units)
        robot_radius_cells = self.robot_radius / self.map_settings_dict["resolution"]
        # Convert points to cells
        cells = self.point_to_cell(points)
        # Use cell indices to create a circle around the robot for determining the cells that it occupies
        robot_footprint = [[], []]
        for cell in cells.T:
            rr, cc = disk(cell, robot_radius_cells, shape=self.map_shape) # Pixel coordinates of robot footprint
            robot_footprint = np.hstack((robot_footprint, np.vstack((rr, cc))))
        return robot_footprint
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        goal_reached = False
        max_iter = 1000
        iter = 0
        while not goal_reached or iter <= max_iter: #Most likely need more iterations than this to complete the map!
            print("RRT Iteration: ", iter)
            #Sample map space
            point = self.sample_map_space()
            print("Sampled Point: ", point)

            #Get the closest point
            closest_node_id = self.closest_node(point)
            print("Closest Node: ", self.nodes[closest_node_id].point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            """Check for collisions and add safe points to list of nodes."""
            # Get the robot footprint for the trajectory points
            robot_footprint = self.points_to_robot_circle(trajectory_o[0:2, :])
            # Check if any of the robot footprint points are in an occupied cell
            if np.any(self.occupancy_map[robot_footprint[0, :], robot_footprint[1, :]]):
                continue
            # Add the point to the list of nodes
            self.nodes.append(Node(point, closest_node_id, 0))
            # Update the parent node's children list
            self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)
            #Check if goal has been reached
            """Check if at goal point."""
            if np.linalg.norm(self.nodes[-1].point[0:2] - self.goal_point) < self.stopping_dist:
                print("Goal Reached.")
                goal_reached = True
            iter += 1

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    #map_filename = "willowgarageworld_05res.png"
    #map_setings_filename = "willowgarageworld_05res.yaml"
    map_filename = "myhal.png"
    map_setings_filename = "myhal.yaml"

    #robot information
    goal_point = np.array([[7], [0]]) #m
    #goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_planning()
    # nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
