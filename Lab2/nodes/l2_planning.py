#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter_aa
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
        self.goal_reached = False

        if map_filename == "simple_map.png":
            self.map = self.occupancy_map[:, :, 0]
        else:
            self.map = self.occupancy_map

        self.goal_point = goal_point  # m
        self.stopping_dist = stopping_dist  # m

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0] #-21
        self.bounds[1, 0] = self.map_settings_dict["origin"][1] #-49.25
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"] #60-ish
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"] #31 ish

        if "obstacles" in self.map_settings_dict:
            for box in self.map_settings_dict["obstacles"].keys():
                print(self.map_settings_dict["obstacles"][box])
                self.map_settings_dict["obstacles"][box] = [val / self.map_settings_dict["resolution"] for val in
                                                            self.map_settings_dict["obstacles"][box]]

            print(self.map_settings_dict["obstacles"])


        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        # Control Parameters
        self.kP = 0.7
        self.kI = 0.001
        self.kD = 0.05
        self.linear_prop = 20.0
        self.prev_heading_error = 0.0
        self.accum_heading_error = 0.0

        # Sample parameters
        self.sample_counter = 0
        self.sample_frequency = 15

        # for willow garage map adjusted bounds
        self.x_min = 0
        self.x_max = 44
        self.y_min = -45
        self.y_max = 10
        self.res_round = 2 
        self.samp_count = 0 

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
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist, map_name=map_filename)
        return

    def sample_map_space(self):
        # Return an [x,y] coordinate to drive the robot towards
        # print("TO DO: Sample point to drive towards")
        self.samp_count += 1

        # every nth sample will be the goal itself
        if self.samp_count % self.sample_frequency == 0:
            # print('Goal point sampled!')
            return self.goal_point

        x = round((self.x_max - self.x_min) * np.random.random_sample() + self.x_min, self.res_round)
        y = round((self.y_max - self.y_min) * np.random.random_sample() + self.y_min, self.res_round)

        return np.array([[x], [y]])
    
    def check_if_duplicate(self, point):
        '''Check if a point is a duplicate of an already existing node
        '''
        for i in range(len(self.nodes)):
            if (self.nodes[i].point[:2] == point).all():
                return [True, i]

        return [False, None]
    
    def closest_node(self, point):
        #Returns the index of the closest node
        """Implement a method to get the closest node to a sapled point"""
        min_dist = np.inf
        closest_node_id = -1
        for i, node in enumerate(self.nodes):
            dist = np.linalg.norm(node.point[:2] - point)
            if dist < min_dist:
                min_dist = dist
                closest_node_id = i
        return closest_node_id


    def normalize_angle(self, theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

    def simulate_trajectory(self, node_i, point_s):
        """Simulate the robot trajectory to the goal point"""
        if np.linalg.norm(point_s - node_i[:2]) < 0.2:
            return np.array([[], [], []])

        x, y, theta = node_i[0], node_i[1], node_i[2]  # pos., orient. of robot wrt inertial frame {I}

        # Initialize velocities
        vel, rot_vel = self.robot_controller(node_i, point_s)
        robot_traj = self.trajectory_rollout(vel, rot_vel, theta) + node_i 

        cur_node = robot_traj[:, -1].reshape(3, 1)
        dist_to_goal = np.linalg.norm(point_s - cur_node[:2])

        iter = 1

        while dist_to_goal > 0.2 and iter < 5:
            vel, rot_vel = self.robot_controller(cur_node, point_s)
            step_traj = self.trajectory_rollout(vel, rot_vel, cur_node[2]) + cur_node
            robot_traj = np.hstack((robot_traj, step_traj))
            cur_node = robot_traj[:, -1].reshape(3, 1)
            dist_to_goal = np.linalg.norm(point_s - cur_node[:2])
            iter += 1

        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        """PD controller to get the robot to move towards the goal point"""
        theta_d = np.arctan2((point_s[1] - node_i[1]), (point_s[0] - node_i[0]))
        theta = node_i[2]
        heading_error = np.around(self.normalize_angle(theta_d - theta), 3)
        rot_vel = np.round(self.kP * (heading_error) + self.kD * (heading_error - self.prev_heading_error) / (self.timestep), 2)
    
        if rot_vel > self.rot_vel_max:
            rot_vel = self.rot_vel_max
        if rot_vel < -self.rot_vel_max:
            rot_vel = -self.rot_vel_max

        vel = np.around(self.vel_max / (6 * abs(rot_vel) + 1), 2)
        self.prev_heading_error = heading_error

        return vel, rot_vel

    def trajectory_rollout(self, vel, rot_vel, theta_i):
        """Implement a way to rollout the controls chosen"""
        trajectory = np.array([[],[],[]])                          # initialize array
        t = np.linspace(0.1, self.timestep, self.num_substeps+1)
        # If the robot is not rotating, use the unicycle model with zero angle moves
        if rot_vel == 0:
            # Cant use exact equations, approximate with first order differential
            x_I = [np.around((vel*t*np.cos(theta_i)),2)]
            y_I = [np.around((vel*t*np.sin(theta_i)),2)]
            theta_I = [np.zeros(self.num_substeps+1)]
        else:
            # Integrate xdot = v*costheta and ydot = v*sintheta:
            # x = (v/w)*sin(wt + theta_i) - (v/w)*sin(theta_i)
            x_I = [np.around((vel/rot_vel)*(np.sin(rot_vel*t + theta_i)-np.sin(theta_i)), 2)]
            # y = -(v/w)*cos(wt + theta_i) + (v/w)*cos(theta_i)
            y_I = [np.around((vel/rot_vel)*(np.cos(theta_i)-np.cos(rot_vel*t + theta_i)), 2)]
            # theta = rot_vel*t
            theta_I = [np.around(rot_vel*t, 2)]

        trajectory = np.vstack((x_I, y_I, theta_I))
        # Return 3xN trajectory set of waypoints
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
        robot_footprint = np.empty((2, 0), dtype=int)

        for cell in cells.T:
            rr, cc = disk(cell, robot_radius_cells, shape=self.map_shape) # Pixel coordinates of robot footprint
            robot_footprint = np.hstack((robot_footprint, np.vstack((rr,cc))))
        return robot_footprint

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        """Implement a way to connect two already existing nodes."""
        return self.simulate_trajectory(node_i, point_f)
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        """Implement a cost to come metric"""
        # Give euclidean distance as cost to come per node in trajectory
        # Loop through waypoint costs to ensure zig-zagging is penalized
        waypoint_costs = np.zeros(trajectory_o.shape[1])
        for i in range(1, trajectory_o.shape[1]):
            waypoint_costs[i] = np.linalg.norm(trajectory_o[0:2, i] - trajectory_o[0:2, i-1])
        return np.sum(waypoint_costs)
    
    def update_children(self, node_id, cost_delta):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        """Update the costs of connected nodes after rewiring."""
        # Update the cost of the node children based on the new cost of the parent node and its delta cost from previous
        children_ids = self.nodes[node_id].children_ids
        for id in range(len(children_ids)):
            self.nodes[children_ids[id]].cost -= cost_delta
            self.update_children(children_ids[id], cost_delta)
        return
    
    def rewire(self):
        """Rewire the tree to find a better path."""
        # Get the ball radius for rewiring
        ball_radius = 0.4
        node_id = len(self.nodes) - 1
        cur_parent_id = self.nodes[node_id].parent_id
        updated = False
        min_node_id    = None
        min_node_cost   = self.nodes[node_id].cost

        # Use ball radius around last node to find near nodes to simulate trajectory to
        for i in range(len(self.nodes) - 1):
            if np.linalg.norm(self.nodes[i].point[0:2] - self.nodes[node_id].point[0:2]) < ball_radius:
                # Get the cost to come for the new trajectory
                trajectory_o = self.connect_node_to_point(self.nodes[i].point, self.nodes[node_id].point[0:2])
                collision  = self.check_collision(trajectory_o)
                cost_to_come = self.cost_to_come(trajectory_o) + self.nodes[i].cost
                # Check if the new cost to come is less than the current cost to come
                if cost_to_come < self.nodes[node_id].cost and not collision:
                    # Store the parent and cost of the node if no collision
                    min_node_id = i
                    min_node_cost = cost_to_come
                    updated = True
        # Update the parent and cost of the node with minimum if found
        if min_node_id is not None and not collision:
            # Update plot
            self.window.remove_line(self.nodes[node_id].point[:2].flatten(), self.nodes[cur_parent_id].point[:2].flatten())
            # Remove final node from parent children
            self.nodes[cur_parent_id].children_ids.remove(node_id)
            # Update parent and cost of the node
            self.nodes[node_id].parent_id = min_node_id
            self.nodes[node_id].cost = min_node_cost
            self.nodes[min_node_id].children_ids.append(node_id)
            # Add new line to plot after checking for collisions
            traj = self.connect_node_to_point(self.nodes[min_node_id].point, self.nodes[node_id].point[0:2])
            collision = self.check_collision(traj)
            if not collision:
                self.window.add_line(self.nodes[node_id].point[:2].flatten(), self.nodes[min_node_id].point[:2].flatten())

        # After rewiring the other nodes in the ball need to be checked again to see if new path edge helps
        if updated:
            for i in range(len(self.nodes) - 1):
                if np.linalg.norm(self.nodes[i].point[0:2] - self.nodes[node_id].point[0:2]) < ball_radius and i != node_id:
                    # Get the cost to come
                    trajectory_o = self.connect_node_to_point(self.nodes[node_id].point, self.nodes[i].point[0:2])
                    collision  = self.check_collision(trajectory_o)
                    cost_to_come = self.cost_to_come(trajectory_o) + self.nodes[node_id].cost
                    # Check if the new cost to come is less than the current cost to come
                    if cost_to_come < self.nodes[i].cost and not collision:
                        # Update the parent and cost of the node
                        cost_delta = self.nodes[i].cost - cost_to_come
                        cur_parent_id = self.nodes[i].parent_id
                        self.window.remove_line(self.nodes[i].point[:2].flatten(), self.nodes[cur_parent_id].point[:2].flatten())
                        self.nodes[cur_parent_id].children_ids.remove(i)
                        self.nodes[i].parent_id = node_id
                        self.nodes[i].cost = cost_to_come
                        self.nodes[node_id].children_ids.append(i)
                        # Add new line to plot after checking for collisions
                        traj = self.connect_node_to_point(self.nodes[node_id].point, self.nodes[i].point[0:2])
                        collision = self.check_collision(traj)
                        if not collision:
                            self.window.add_line(self.nodes[node_id].point[:2].flatten(), self.nodes[i].point[:2].flatten())
                        node_id = i
                        # Update the children of the node
                        self.update_children(node_id, cost_delta)

    def rrt_planning(self):
        # This function performs RRT on the given map and robot
        # You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        iter = 0
        while not self.goal_reached:  # Most likely need more iterations than this to complete the map!
            # for i in range(50):
            # Sample map space
            point = self.sample_map_space()
            print("RRT Iteration: ", iter)
            iter+=1
            # point = np.array([[10],[-5]])

            # *** Plotting for sampling ***#
            # print('Sampled point: ', point)
            # self.window.add_point(point.flatten())

            # Get the closest point
            closest_node_id = self.closest_node(point)
            # self.window.add_line(point.flatten(), [4, 4])
            # print('Closest Node: ', self.nodes[closest_node_id].point)

            # Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            # Check for collisions
            self.add_nodes_to_graph(closest_node_id, trajectory_o)

            # raw_input("Press Enter for loop to continue!")
        return self.nodes


    def add_nodes_to_graph(self, closest_node_id, robot_traj):
        """Add nodes to the graph after checking for collisions."""
        points_shape = np.shape(robot_traj)
        points_total = points_shape[1]

        collision = self.check_collision(robot_traj)

        # list to keep track of duplicate elements
        dupl_list = {}
        self.nodes_added = 0

        # no collision in the trajectory
        if not collision:
            for i in range(points_total):
                # check for duplicate node
                duplicate = self.check_if_duplicate(robot_traj[:2, i].reshape(2, 1))

                if duplicate[0]:
                    dupl_list[i] = duplicate[1]
                else:
                    # closest_node is the parent
                    if i == 0:
                        cmp = np.hstack((self.nodes[closest_node_id].point, robot_traj[:, i].reshape((3, 1))))
                        cost_to_come = self.cost_to_come(cmp) + self.nodes[closest_node_id].cost
                        self.nodes.append(Node(robot_traj[:, i].reshape((3, 1)), closest_node_id, cost_to_come))
                        self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)
                        # Add line after checking for collision
                        traj = self.connect_node_to_point(self.nodes[closest_node_id].point, robot_traj[:2, i])
                        collision = self.check_collision(traj)
                        if not collision:
                            self.window.add_line(self.nodes[closest_node_id].point[:2].flatten(), robot_traj[:2, i].flatten())
                        self.nodes_added += 1

                    else:
                        if (i - 1) in dupl_list.keys():
                            prev_node_idx = dupl_list[i - 1]
                        else:
                            prev_node_idx = -1
                        cmp = np.hstack((self.nodes[prev_node_idx].point, robot_traj[:, i].reshape((3, 1))))
                        cost_to_come = self.cost_to_come(cmp) + self.nodes[prev_node_idx].cost
                        self.nodes.append(Node(robot_traj[:, i].reshape((3, 1)), len(self.nodes) - 1, cost_to_come))
                        self.nodes[-2].children_ids.append(len(self.nodes) - 1)
                        # Add line after checking for collision
                        traj = self.connect_node_to_point(self.nodes[closest_node_id].point, robot_traj[:2, i])
                        collision = self.check_collision(traj)
                        if not collision:
                            self.window.add_line(self.nodes[-2].point[:2].flatten(), robot_traj[:2, i].flatten())
                        self.nodes_added += 1

                    dist_from_goal = np.linalg.norm(self.nodes[-1].point[:2] - self.goal_point)
                    if dist_from_goal <= self.stopping_dist:
                        self.goal_reached = True

    def check_collision(self, robot_traj):
        '''
        about: checks collision for a point in the occupancy map

        input: robot_traj (3xN array) - series of robot poses in map frame {I}
        output: bool (True if collision occurs)
        '''
        # point = np.array([[6.5], [-17]])
        # self.window.add_point(point.flatten())
        footprint = self.points_to_robot_circle(robot_traj[:2, :])

        points_shape = np.shape(footprint)
        points_total = points_shape[1]

        for i in range(points_total):
            if self.map[int(footprint[1, i]), int(footprint[0, i])] == 0:
                return True

            if "obstacle" in self.map_settings_dict:
                for box in self.map_settings_dict["obstacles"].keys():
                    val = self.map_settings_dict["obstacles"][box]
                    x1 = int(val[0])
                    y1= int(val[1])
                    x2 = x1 + int(val[2])
                    y2 = y1 + int(val[3])
                    if y1 <= int(footprint[1, i]) <=y2 and x1 <= int(footprint[0, i]) <= x2:

                        return True

        return False

    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot
        goal_reached = False
        max_iter = 100000
        iter = 0
        while not goal_reached and iter <= max_iter:
            print("RRT Iteration: ", iter)
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)
            print("Closest Node: ", self.nodes[closest_node_id].point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            #Check for Collision
            # Get the robot footprint for the trajectory points
            robot_footprint = self.points_to_robot_circle(trajectory_o[0:2, :])
            # Check if any of the robot footprint points are in an occupied cell
            self.add_nodes_to_graph(closest_node_id, trajectory_o)

            #Last node rewire and close node rewire
            self.rewire()

            #Check for early end
            if np.linalg.norm(self.nodes[-1].point[0:2] - self.goal_point) < self.stopping_dist:
                print("Goal Reached.")
                goal_reached = True
            iter += 1
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
    # Set map information
    #map_filename = "simple_map.png"
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"
    #map_filename = "myhal.png"
    #map_setings_filename = "myhal.yaml"

    # robot information
    goal_point = np.array([[42], [-44.5]])  # m
    stopping_dist = 0.5  # m

    start_time = time.time()
    # RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    #nodes = path_planner.rrt_planning()
    nodes = path_planner.rrt_star_planning()

    node_path_metric = np.hstack(path_planner.recover_path())
    np.save("RRTstar_path_willowgarageworld.npy", node_path_metric)

    plot_name = 'RRTstar_willowgarageworld.png'
    pygame.image.save(path_planner.window.screen, plot_name)
    print('Time taken to find the path: ', (time.time() - start_time) / 60, ' min')

    #Visualize the path using matplotlib
    plt.imshow(path_planner.occupancy_map, cmap='gray')
    plt.plot(node_path_metric[0, :], node_path_metric[1, :], 'r')
    plt.show()

if __name__ == '__main__':
    main()