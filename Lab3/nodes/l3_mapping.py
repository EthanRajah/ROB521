#!/usr/bin/env python3
from __future__ import division, print_function

import numpy as np
import rospy
import tf2_ros
from skimage.draw import line as ray_trace
import rospkg
import matplotlib.pyplot as plt

# msgs
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan

from utils import convert_pose_to_tf, convert_tf_to_pose, euler_from_ros_quat, \
     tf_to_tf_mat, tf_mat_to_tf


ALPHA = 1
BETA = 1
MAP_DIM = (7, 7)
CELL_SIZE = .01
NUM_PTS_OBSTACLE = 3
SCAN_DOWNSAMPLE = 10

class OccupancyGripMap:
    def __init__(self):
        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # subscribers and publishers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb, queue_size=1)
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

        # attributes
        width = int(MAP_DIM[0] / CELL_SIZE); height = int(MAP_DIM[1] / CELL_SIZE)
        self.log_odds = np.zeros((width, height))
        self.np_map = np.ones((width, height), dtype=np.uint8) * -1  # -1 for unknown
        self.map_msg = OccupancyGrid()
        self.map_msg.info = MapMetaData()
        self.map_msg.info.resolution = CELL_SIZE
        self.map_msg.info.width = width
        self.map_msg.info.height = height

        # transforms
        self.base_link_scan_tf = self.tf_buffer.lookup_transform('base_link', 'base_scan', rospy.Time(0),
                                                            rospy.Duration(2.0))
        odom_tf = self.tf_buffer.lookup_transform('odom', 'base_link', rospy.Time(0), rospy.Duration(2.0)).transform

        # set origin to center of map
        rob_to_mid_origin_tf_mat = np.eye(4)
        rob_to_mid_origin_tf_mat[0, 3] = -width / 2 * CELL_SIZE
        rob_to_mid_origin_tf_mat[1, 3] = -height / 2 * CELL_SIZE
        odom_tf_mat = tf_to_tf_mat(odom_tf)
        self.map_msg.info.origin = convert_tf_to_pose(tf_mat_to_tf(odom_tf_mat.dot(rob_to_mid_origin_tf_mat)))

        # map to odom broadcaster
        self.map_odom_timer = rospy.Timer(rospy.Duration(0.1), self.broadcast_map_odom)
        self.map_odom_tf = TransformStamped()
        self.map_odom_tf.header.frame_id = 'map'
        self.map_odom_tf.child_frame_id = 'odom'
        self.map_odom_tf.transform.rotation.w = 1.0

        rospy.spin()
        plt.imshow(100-self.np_map, cmap='gray', vmin=0, vmax=100)
        rospack = rospkg.RosPack()
        path = rospack.get_path("rob521_lab3")
        plt.savefig(path+"/map.png")

    def broadcast_map_odom(self, e):
        self.map_odom_tf.header.stamp = rospy.Time.now()
        self.tf_br.sendTransform(self.map_odom_tf)

    def scan_cb(self, scan_msg):
        # read new laser data and populate map
        # get current odometry robot pose
        try:
            odom_tf = self.tf_buffer.lookup_transform('odom', 'base_scan', rospy.Time(0)).transform
        except tf2_ros.TransformException:
            rospy.logwarn('Pose from odom lookup failed. Using origin as odom.')
            odom_tf = convert_pose_to_tf(self.map_msg.info.origin)

        # get odom in frame of map
        odom_map_tf = tf_mat_to_tf(
            np.linalg.inv(tf_to_tf_mat(convert_pose_to_tf(self.map_msg.info.origin))).dot(tf_to_tf_mat(odom_tf))
        )
        odom_map = np.zeros(3)
        odom_map[0] = odom_map_tf.translation.x
        odom_map[1] = odom_map_tf.translation.y
        odom_map[2] = euler_from_ros_quat(odom_map_tf.rotation)[2]

        # YOUR CODE HERE!!! Loop through each measurement in scan_msg to get the correct angle and
        # x_start and y_start to send to your ray_trace_update function.
        for i, range_mes in enumerate(scan_msg.ranges[::SCAN_DOWNSAMPLE]):
            # If measurement is outside of meter range of lidar, skip
            if (scan_msg.range_min < range_mes < scan_msg.range_max):
                # Use SCAN_DOWNSAMPLE to skip some of the measurement rays for efficiency
                angle = odom_map[2] + SCAN_DOWNSAMPLE * i * scan_msg.angle_increment
                self.np_map, self.log_odds = self.ray_trace_update(self.np_map, self.log_odds, odom_map[0], odom_map[1], angle, range_mes)

        # publish the message
        self.map_msg.info.map_load_time = rospy.Time.now()
        self.map_msg.data = self.np_map.flatten()
        self.map_pub.publish(self.map_msg)

    def ray_trace_update(self, map, log_odds, x_start, y_start, angle, range_mes):
        """
        A ray tracing grid update as described in the lab document.

        :param map: The numpy map.
        :param log_odds: The map of log odds values.
        :param x_start: The x starting point in the map coordinate frame (i.e. the x 'pixel' that the robot is in).
        :param y_start: The y starting point in the map coordinate frame (i.e. the y 'pixel' that the robot is in).
        :param angle: The ray angle relative to the x axis of the map.
        :param range_mes: The range of the measurement along the ray.
        :return: The numpy map and the log odds updated along a single ray.
        """
        # YOUR CODE HERE!!! You should modify the log_odds object and the numpy map based on the outputs from
        # ray_trace and the equations from class. Your numpy map must be an array of int8s with 0 to 100 representing
        # probability of occupancy, and -1 representing unknown.

        # Get the x and y end points of the ray
        x_end = (x_start + range_mes * np.cos(angle))
        y_end = (y_start + range_mes * np.sin(angle))

        # Consider range of pixels around the end point as part of the obstacle in map frame
        surr_end_range = NUM_PTS_OBSTACLE * CELL_SIZE + range_mes
        x_end_range = (x_start + surr_end_range * np.cos(angle))
        y_end_range = (y_start + surr_end_range * np.sin(angle))

        # Convert to pixel coordinates
        x_start = int(x_start / CELL_SIZE)
        y_start = int(y_start / CELL_SIZE)
        x_end = int(x_end / CELL_SIZE)
        y_end = int(y_end / CELL_SIZE)
        x_end_range = int(x_end_range / CELL_SIZE)
        y_end_range = int(y_end_range / CELL_SIZE)

        # Get the x and y indices of the ray
        x_idx, y_idx = ray_trace(y_start, x_start, y_end, x_end)
        x_idx_end, y_idx_end = ray_trace(y_end, x_end, y_end_range, x_end_range)

        # Handle wrap around from the ray trace by removing out of bounds indices
        mask = (x_idx >= 0) & (x_idx < map.shape[0]) & (y_idx >= 0) & (y_idx < map.shape[1])
        x_idx = x_idx[mask]
        y_idx = y_idx[mask]

        mask = (x_idx_end >= 0) & (x_idx_end < map.shape[0]) & (y_idx_end >= 0) & (y_idx_end < map.shape[1])
        x_idx_end = x_idx_end[mask]
        y_idx_end = y_idx_end[mask]

        # Update unoccupied cells with -BETA and occupied cells with ALPHA
        log_odds[x_idx, y_idx] -= BETA
        log_odds[x_idx_end, y_idx_end] += ALPHA

        # Update the map with probabilities from 0 to 100 only for the points that were ray traced
        all_map = (self.log_odds_to_probability(log_odds) * 100).astype(np.int8)
        map[x_idx, y_idx] = all_map[x_idx, y_idx]
        map[x_idx_end, y_idx_end] = all_map[x_idx_end, y_idx_end]

        return map, log_odds

    def log_odds_to_probability(self, values):
        # print(values)
        return np.exp(values) / (1 + np.exp(values))


if __name__ == '__main__':
    try:
        rospy.init_node('mapping')
        ogm = OccupancyGripMap()
    except rospy.ROSInterruptException:
        pass