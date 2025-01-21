#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def publisher_node():
    """TODO: initialize the publisher node here, \
            and publish wheel command to the cmd_vel topic')"""
    cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(10)
    twist = Twist()
    while not rospy.is_shutdown():
        twist.linear.x = 0.25
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time < rospy.Duration(4.0)):
            # Move straight for 4s
            rospy.loginfo(twist)
            cmd_pub.publish(twist)
            rate.sleep()

        twist.linear.x = 0
        twist.angular.z = 0.25
        while (rospy.Time.now() - start_time < rospy.Duration(25.1)):
            rospy.loginfo(twist)
            cmd_pub.publish(twist)
            rate.sleep()

def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
