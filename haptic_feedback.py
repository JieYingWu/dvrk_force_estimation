# Author: Nam Tran
# Date: 2019-08-13

# To run this script with feedback from neural network: 
# > python dvrk_mtm_haptic_feedback.py neural_network <path_to_pretrained_model>
# To run this script with feedback from force sensor: 
# > python dvrk_mtm_haptic_feedback.py force_sensor 

from __future__ import print_function
import rospy
import sys
import rospy
import numpy as np
import collections
import itertools
import time
import neural_network
import threading
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Wrench, WrenchStamped
from np_ringbuffer import RingBuffer

JOINTS = 6
epoch_to_use = 10

class haptic_feedback_application:

    # configuration
    def configure(self):
        self.datapoints_per_row = 100
        self.single_datapoint_length = 12
        self.label_length = 3
        self.input_size = self.datapoints_per_row * self.single_datapoint_length
        self.labels = RingBuffer(capacity = self.label_length, dtype = float)
        self.data = RingBuffer(capacity = self.input_size, dtype = float)
        if sys.argv[1] == "neural_network":
            networks = []
            for i in range(JOINTS):
                model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
                networks.append(torqueNetwork())
                networks[j].load_state_dict(state['model'])
        self.following = False
        rospy.Subscriber('/dvrk/MTMR_PSM1/following', Bool, self.following_event_cb)
        rospy.Subscriber('/dvrk/PSM1/state_joint_current', JointState, self.get_joint_state)
        rospy.Subscriber('/dvrk/PSM1/jacobian_spatial', Float64MultiArray, self.__jacobian_spatial_cb)
        rospy.Subscriber('/atinetft/wrench', WrenchStamped, self.get_wrench)
        self.wrench_orientation_pub = rospy.Publisher('/dvrk/MTMR/set_wrench_body_orientation_absolute', Bool, queue_size = 1)
        self.wrench_pub = rospy.Publisher('/dvrk/MTMR/set_wrench_body', Wrench, queue_size = 1)

    def get_wrench(self, msg):
        self.labels.extend([msg.wrench.force.x]+[msg.wrench.force.y]+[msg.wrench.force.z])
                    
    def get_joint_state(self, msg):
        self.data.extend(msg.velocity + msg.effort)

    def __jacobian_spatial_cb(self, data):
        """Callback for the Jacobian in spatial frame.
        :param data: Jacobian."""
        jacobian = np.asarray(data.data)
        jacobian.shape = data.layout.dim[0].size, data.layout.dim[1].size
        self.__jacobian_spatial = jacobian
        
    def following_event_cb(self, msg):
        self.following = msg.data
        if self.following:
            ori = Bool()
            ori.data = True
            self.wrench_orientation_pub.publish(ori)

    def enable_neural_network_haptic_feedback(self):
        rospy.init_node('haptic_feedback', anonymous = True, log_level = rospy.WARN)
        rate = rospy.Rate(1000)        

        while not rospy.is_shutdown():
            data_array = np.array(self.data)
            
            if self.following and len(data_array) == self.single_datapoint_length*self.datapoints_per_row: 
                data_array = data_array.reshape(1,-1)
                predicted_force = np.zeros(6,1)
                for j in range(JOINTS):
                    predicted_force[j] = networks(data_array).item()
                predicted_force = np.linalg.inv(__jacobian_spatial).dot(prediction)
                w = Wrench()
                w.force.x = predicted_force[0]
                w.force.y = predicted_force[1]
                w.force.z = predicted_force[2]
                w.torque.x = predicted_force[3]
                w.torque.y = predicted_force[4]
                w.torque.z = predicted_force[5]
                self.wrench_pub.publish(w)
                print(predicted_force)

    def enable_force_sensor_haptic_feedback(self):
        rospy.init_node('haptic_feedback', anonymous = True, log_level = rospy.WARN)
        rate = rospy.Rate(1000)        

        while not rospy.is_shutdown():
            label_array = np.array(self.labels)
            
            if self.following and len(label_array) == self.label_length: 
                w = Wrench()
                w.force.x = label_array[0]
                w.force.y = label_array[1]
                w.force.z = label_array[2]
                w.torque.x = 0.0
                w.torque.y = 0.0
                w.torque.z = 0.0
                self.wrench_pub.publish(w)
                print((label_array[0],label_array[1],label_array[2]))

if sys.argv[1] == "neural_network":
    pretrained_model = sys.argv[2]

def main():
    application = haptic_feedback_application()
    application.configure()
    if sys.argv[1] == "neural_network":
        application.enable_neural_network_haptic_feedback()
    elif sys.argv[1] == "force_sensor":
        application.enable_force_sensor_haptic_feedback()

if __name__ == '__main__':
    main()
