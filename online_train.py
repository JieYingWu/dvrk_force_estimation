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

JOINTS = 6
epoch_to_use = 10

class OnlineTrain():

    def __init__(self, joints=6, pretrained_model=None):
        self.joints = joints
        self.device =  torch.device("cuda:0" if torch.cuda.is_available(), else "cpu")

        networks = []
        optimizers = []
        scheduler = []
        for j in range(joints):
            networks.append(torqueNetwork())
            networks[j].to(device)
            optimizers.append(torch.optim.SGD(networks[j].parameters(), lr))
            schedulers.append(ReduceLROnPlateau(optimizers[j]))

        loss_fn = torch.nn.MSELoss()
        try:
            model_root = root / "models"
            model_root.mkdir(mode=0o777, parents=False)
        except OSError:
            print("Model path exists")
            
        if pretrained_model:
            
        
    # configuration
    def configure(self):
        self.input_size = 12
        self.label_size = 6
        networks = []
        for i in range(JOINTS):
            model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
            networks.append(torqueNetwork())
            networks[j].load_state_dict(state['model'])
        rospy.Subscriber('/dvrk/PSM1/state_joint_current', JointState, self.get_joint_state)

    def get_joint_state(self, msg):
        self.data.extend(msg.position + msg.velocity + msg.effort)

    def enable_neural_network_haptic_feedback(self):
        rospy.init_node('haptic_feedback', anonymous = True, log_level = rospy.WARN)
        rate = rospy.Rate(1000)        

        while not rospy.is_shutdown():
            data_array = np.array(self.data)
         
            data_array = data_array.reshape(1,-1)
            predicted_force = np.zeros(6,1)
            for j in range(JOINTS):
                predicted_force[j] = networks(data_array).item()

                
if __name__ == '__main__':
    main()
