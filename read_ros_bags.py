import os
from pathlib import Path
import rosbag
import numpy as np 
import time
import argparse
from scipy import interpolate

class RosbagParser():

    def __init__(self, args):
        
        self.args = args
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        try:
            joint_path = Path(self.output) / "joints"
            joint_path.mkdir(mode=0o777, parents=False)
            jacobian_path = Path(self.output) / "jacobian"
            jacobian_path.mkdir(mode=0o777, parents=False)
            sensor_path = Path(self.output) / "sensor"
            sensor_path.mkdir(mode=0o777, parents=False)
        except OSError:
            print("Data path exists")

    def interp(self, time, mat):
        new_mat = np.zeros((len(time), mat.shape[1]))
        new_mat[:,0] = time
        for i in range(mat.shape[1]):
            f = interpolate.interp1d(mat[:,0], mat[:,i])
            new_mat[:,i] = f(time)
        return new_mat
        
            
    def single_datapoint_processing(self, file_name): # parses data into the global dictionaries
   
        bag = rosbag.Bag(self.folder+file_name, 'r')

        force_sensor = []
        force_sensor_timestamps = []
        joint_position = []
        joint_velocity = []
        joint_effort = []
        joint_timestamps = []
        jacobian = []
        jacobian_timestamps = []

        length_force_sensor = 0
        length_state_joint_current = 0
        length_jacobian = 0

        print("Processing " + file_name)
        state_joint_current = bag.read_messages(topics=['/dvrk/PSM1/state_joint_current'])
        for topic, msg, t in state_joint_current:
            joint_timestamps.append(t.secs+t.nsecs*10**-9)

            # handles velocity for six joints
            joint_velocity.append(list(msg.velocity))

            # handles position for six joints
            joint_position.append(list(msg.position))

            # handles effort for six joints
            joint_effort.append(list(msg.effort))
            length_state_joint_current+=1

        jacobian_spatial = bag.read_messages(topics=['/dvrk/PSM1/jacobian_spatial'])
        for topic, msg, t in jacobian_spatial:
            jacobian_timestamps.append(t.secs+t.nsecs*10**-9)
            # jacobian
            jacobian.append(list(msg.data))
            length_jacobian+=1
        
            
        wrench = bag.read_messages(topics=['/atinetft/wrench'])
        for topic, msg, t in wrench:
            timestamps = t.secs+t.nsecs*10**-9
            force_sensor_timestamps.append(timestamps)
            x = msg.wrench.force.x
            y = msg.wrench.force.y
            z = msg.wrench.force.z # the sensor is probably most accurate in the z direction
            force_sensor.append(list((x,y,z)))
            length_force_sensor+=1

        bag.close()
                                      
        print("Processed wrench: counts: {}".format(length_force_sensor))
        print("Processed state joint current: count: {}".format(length_state_joint_current))
        print("Processed Jacobian: count: {}".format(length_jacobian))
        
        joints = np.column_stack((joint_timestamps, joint_position, joint_velocity, joint_effort))
        if length_force_sensor: 
            force_sensor = np.column_stack((force_sensor_timestamps,force_sensor))
            jacobian = np.column_stack((jacobian_timestamps, jacobian))
        else:
            force_sensor = None
            jacobian = np.column_stack((jacobian_timestamps, jacobian))

        if self.interpolate:
            joints = joints[joints[:,0] > jacobian[0,0],:]
            joints = joints[joints[:,0] < jacobian[-1,0],:]
            print('Interpolating Jacobian')
            jacobian = self.interp(joints[:,0], jacobian)
            
        if self.interpolate and length_force_sensor > 0:            
            force_sensor = force_sensor[force_sensor[:,0] > joints[0,0],:]
            force_sensor = force_sensor[force_sensor[:,0] < joints[-1,0],:]
            print('Interpolating Joints')
            joints = self.interp(force_sensor[:,0], joints)
            jacobian = self.interp(force_sensor[:,0], jacobian)
            
        return joints, force_sensor, jacobian

    def write(self, joints, force_sensor, jacobian):
        file_name = self.prefix + str(self.index)        
        np.savetxt(self.output + "joints/" + file_name + ".csv", joints, delimiter=',')
        np.savetxt(self.output + "jacobian/" + file_name + ".csv", jacobian, delimiter=',')
        if force_sensor is not None:
            np.savetxt(self.output + "sensor/" + file_name + ".csv", force_sensor, delimiter=',')
        print("Wrote out " + file_name)
        print("")
        
    def parse_bags(self):

        print("\nParsing\n")
        files = os.listdir(self.folder)
        files.sort()
        for file_name in files:
            if file_name.endswith('.bag'):
                joints, force_sensor, jacobian = self.single_datapoint_processing(file_name)
                self.write(joints, force_sensor, jacobian)
                self.index += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='../data/', type=str, help='Path to Rosbag folder')
    parser.add_argument('-o', '--output', default='./parsed_data/', type=str, help='Path to write out parsed csv')
    parser.add_argument('--prefix', default='bag_', type=str, help='Prefix for the output csv names')
    parser.add_argument('--index', default=0, type=int, help='Starting index for the output csv names')
    parser.add_argument('--interpolate', default=False, type=bool, help='Interpolate joint to match force sensor')
    args = parser.parse_args()
    start = time.time()
    rosbag_parser = RosbagParser(args)
    rosbag_parser.parse_bags()
    print("Parsing complete") 
    end = time.time()
    print("The entire process takes {} seconds".format(end - start))

if __name__ == "__main__":
    main()

