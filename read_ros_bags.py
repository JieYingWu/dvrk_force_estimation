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
        jaw = []
        jaw_position = []
        jaw_velocity = []
        jaw_effort = []
        jaw_timestamps = []

        length_force_sensor = 0
        length_joint = 0
        length_jacobian = 0
        length_jaw = 0

        print("Processing " + file_name)
        joint = bag.read_messages(topics=['/dvrk/PSM2/state_joint_current'])
        for topic, msg, t in joint:
            joint_timestamps.append(t.secs+t.nsecs*10**-9)

            # handles velocity for six joints
            joint_velocity.append(list(msg.velocity))

            # handles position for six joints
            joint_position.append(list(msg.position))

            # handles effort for six joints
            joint_effort.append(list(msg.effort))
            length_joint+=1

        jacobian_spatial = bag.read_messages(topics=['/dvrk/PSM2/jacobian_spatial'])
        for topic, msg, t in jacobian_spatial:
            jacobian_timestamps.append(t.secs+t.nsecs*10**-9)
            jacobian.append(list(msg.data))
            length_jacobian+=1

        jaw = bag.read_messages(topics=['/dvrk/PSM2/state_jaw_current'])
        for topic, msg, t in jaw:
            jaw_timestamps.append(t.secs+t.nsecs*10**-9)
            jaw_velocity.append(list(msg.velocity))
            jaw_position.append(list(msg.position))
            jaw_effort.append(list(msg.effort))
            length_jaw+=1
            
        wrench = bag.read_messages(topics=['/atinetft/wrench'])
        for topic, msg, t in wrench:
            timestamps = t.secs+t.nsecs*10**-9
            force_sensor_timestamps.append(timestamps)
            x = msg.wrench.force.x
            y = msg.wrench.force.y
            z = msg.wrench.force.z # the sensor is probably most accurate in the z direction
            force_sensor.append([x,y,z])
            length_force_sensor+=1

        bag.close()
                                      
        print("Processed wrench: counts: {}".format(length_force_sensor))
        print("Processed state joint current: count: {}".format(length_joint))
        print("Processed state jaw current: count: {}".format(length_jaw))
        print("Processed Jacobian: count: {}".format(length_jacobian))

        try:
            joint_path = Path(self.output) / "joints"
            joint_path.mkdir(mode=0o777, parents=False)
        except OSError:
            print("Joints path exists")
            
        try:
            jacobian_path = Path(self.output) / "jacobian"
            jacobian_path.mkdir(mode=0o777, parents=False)
        except OSError:
            print("Jacobian path exists")
            
        if length_force_sensor > 0:
            try:
                sensor_path = Path(self.output) / "sensor"
                sensor_path.mkdir(mode=0o777, parents=False)
            except OSError:
                print("Sensor path exists")
                        
        if length_jaw > 0:
            try:
                jaw_path = Path(self.output) / "sensor"
                jaw_path.mkdir(mode=0o777, parents=False)
            except OSError:
                print("Jaw path exists")
        
        start_time = joint_timestamps[0]
        joint_timestamps = np.array(joint_timestamps) - start_time
        jacobian_timestamps = np.array(jacobian_timestamps) - start_time
        joints = np.column_stack((joint_timestamps, joint_position, joint_velocity, joint_effort))
        jacobian = np.column_stack((jacobian_timestamps, jacobian))
        if length_force_sensor:
            force_sensor_timestamps = np.array(force_sensor_timestamps) - start_time
            force_sensor = np.column_stack((force_sensor_timestamps,force_sensor))
        else:
            force_sensor = None
            
        if length_jaw:
            jaw = np.column_stack((joint_timestamps, joint_position, joint_velocity, joint_effort))
            jaw_timestamps = np.array(jaw_timestamps) - start_time
            jaw = np.column_stack((jaw_timestamps, jaw))
        else:
            jaw = None
            
        return joints, force_sensor, jacobian, jaw

    def write(self, joints, force_sensor, jacobian, jaw):
        file_name = self.prefix + str(self.index)        
        np.savetxt(self.output + "joints/" + file_name + ".csv", joints, delimiter=',')
        np.savetxt(self.output + "jacobian/" + file_name + ".csv", jacobian, delimiter=',')
        if force_sensor is not None:
            np.savetxt(self.output + "sensor/" + file_name + ".csv", force_sensor, delimiter=',')
        if jaw is not None:
            np.savetxt(self.output + "jaw/" + file_name + ".csv", jaw, delimiter=',')
        print("Wrote out " + file_name)
        print("")
        
    def parse_bags(self):

        print("\nParsing\n")
        files = os.listdir(self.folder)
        files.sort()
        for file_name in files:
            if file_name.endswith('.bag'):
                joints, force_sensor, jacobian, jaw = self.single_datapoint_processing(file_name)
                self.write(joints, force_sensor, jacobian, jaw)
                self.index += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='../data/', type=str, help='Path to Rosbag folder')
    parser.add_argument('-o', '--output', default='./parsed_data/', type=str, help='Path to write out parsed csv')
    parser.add_argument('--prefix', default='bag_', type=str, help='Prefix for the output csv names')
    parser.add_argument('--index', default=0, type=int, help='Starting index for the output csv names')
    args = parser.parse_args()
    start = time.time()
    rosbag_parser = RosbagParser(args)
    rosbag_parser.parse_bags()
    print("Parsing complete") 
    end = time.time()
    print("The entire process takes {} seconds".format(end - start))

if __name__ == "__main__":
    main()

