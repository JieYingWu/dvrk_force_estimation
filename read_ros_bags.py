# Author: Nam Tran
# Date: 2019-08-13

# To read raw ROS bags into processed text files (example: ROS bag X generates 2 files bag_X_joint_values and bag_X_phantom_force_sensor):
# > python read_ros_bags.py <directory_to_put_parsed_files> <num_bags> <starting_bag_index>

import sys
import math
import dvrk
import rospy
import rosbag
import numpy
import csv
import time
import threading

bag_phantom_force_sensor = {}
bag_phantom_force_sensor_timestamps = {}
bag_joint_position = {} 
bag_joint_velocity = {}
bag_joint_effort = {}
bag_joint_timestamps = {}
bag_actuator_current_measured = {} 
bag_actuator_current_measured_timestamps = {} 

def single_datapoint_processing(folder_name, bag_index): # parses data into the global dictionaries

    global bag_phantom_force_sensor
    global bag_phantom_force_sensor_timestamps 
    global bag_joint_position 
    global bag_joint_velocity
    global bag_joint_effort 
    global bag_joint_timestamps 

    bag = rosbag.Bag(folder_name + str(bag_index) + ".bag", 'r')

    bag_phantom_force_sensor[bag_index] = []
    bag_phantom_force_sensor_timestamps[bag_index] = []
    bag_joint_position[bag_index] = []
    bag_joint_velocity[bag_index] = []
    bag_joint_effort[bag_index] = []
    bag_joint_timestamps[bag_index] = []
    bag_actuator_current_measured_timestamps[bag_index] = [] 

    length_force_sensor = 0
    length_state_joint_current = 0
    length_jacobian_body = 0
    length_jacobian_spatial = 0
    length_actuator_current_measured = 0

    print("Processing {}".format(folder_name + str(bag_index) + ".bag"))
    state_joint_current = bag.read_messages(topics=['/dvrk/PSM1/state_joint_current'])
    for topic, msg, t in state_joint_current:
        bag_joint_timestamps[bag_index].append(t.secs+t.nsecs*10**-9)

        # handles velocity for six joints
        dummy = []
        dummy.extend(msg.velocity)
        bag_joint_velocity[bag_index].append(dummy)

        # handles position for six joints
        dummy = []
        dummy.extend(msg.position)
        bag_joint_position[bag_index].append(dummy) 

        # handles effort for six joints
        dummy = []
        dummy.extend(msg.effort)
        bag_joint_effort[bag_index].append(dummy)

        length_state_joint_current+=1

    if length_state_joint_current == 0:
        state_joint_current = bag.read_messages(topics=['dvrk/PSM1/state_joint_current'])
        for topic, msg, t in state_joint_current:
            bag_joint_timestamps[bag_index].append(t.secs+t.nsecs*10**-9)

            # handles velocity for six joints
            dummy = []
            dummy.extend(msg.velocity)
            bag_joint_velocity[bag_index].append(dummy)

            # handles position for six joints
            dummy = []
            dummy.extend(msg.position)
            bag_joint_position[bag_index].append(dummy) 

            # handles effort for six joints
            dummy = []
            dummy.extend(msg.effort)
            bag_joint_effort[bag_index].append(dummy)

            length_state_joint_current+=1

    wrench = bag.read_messages(topics=['/atinetft/wrench'])
    for topic, msg, t in wrench:
        timestamps = t.secs+t.nsecs*10**-9
        bag_phantom_force_sensor_timestamps[bag_index].append(timestamps)
        x = msg.wrench.force.x
        y = msg.wrench.force.y
        z = msg.wrench.force.z # the sensor is probably most accurate in the z direction
        bag_phantom_force_sensor[bag_index].append(list((x,y,z)))
        length_force_sensor+=1

    bag.close()

    print("Processed wrench: counts: {}".format(length_force_sensor))
    print("Processed state joint current: count: {}".format(length_state_joint_current))
    print("")

def thread_write(folder_name, bag_index, list_of_instances, mode):

    if mode == "joint_values":
        with open("../data/csv/bag_"+str(bag_index)+"_joint_values.csv", mode = "w") as outfile:
            outfile_handle = csv.writer(outfile)
            outfile_handle.writerows(list_of_instances)
            print("Wrote to file {}".format("../data/csv/bag_"+str(bag_index)+"_joint_values"))

    elif mode == "phantom_force_sensor":
        with open("../data/csv/bag_"+str(bag_index)+"_phantom_force_sensor.csv", mode = "w") as outfile:
            outfile_handle = csv.writer(outfile)
            outfile_handle.writerows(list_of_instances)
            print("Wrote to file {}".format("../data/csv/bag_"+str(bag_index)+"_phantom_force_sensor"))


def output_prep(folder_name, bag_index): # create rows that can be written to files

    global bag_phantom_force_sensor
    global bag_phantom_force_sensor_timestamps 
    global bag_joint_position 
    global bag_joint_velocity
    global bag_joint_effort 
    global bag_joint_timestamps 

    write_threads = []
    joint_values = []
    for index, timestamp in enumerate(bag_joint_timestamps[bag_index]):
        dummy = []
        dummy.append(timestamp)
        dummy.extend(bag_joint_position[bag_index][index])
        dummy.extend(bag_joint_velocity[bag_index][index])
        dummy.extend(bag_joint_effort[bag_index][index])
        joint_values.append(dummy)

    thread = threading.Thread(target = thread_write, args = (folder_name, bag_index, joint_values, "joint_values"))
    thread.start()
    write_threads.append(thread)

    phantom_force_sensor = []
    for index, timestamp in enumerate(bag_phantom_force_sensor_timestamps[bag_index]):
        dummy = []
        dummy.append(timestamp)
        dummy.extend(bag_phantom_force_sensor[bag_index][index])
        phantom_force_sensor.append(dummy)

    thread = threading.Thread(target = thread_write, args = (folder_name, bag_index, phantom_force_sensor, "phantom_force_sensor"))
    thread.start()
    write_threads.append(thread)

    return write_threads

def parser():

    global bag_phantom_force_sensor
    global bag_phantom_force_sensor_timestamps 
    global bag_joint_position 
    global bag_joint_velocity
    global bag_joint_effort 
    global bag_joint_timestamps 

    folder_name = sys.argv[1]
    num_bags = sys.argv[2]
    print("\nParsing\n")
    starting_index = sys.argv[3]
    for bag_index in range(int(num_bags)):
        single_datapoint_processing(folder_name, int(starting_index)+bag_index)
        write_threads = output_prep(folder_name, int(starting_index)+bag_index)
        for thread in write_threads:
            thread.join()

        bag_phantom_force_sensor.clear()
        bag_phantom_force_sensor_timestamps.clear() 
        bag_joint_position.clear() 
        bag_joint_velocity.clear()
        bag_joint_effort.clear() 
        bag_joint_timestamps.clear() 

def main():

    start = time.time()
    parser()
    print("Parsing complete") 
    end = time.time()
    print("The entire process takes {} seconds".format(end - start))

if __name__ == "__main__":
    main()

