# Author: Nam Tran
# Date: 2019-08-13

# To run the script: 
# > python data_prep.py <num_bags> <starting_index>

import operator
import sys
import pyfiglet
import time
import string
import os
import sys
import numpy as np
import csv
import io
import threading

def interpolate(target_timestamp_list, timestamp_list, data_list):
    
    components = {}
    functions = []
    interpolated_components = {}
    interpolated_data_list = []

    for index in range(len(data_list[0])):
        components[index] = []
        interpolated_components[index] = [] 

    for each_list in data_list:
        for index, element in enumerate(each_list):
            components[index].append(element)        

    for index in range(len(data_list[0])):
        interpolated_components[index].extend(np.interp(target_timestamp_list, timestamp_list, components[index]))

    for index, element in enumerate(target_timestamp_list):
        dummy = []
        for i, component in interpolated_components.items():
            dummy.append(component[index])
        interpolated_data_list.append(dummy)

    print("Length of interpolated_list is {}".format(len(interpolated_data_list)))

    return interpolated_data_list

def parsing(use_interpolation, bag_index):

    joint_values_raw = {}
    phantom_force_sensor_raw = {}

    joint_values_preinterp = {}
    joint_values_timestamps_preinterp = {}
    phantom_force_sensor_preinterp = {}
    phantom_force_sensor_timestamps_preinterp = {}

    list_of_instances = []
    list_of_labels = []    

    phantom_force_sensor_timestamps = []
    phantom_force_sensor = {}
    joint_values_timestamps = []
    joint_values = {}

    print("Parsing {}bag_{}_joint_values".format(directory, bag_index))
    print("Parsing {}bag_{}_phantom_force_sensor".format(directory, bag_index))
    joint_values_raw[bag_index] = np.loadtxt(directory + "bag_" + str(bag_index) + "_joint_values", delimiter = ",")
    phantom_force_sensor_raw[bag_index] = np.loadtxt(directory + "bag_" + str(bag_index) + "_phantom_force_sensor", delimiter = ",")
    joint_values[bag_index] = []
    phantom_force_sensor[bag_index] = []
    joint_values_preinterp[bag_index] = []
    joint_values_timestamps_preinterp[bag_index] = []
    phantom_force_sensor_preinterp[bag_index] = []
    phantom_force_sensor_timestamps_preinterp[bag_index] = []


    for bag_index, list_of_force_sensor_data in phantom_force_sensor_raw.items():
        for element in list_of_force_sensor_data:
            phantom_force_sensor_timestamps_preinterp[bag_index].append(element[0])
            phantom_force_sensor_preinterp[bag_index].append(element[1:])
    for bag_index, list_of_joint_values in joint_values_raw.items():
        for index, element in enumerate(list_of_joint_values):
            if element[0] > phantom_force_sensor_timestamps_preinterp[bag_index][0] and element[0] < phantom_force_sensor_timestamps_preinterp[bag_index][len(phantom_force_sensor_timestamps_preinterp[bag_index])-1]:
                joint_values_timestamps_preinterp[bag_index].append(element[0])
                dummy = []
                dummy.extend(element[7:]) #we only use 12 values: 6 joint velocities and 6 joint efforts values
                joint_values_preinterp[bag_index].append(dummy)

    #we now know that joint values fall within the first and last force sensor timestamps
    print("Interpolating")
    interpolated_force_list = interpolate(joint_values_timestamps_preinterp[bag_index],  phantom_force_sensor_timestamps_preinterp[bag_index], phantom_force_sensor_preinterp[bag_index])
    phantom_force_sensor[bag_index].extend(interpolated_force_list)
    joint_values[bag_index].extend(joint_values_preinterp[bag_index])

    #can handle multiple datapoints per row
    count = 0
    for bag_index, element in phantom_force_sensor.items():
        for i, force_tuple in enumerate(element):
            if count % datapoints_per_row == 0:
                if count != 0:
                    list_of_instances.append(dummy)
                    list_of_labels.append(dummy_2[len(dummy_2)-1])

                dummy = []
                dummy_2 = []
            dummy.extend(joint_values[bag_index][i])
            dummy_2.append(force_tuple)
            count+=1

    print("Length of instances from current bag is {}".format(len(list_of_instances)))
    print("Length of labels from current bag is {}".format(len(list_of_labels)))
    print("Wrote/Appended to data_file and label_file")

    if bag_index == starting_index:
        mode = "w"
    else:
        mode = "a"

    with open(data_file_name, mode = mode, newline='') as outfile:
        outfile_handle = csv.writer(outfile)
        outfile_handle.writerows(list_of_instances)

    with open(label_file_name, mode = mode, newline='') as outfile:
        outfile_handle = csv.writer(outfile)
        outfile_handle.writerows(list_of_labels)

    
    print("Finishes {}bag_{}_joint_values".format(directory, bag_index))
    print("Finishes {}bag_{}_phantom_force_sensor".format(directory, bag_index))
    print("--------------------------------------------------------------------")

datapoints_per_row = 100
directory = "./data/"
data_file_name = "data_file"
label_file_name = "label_file"
num_bags = sys.argv[1]
starting_index = sys.argv[2]

def main():

    start_time = time.time()
    for i in range(int(num_bags)):
        parsing(use_interpolation, int(starting_index)+i)
    elapsed_time = time.time() - start_time
    print("The program took {} seconds".format(elapsed_time))

if __name__ == "__main__":
    main()

