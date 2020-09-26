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
            joint_path = Path(self.output) / "fsr"
            joint_path.mkdir(mode=0o777, parents=False)
            jacobian_path = Path(self.output) / "ati"
            jacobian_path.mkdir(mode=0o777, parents=False)

        except OSError:
            print("Data path exists")

    def single_datapoint_processing(self, file_name): # parses data into the global dictionaries
   
        bag = rosbag.Bag(self.folder+file_name, 'r')

        ati = []
        ati_timestamps = []
        fsr = []
        fsr_timestamps = []


        length_ati = 0
        length_fsr = 0

        print("Processing " + file_name)
        fsr_msg = bag.read_messages(topics=['/FSR'])
        for topic, msg, t in fsr_msg:
            fsr_timestamps.append(t.secs+t.nsecs*10**-9)
            fsr.append(msg.data)
            length_fsr+=1
            
        wrench = bag.read_messages(topics=['/atinetft/wrench'])
        for topic, msg, t in wrench:
            timestamps = t.secs+t.nsecs*10**-9
            ati_timestamps.append(timestamps)
            x = msg.wrench.force.x
            y = msg.wrench.force.y
            z = msg.wrench.force.z # the sensor is probably most accurate in the z direction
            ati.append([x,y,z])
            length_ati+=1

        bag.close()
                                      
        print("Processed wrench: counts: {}".format(length_ati))
        print("Processed fsr: count: {}".format(length_fsr))

        start_time = fsr_timestamps[0]
        fsr_timestamps = np.array(fsr_timestamps) - start_time
        fsr = np.column_stack((fsr_timestamps, fsr))
        ati_timestamps = np.array(ati_timestamps) - start_time
        ati = np.column_stack((ati_timestamps, ati))

        return fsr, ati

    def write(self, fsr, ati):
        file_name = self.prefix + str(self.index)        
        np.savetxt(self.output + "fsr/" + file_name + ".csv", fsr, delimiter=',')
        np.savetxt(self.output + "ati/" + file_name + ".csv", ati, delimiter=',')
        print("Wrote out " + file_name)
        print("")
        
    def parse_bags(self):

        print("\nParsing\n")
        files = os.listdir(self.folder)
        files.sort()
        for file_name in files:
            if file_name.endswith('.bag'):
                fsr, ati = self.single_datapoint_processing(file_name)
                self.write(fsr, ati)
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

