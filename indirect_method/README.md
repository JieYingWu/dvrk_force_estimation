There are many scripts here to calculate errors and plot things in a specific way. They are there for reference for how the errors in the paper are calculated. 

----------------------------------------------------------train.py----------------------------------------------------------

This script reads raw ROS bags into csv files. Makes subdirectories in the given output path for each of the topics it reads, including joint states, Jacobians, sensor readings, jaw states, and two FSR readings. Data is saved in separate files and the not interpolated.

    python read_ros_bag.py -f <Path to Rosbag folder> -o <Path to write out parsed csv> --prefix <Prefix for the output csv name> --index <Starting index for the output csv name>
	

----------------------------------------------------------print_err.m----------------------------------------------------------

Cute little script to print things in Latex format for easy copy and paste. It assumes that the RMS error and standard deviation are already loaded in the Matlab environment

----------------------------------------------------------train.py----------------------------------------------------------

This is the base script to train each network. There are variations on this for training each of the cases listed in the ISMR 2021 paper. It takes two arguemnts, the first is which data to load (choice between 'free_space' and 'trocar'). The second is a boolean for whether to use the RNN or not. It assumes the path to data is at '../data/csv/< 'train', 'val' >/< 'free_space', 'trocar' >'

	python train.py trocar 1
    

----------------------------------------------------------test.py----------------------------------------------------------

This is the base script to test each network. There are variations on this for testing each of the cases listed in the ISMR 2021 paper. It takes three arguemnts, the first is which experiment to load as a striong, the second is which network to use ('lstm' or not), and the third is whether to use the seal or base case. It assumes the path to data is at '../data/csv/test/< 'no_contact', 'with_contact' >/<exp>

	python test.py <exp> lstm seal