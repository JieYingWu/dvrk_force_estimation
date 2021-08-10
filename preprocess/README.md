----------------------------------------------------------read_ros_bags.py----------------------------------------------------------

This script reads raw ROS bags into csv files. Makes subdirectories in the given output path for each of the topics it reads, including joint states, Jacobians, sensor readings, jaw states, and two FSR readings. Data is saved in separate files and the not interpolated.

    python read_ros_bag.py -f <Path to Rosbag folder> -o <Path to write out parsed csv> --prefix <Prefix for the output csv name> --index <Starting index for the output csv name>
	
----------------------------------------------------------preprocess.py----------------------------------------------------------

Pared down version fo read_ros_bag.py to read joints states and Jacobians only for creating training data. Expects data file structure to be the path to the data folder followed by folders to the different splits (train, val, or test) <path to data>/<train, val, test>. All data is interpolated to the same time and saved in a single file. 

    python preprocess.py <path to data> <split to process>
	
----------------------------------------------------------read_calibration.py----------------------------------------------------------

Pared down script to read the FSR and the ATI measurements. Option to start numbering the output files at different index to run on different batches of data.

	python read_calibration.py -f <Path to Rosbag folder> -o <Path to write out parsed csv> --prefix <Prefix for the output csv name> --index <Starting index for the output csv name>