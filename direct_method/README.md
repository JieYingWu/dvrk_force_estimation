**dVRK Deep Learning for Force Sensing and Haptic Feedback**

Paper: *A Deep Learning Approach to Intrinsic Force Sensing on the da Vinci Surgical Robot* <br />
Authors: Nam Tran, Jie Ying Wu, Anton Deguet, Peter Kazanzides

- This repository contains a Keras-based deep neural network (DNN) that can train and make predictions with sliding window or non-overlapping modes, haptic feedback script and other miscellaneous data preparation and handling scripts.
- Compatible with Python 2.7 and after. 
- Sometimes, on lab machines, ROS libraries do not work on Python 3 and after, so the user may need to try running with both "python" and "python3". 

This project uses the following libraries (==version tested):
- pyfiglet==0.8.post1
- numpy==1.16.2
- scipy==1.2.1
- Keras==2.2.4
- tensorflow==1.13.1
- matplotlib==3.0.3
- numpy_ringbuffer==0.2.1
- scikit_learn==0.21.3

Complete data flow: 

1. Collect ROS bags from da Vinci robot
2. Read raw ROS bags into processed text files using read_ros_bags.py 
3. (optional) To inspect individual processed phantom force sensor text files through plotting, use data_plotter.py
3. Transform and prepare the the text files into data_file and label_file using data_prep.py 
4. Train the neural network using neural_network.py 
5. Test or use haptic feedback with the trained model either with the test mode of neural_network.py or dvrk_mtm_haptic_feedback.py
6. (optional) To open pickled figs later, use open_pickled_plots.py 

The folder best_model contains all the files associated with the best predictive model thus far trained. The folder processed_test_files contains two processed bags used for testing; 63 is horizontal configuration and 99 is vertical configuration. 


----------------------------------------------------------data_plotter.py----------------------------------------------------------

This script plots the x, y and z components of a processed phantom force sensor data file. <bag_id> is the index of the desired processed data file.

    python data_plotter.py <bag_id>

----------------------------------------------------------data_prep.py----------------------------------------------------------

This script takes in processed data files as inputs and yields 2 files as outputs: data_file and label_file. This script should only be run whenever there is new data collected from the robot. Interpolation is used to align phantom force sensor data with joint values in time. <num_bags> stands for the number of processed bags that will be used to extract data from. The <starting_index> is the index of the first bag in the sequence of desired bags. Note that the input processed data files must be in the folder named "data". This folder, "data", must be in the same directory as the neural_network.py script.

    python data_prep.py <num_bags> <starting_index>

----------------------------------------------------------train_direct.py----------------------------------------------------------

This script is used for training

    ``python train_direct.py``

----------------------------------------------------------haptic_feedback.py----------------------------------------------------------

This script contains the haptic system, available in 2 modes: neural_network and force_sensor. When using force_sensor mode, the script generates haptic feedback directly from the force sensor. When using neural_network mode, the script generates haptic feedback using real-time predictions of the neural network. In both cases, the script outputs the feedback from the force sensor or the neural network without any adjustments. 

	python haptic_feedback.py force_sensor
	python haptic_feedback.py neural_network <model_for_use>

----------------------------------------------------------open_pickled_plots.py----------------------------------------------------------

This script shows pickled plots in the x, y and z directions. Doing predictions on a model generates 3 pickled plots, since we have 3 directions: x, y, z. For example, running neural network test on the model "demo" will generate 3 pickled files: X - demo, Y - demo and Z - demo. 

    python open_pickled_plots.py <path_to_model_file>

----------------------------------------------------------zoomed_inset_snips.py----------------------------------------------------------

This script contains two analyze_results functions, one to draw horizontal configuration plots and one to draw vertical configuration plots (all with zoomed insets). To do plotting, the user simply needs to copy the desired analyze_results function from this script and overwrite the existing analyze_results function in the neural_network.py script. __

----------------------------------------------------------error_distribution.py----------------------------------------------------------

This script creates the two error distribution plots, one for horizontal configuration and one for vertical configuration. It requires two input files to be put in the same directory: horizontal_prediction_error and vertical_prediction_error. These two input files can be generated from neural_network.py by writing out the error. 

    python error_distribution.py
