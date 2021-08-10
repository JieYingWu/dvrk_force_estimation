# dVRK Force Estimation
Using neural network to estimate forces on the dVRK. Start with preprocessing to convert from Rosbag to CSV files for training the networks on. Use either the direct or the indirect method for force estimation. The indirect method includes an extension to account for trocar interaction froces. The FSR section is a work in progress to include grip force estimation. Each section is further detailed by the README in its directory. 

We test two methods here 
 * Indirect - which learns the torque to move the joints in free space as described in this paper https://www.researchgate.net/publication/341879597_Neural_Network_based_Inverse_Dynamics_Identification_and_External_Force_Estimation_on_the_da_Vinci_Research_Kit
 * Direct - which learns the forces directly from a sensor in the environment as descriped in this paper https://ieeexplore.ieee.org/abstract/document/9287941