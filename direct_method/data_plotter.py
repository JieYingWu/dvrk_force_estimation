# Author: Nam Tran
# Date: 2019-08-13

# To run the script: 
# > python data_plotter.py <bag_id>

import matplotlib.pyplot as plt
import pickle as pl
import operator as op
import time
import sys
import pyfiglet
import random
import string
import sys
import numpy as np
import threading
import csv
import io   

bag_id = sys.argv[1]
phantom_force_sensor_raw = np.loadtxt("./bag_" + str(bag_id) + "_phantom_force_sensor", delimiter = ",")

x_component = []
y_component = []
z_component = []

for element in phantom_force_sensor_raw:
        x_component.append(element[1])
        y_component.append(element[2])
        z_component.append(element[3])

fig, axs = plt.subplots(3, sharex='row')
fig.suptitle("Phantom force sensor data")
axs[0].plot(x_component, label = "X-component")
axs[0].set(xlabel = 'X-component')
axs[0].legend()
axs[1].plot(y_component, label = "Y-component")
axs[1].set(xlabel = 'Y-component')
axs[1].legend()
axs[2].plot(z_component)
axs[2].set(xlabel = 'Z-component', label = "Z-component")
axs[2].legend()
plt.grid()
plt.show()
