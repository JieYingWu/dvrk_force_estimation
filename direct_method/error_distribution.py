# Author: Nam Tran
# Date: 2020-10-21

# This code creates the two distribution plots (horizontal and vertical), requiring two files as input:
# horizontal_prediction_error and vertical_prediction_error. These two input files can be generated 
# from neural_network.py by writing out the error.  

import numpy as np
import csv
import random
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
import statistics
import pandas as pd

with open("horizontal_prediction_error", "r", newline = "") as f:
    reader = csv.reader(f)
    h_data = list(reader)

with open("vertical_prediction_error", "r", newline = "") as f:
    reader = csv.reader(f)
    v_data = list(reader)[:len(h_data)]

components = ["X-component", "Y-component", "Z-component"]

h_error = {}
v_error = {}
for c in components:
    h_error[c] = []
    v_error[c] = []

for i in range(len(v_data)):
    for j in range(len(v_data[i])):
        h_data[i][j] = float(h_data[i][j])
        v_data[i][j] = float(v_data[i][j])
    v_error["X-component"].append(v_data[i][0])
    v_error["Y-component"].append(v_data[i][1])
    v_error["Z-component"].append(v_data[i][2])

    h_error["X-component"].append(h_data[i][0])
    h_error["Y-component"].append(h_data[i][1])
    h_error["Z-component"].append(h_data[i][2])


colors = ["magenta", "tan", "lime"]

df_v=pd.DataFrame.from_dict(v_error,orient='index').transpose()


f, axs = plt.subplots(3, sharex=False, gridspec_kw={'hspace': 0.65}) # "height_ratios": (.16, .16, .16, .16, .16, .16)

c_palette = {"X-component": "magenta","Y-component":  "tan", "Z-component": "lime"}
#sns.violinplot(data=df_v, palette = c_palette, orient="h").set(
#    xlabel='Error (N)'
#) #ax=axs[0]
sns.distplot(df_v["X-component"], ax=axs[0], color = colors[0], hist = False, kde = True).set(
    xlabel='Error for X-Component (N)', 
    ylabel='Density'
)
sns.distplot(df_v["Y-component"], ax=axs[1], color = colors[1], hist = False, kde = True).set(
    xlabel='Error for Y-component (N)', 
    ylabel='Density'
)
sns.distplot(df_v["Z-component"], ax=axs[2], color = colors[2], hist = False, kde = True).set(
    xlabel='Error for Z-component (N)', 
    ylabel='Density'
)
f.suptitle("Error Distribution for Vertical Configuration")



df_h=pd.DataFrame.from_dict(h_error,orient='index').transpose()


f_h, axs_h = plt.subplots(3, sharex=False, gridspec_kw={'hspace': 0.65}) # "height_ratios": (.16, .16, .16, .16, .16, .16)

#sns.violinplot(data=df_h, palette = c_palette, orient="h").set( 
#    xlabel='Error (N)'
#) #ax=axs_h[0], 

sns.distplot(df_h["X-component"], ax=axs_h[0], color = colors[0], hist = False, kde = True).set(
    xlabel='Error for X-component (N)', 
    ylabel='Density'
)
sns.distplot(df_h["Y-component"], ax=axs_h[1], color = colors[1], hist = False, kde = True).set(
    xlabel='Error for Y-component (N)', 
    ylabel='Density'
)
sns.distplot(df_h["Z-component"], ax=axs_h[2], color = colors[2], hist = False, kde = True).set(
    xlabel='Error for Z-component (N)', 
    ylabel='Density'
)
f_h.suptitle("Error Distribution for Horizontal Configuration")

sns.set(font_scale=2)
plt.show()
 
#import pdb; pdb.set_trace()
#mu=np.array([1,10,20])
# Let's change this so that the points won't all lie in a plane...
#sigma=np.matrix([[20,10,10],
#                 [10,25,1],
#                 [10,1,50]])

#data=np.random.multivariate_normal(mu,sigma,1000)
#random.shuffle(data)
#data = np.array(data[:10000])
#values = data.T

'''x = np.array(x)
y = np.array(y)
z = np.array(z)

#print(x.shape)

xyz = np.vstack([x,y,z])
kde = stats.gaussian_kde(xyz)


def calc_kde(data):
    print(len(data))
    return kde(data.T)

def main():


    # Evaluate kde on a grid
    xmin, ymin, zmin = x.min(), y.min(), z.min()
    xmax, ymax, zmax = x.max(), y.max(), z.max()
    xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=4)
    results = pool.map(calc_kde, np.array_split(coords.T, 2))
    density = np.concatenate(results).reshape(xi.shape)
    #density = kde(coords).reshape(xi.shape)

    # Plot scatter with mayavi
    figure = mlab.figure('DensityPlot')

    grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
    min = density.min()
    max = density.max()
    mlab.pipeline.volume(grid, vmin=min ,vmax=min + .8*(max-min)) #, vmin=min ,vmax=min + .5*(max-min)

    mlab.axes()
    mlab.savefig(filename='test.png', magnification = 10)
    mlab.show()

if __name__ == '__main__':   
    main()

'''