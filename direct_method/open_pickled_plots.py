# Author: Nam Tran
# Date: 2019-08-13

# To open a pickled plot: 
# > python open_pickled_plots.py <path_to_model_file>

import pickle
import matplotlib.pyplot as plt
import sys

def main():
    name = sys.argv[1]
    figx = pickle.load(open("X - {}".format(name), 'rb'))
    figy = pickle.load(open("Y - {}".format(name), 'rb'))
    figz = pickle.load(open("Z - {}".format(name), 'rb'))

    plt.show()

if __name__ == "__main__":
    main()