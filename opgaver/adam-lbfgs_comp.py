import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from softadapt import *
import numpy as np
import scipy.io
import os
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import pandas as pd
import re

with open('data/nls-adam.txt', 'r') as file:
    lines_adam = file.readlines()
with open('data/nls-lbfgs.txt', 'r') as file:
    lines_lbfgs = file.readlines()

# Initialize lists to store the epoch numbers and loss values
epochs_adam = []
losses_adam = []
losses_lbfgs = []
pattern_adam = re.compile(r'Epoch \[(\d+)/50000\], Loss: (\d+\.\d+)')
pattern_lfbgs = re.compile(r'Loss: (\d+\.\d+)')
# Iterate over the lines and extract the epoch numbers and loss values
for line in lines_adam:
    match = pattern_adam.search(line)
    re.split(r'Epoch \[()',line)
    if match:
        epochs_adam.append(int(match.group(1)))
        losses_adam.append(float(match.group(2)))
for line in lines_lbfgs:
    match = pattern_lfbgs.search(line)
    re.split(r'Epoch \[()',line)
    if match:
        losses_lbfgs.append(float(match.group(1)))
epochs_lbfgs = np.arange(start = epochs_adam[-1],stop=epochs_adam[-1]+len(losses_lbfgs))
# New list to store the modified losses
new_losses_adam = []

# Number of neighbors to consider on each side
n_neighbors = 5

# Iterate through the losses list
for i in range(len(losses_adam)):
    # Determine the start and end of the window
    start = max(0, i - n_neighbors)
    end = min(len(losses_adam), i + n_neighbors + 1)
    
    # Compute the minimum value in the window
    window_min = min(losses_adam[start:end])
    
    # Append the minimum value to the new_losses_adam list
    new_losses_adam.append(window_min)

plt.plot(epochs_adam,new_losses_adam,label = "NLS")
plt.plot(epochs_lbfgs,losses_lbfgs,color='orange', linestyle='dashed',label = "L-BLFGS")
plt.plot
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper center')
plt.yscale('log')
plt.savefig("Adam_solo_performance.svg")
plt.clf()
