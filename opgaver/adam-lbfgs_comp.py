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

with open('data/output_schrodinger_plain_adam_hpc.txt', 'r') as file:
    lines_adam = file.readlines()

with open('output_schrodinger_plain_adam_sobel_regularization.txt', 'r') as file:
    lines_adam_opt = file.readlines()   

with open('data/nls-lbfgs.txt', 'r') as file:
    lines_lbfgs = file.readlines()

# Initialize lists to store the epoch numbers and loss values
epochs_adam = []
losses_adam = []
losses_adam_opt = []
losses_lbfgs = []
pattern_adam = re.compile(r'Epoch \[(\d+)/55000\], Loss: (\d+\.\d+)')
pattern_adam_opt = re.compile(r'Epoch \[(\d+)/50000\], Loss: (\d+\.\d+)')
pattern_lfbgs = re.compile(r'Loss: (\d+\.\d+)')
# Iterate over the lines and extract the epoch numbers and loss values

for line in lines_adam:
    match = pattern_adam.search(line)
    re.split(r'Epoch \[()',line)
    if match:
        epochs_adam.append(int(match.group(1)))
        losses_adam.append(float(match.group(2))/1000)
losses_adam = losses_adam[0:50000]
epochs_adam = epochs_adam[0:50000]

for line in lines_adam_opt:
    match = pattern_adam_opt.search(line)
    re.split(r'Epoch \[()',line)
    if match:
        #epochs_adam.append(int(match.group(1)))
        losses_adam_opt.append(float(match.group(2))/1000)


for line in lines_lbfgs:
    match = pattern_lfbgs.search(line)
    re.split(r'Epoch \[()',line)
    if match:
        losses_lbfgs.append(float(match.group(1))/1000)
epochs_lbfgs = np.arange(start = 50000,stop=50000+len(losses_lbfgs))
# New list to store the modified losses


plt.plot(epochs_adam,losses_adam,label = "Adam_original")
plt.plot(epochs_adam,losses_adam_opt,label = "Adam_optimized")
#plt.plot(epochs_lbfgs,losses_lbfgs,color='orange', linestyle='dashed',label = "L-BLFGS")
plt.plot
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper center')
plt.yscale('log')
plt.savefig("Adam_opt_simple_comp.png")
plt.clf()

plt.plot(epochs_adam,losses_adam,label = "Adam_original")
#plt.plot(epochs_lbfgs,losses_lbfgs,color='orange', linestyle='dashed',label = "L-BLFGS")
plt.plot
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper center')
plt.yscale('log')
plt.savefig("Adam_solo_performance.svg")
plt.clf()
