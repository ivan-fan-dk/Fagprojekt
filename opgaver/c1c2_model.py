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

hidden_units = 2**7
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(4,hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units,hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units,hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units,2)
        )

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, y, c1, c2):
        x_combined = torch.cat((x, y, c1, c2),dim=4)
        logits = self.fn_approx(x_combined)
        return logits
    
model = NeuralNetwork().to("cpu")

# Load the state dict previously saved
state_dict = torch.load("schrodinger_model_c1c2_from_hpc.pth", map_location=torch.device("cpu"))

# Load the state dict to the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

#do some freaky plots:
u_exact = np.loadtxt("opgaver/u (2).txt", dtype=complex)
x_exact = np.loadtxt("opgaver/x (2).txt")
t_exact = np.loadtxt("opgaver/t (2).txt")

import matplotlib.pyplot as plt
#do the abs value thing:
u_abs_exact = np.abs(u_exact)
print("here", u_abs_exact.shape)


t1 = np.argmin(np.abs(t_exact-0.59))
#our closes t-val is 0.785....
print(t_exact[t1])

t2 = np.argmin(np.abs(t_exact-0.79))
#our closes t-val is 0.785....
print(t_exact[t2])

t3 = np.argmin(np.abs(t_exact-0.98))
#our closes t-val is 0.785....
print(t_exact[t3])

#real stuff:
#import scipy.io
#data = scipy.io.loadmat("opgaver/_static/NLS.mat")
#
#t = data['tt'].flatten()[:,None]
#x = data['x'].flatten()[:,None]
#Exact = data['uu']
#Exact_u = np.real(Exact)
#Exact_v = np.imag(Exact)
#Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)
#print(Exact_h.shape)

#now add our models prediction to that plot:
x_py = torch.tensor(x_exact).float().view(-1,1,1,1,1)
t_py_1 = t_exact[t1] * torch.ones_like(x_py).view(-1,1,1,1,1)
t_py_2 = t_exact[t2] * torch.ones_like(x_py).view(-1,1,1,1,1)
t_py_3 = t_exact[t3] * torch.ones_like(x_py).view(-1,1,1,1,1)
c1 = torch.ones_like(x_py).view(-1,1,1,1,1)
c2 = 2*torch.ones_like(x_py).view(-1,1,1,1,1)

u_pred_1 = model(x_py,t_py_1,c1,c2).detach()
u_pred_1 = u_pred_1[:,:,0,0,:].squeeze(-1).numpy()
u_pred_abs_1 = np.sqrt(u_pred_1[:,:,0]**2 + u_pred_1[:,:,1]**2)

u_pred_2 = model(x_py,t_py_2,c1,c2).detach()
u_pred_2 = u_pred_2[:,:,0,0,:].squeeze(-1).numpy()
u_pred_abs_2 = np.sqrt(u_pred_2[:,:,0]**2 + u_pred_2[:,:,1]**2)

u_pred_3 = model(x_py,t_py_3,c1,c2).detach()
u_pred_3 = u_pred_3[:,:,0,0,:].squeeze(-1).numpy()
u_pred_abs_3 = np.sqrt(u_pred_3[:,:,0]**2 + u_pred_3[:,:,1]**2)

#plotting
#plt.plot(x_exact,u_abs_exact[:,151], label="Exact")
##plt.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')
#plt.plot(x_exact,u_pred_abs, label="Predicted", linestyle="--")
##add a label and axis labels:
#plt.xlabel("x")
#plt.ylabel("|u|")
#plt.title("Exact solution at t=0.79")
#plt.savefig("u_abs_exact.png")

#stuff:

fig, ax = plt.subplots()
ax.axis('off')

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)


ax = plt.subplot(gs1[0, 0])
ax.plot(x_exact,u_abs_exact[:,t1], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x_exact,u_pred_abs_1, 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$|h(t,x)|$')    
ax.set_title('$t = %.2f$' % (t_exact[t1]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-0.1,5.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x_exact,u_abs_exact[:,t2], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x_exact,u_pred_abs_2, 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$|h(t,x)|$')    
ax.set_title('$t = %.2f$' % (t_exact[t2]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-0.1,5.1])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x_exact,u_abs_exact[:,t3], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x_exact,u_pred_abs_3, 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$|h(t,x)|$')    
ax.set_title('$t = %.2f$' % (t_exact[t3]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-0.1,5.1])

plt.savefig('c1c2_plot.png')
