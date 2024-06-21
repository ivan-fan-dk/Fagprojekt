import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from softadapt import *
import numpy as np
from pyDOE import lhs
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import os
import surf2stl
from opgaver.u_exact import u_exact
from time import process_time
def generate_lhs_samples(num_samples, x_range, t_range):
    # Generate LHS matrix with 2 columns (one for each parameter)
    LHS_matrix = lhs(2, samples=num_samples)
    
    # Scale the LHS samples to the range of x and t
    x_samples = LHS_matrix[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    t_samples = LHS_matrix[:, 1] * (t_range[1] - t_range[0]) + t_range[0]
    
    return torch.tensor(x_samples, dtype=torch.float32,requires_grad=True), torch.tensor(t_samples, dtype=torch.float32,requires_grad=True)


# Placeholder for MSE errors and number of parameters

param_counts = []
time_bucket = []
n_hidden_units = 2**(np.arange(2,7))#7
N = 400
num_epochs = 50
batch_size = N
mse_errors = np.zeros((len(n_hidden_units),num_epochs))
time_p3_bucket = np.empty((len(n_hidden_units),num_epochs))
# Define boundary conditions
t0 = 0.0
t_final = torch.pi/4
x_left = -1.
x_right = 1.
# Use LHS to generate samples
X_samples, t_samples = generate_lhs_samples(N, (x_left, x_right), (t0, t_final))
# Create input data
X_vals = torch.linspace(x_left, x_right, N, requires_grad=True)
t_vals = torch.linspace(t0, t_final, N, requires_grad=True)
X_train_full, t_train_full = torch.meshgrid(X_vals, t_vals, indexing="xy")
train_dataset = TensorDataset(X_samples.unsqueeze(-1), t_samples.unsqueeze(-1))  # Format for DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # DataLoader for handling mini-batches

X_train_full_np = X_train_full.detach().numpy()
t_train_full_np = t_train_full.detach().numpy()
X_vals_ = X_vals.unsqueeze(-1).unsqueeze(-1)
t_vals_ = t_vals.unsqueeze(-1).unsqueeze(-1)

# Define functions h(x), u(x)
phi = lambda x: 2/torch.cosh(x) # initial condition


# Get analytical values here
v=0
A=2
c=0
c1=0.5
c2=1.0
x0=0
u_anal_full_re = (u_exact(X_train_full_np,t_train_full_np,v,A,c,c1,c2,x0)).real.squeeze()
u_anal_full_im = (u_exact(X_train_full_np,t_train_full_np,v,A,c,c1,c2,x0)).imag.squeeze()

#to simulate a complex output we make it spit out two things like this [real, imaginary]
class NeuralNetwork(nn.Module):

    def __init__(self, num_node):
        super().__init__()

        self.fn_approx = nn.Sequential(
            nn.Linear(2,num_node),
            nn.Tanh(),
            nn.Linear(num_node,num_node),
            nn.Tanh(),
            nn.Linear(num_node,num_node),
            nn.Tanh(),
            nn.Linear(num_node,num_node),
            nn.Tanh(),
            nn.Linear(num_node,2)
        )

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, y):
        if x.dim() != y.dim():
            raise AssertionError(f"x and y must have the same number of dimensions, but got x.dim() == {x.dim()} and y.dim() == {y.dim()}")
        x_combined = torch.cat((x, y),dim=2)
        logits = self.fn_approx(x_combined)
        return logits

for n in range(len(n_hidden_units)):       
    model = NeuralNetwork(num_node=n_hidden_units[n])
    model.apply(NeuralNetwork.init_weights)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_counts.append(param_count)    

    #setup for softadapt:

    softadapt_object = SoftAdapt()

    epochs_to_make_updates = 1

    values_of_component_1 = []
    values_of_component_2 = []
    values_of_component_3 = []
    values_of_component_4 = []


    # Initializing adaptive weights to all ones.
    adapt_weights = torch.tensor([1,1,1,1])
    tik = process_time()
    tik_p3 = process_time()
    for epoch in range(num_epochs):
        i = 0
        for X_train, t_train in train_loader:
            i+=1
                # Forward pass
            X_train = X_train.unsqueeze(-1)
            t_train = t_train.unsqueeze(-1)
            u_prediction = model(X_train, t_train)
            u_real = u_prediction[:,:,0].unsqueeze(-1)
            u_imag = u_prediction[:,:,1].unsqueeze(-1)

            u_left = model(x_left*torch.ones_like(X_vals_),t_vals_)
            u_right = model(x_right*torch.ones_like(X_vals_),t_vals_)
            
            u_ic_real = model(X_vals_, torch.zeros_like(t_vals_))[:,:,0].unsqueeze(-1)
            u_ic_imag = model(X_vals_, torch.zeros_like(t_vals_))[:,:,1].unsqueeze(-1) 

            # Compute the first derivatives
            du_dx_real = torch.autograd.grad(u_real, X_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
            du_dx_imag = torch.autograd.grad(u_imag, X_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]

            du_dt_real = torch.autograd.grad(u_real, t_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
            du_dt_imag = torch.autograd.grad(u_imag, t_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]

            d2u_dx2_real = torch.autograd.grad(du_dx_real, X_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
            d2u_dx2_imag = torch.autograd.grad(du_dx_imag, X_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]
            
            
            bound_left_real = du_dx_real[:,0,0].unsqueeze(-1)
            bound_left_imag = du_dx_imag[:,0,0].unsqueeze(-1)

            bound_right_real = du_dx_real[:,-1,0].unsqueeze(-1)
            bound_right_imag = du_dx_imag[:,-1,0].unsqueeze(-1)

            # Compute the loss for the nonlinear schrodinger eq:
            loss_PDE_real = criterion(-du_dt_imag + 0.5 * d2u_dx2_real + (u_real**2 + u_imag**2) * u_real, torch.zeros_like(u_real))
            loss_PDE_imag = criterion(du_dt_real + 0.5 * d2u_dx2_imag + (u_real**2 + u_imag**2) * u_imag, torch.zeros_like(u_imag))
            loss_PDE = loss_PDE_real + loss_PDE_imag

            loss_IC = criterion(u_ic_real, phi(X_vals_))+criterion(u_ic_imag, torch.zeros_like(X_vals_))

            loss_boundary_2 = criterion(bound_left_real, bound_right_real)+criterion(bound_left_imag, bound_right_imag)

            loss_boundary_1 = criterion(u_left, u_right)

            # Backward pass and optimization
            optimizer.zero_grad()

            #softadapt weights update stuff:
            values_of_component_1.append(loss_PDE)
            values_of_component_2.append(loss_boundary_1)
            values_of_component_3.append(loss_boundary_2)
            values_of_component_4.append(loss_IC)


            if i % 2 == 0:
                adapt_weights = softadapt_object.get_component_weights(
                torch.tensor(values_of_component_1), 
                torch.tensor(values_of_component_2), 
                torch.tensor(values_of_component_3),
                torch.tensor(values_of_component_4),
                verbose=False,
                )                                         
             
                # Resetting the lists to start fresh (this part is optional)
                values_of_component_1 = []
                values_of_component_2 = []
                values_of_component_3 = []
                values_of_component_4 = []

            # Change 5: Update the loss function with the linear combination of all components.
            loss = (adapt_weights[0] * loss_PDE + adapt_weights[1] * loss_boundary_1 + adapt_weights[2] * loss_boundary_2 + adapt_weights[3]*loss_IC)

            loss.backward()
            optimizer.step()
        
        tok_p3 = process_time()
        time_p3_bucket[n,epoch] = tok_p3-tik_p3
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
        X_train_np = X_train.detach().numpy()
        t_train_np = t_train.detach().numpy()
        with torch.no_grad():
            #u_pred_new = model(X_train_full.unsqueeze(-1),t_train_full.unsqueeze(-1)).numpy()
            #u_pred_abs = u_pred_new[:,:,0]**2+u_pred_new[:,:,1]**2
            #u_re_abs = u_anal_full_re**2+u_anal_full_im**2
            mse_errors[n,epoch]=loss.item()
    tok = process_time()
    time_bucket.append(tok-tik)

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
epoch_list = np.arange(0,num_epochs)
fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(14, 6))  # 1 row, 3 columns

# Left subplot
for k in range(len(n_hidden_units)):
    ax1.plot(epoch_list, mse_errors[k, :], label=f'Model with {param_counts[k]} parameters')
ax1.set_yscale('log')
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('MSE Error', fontsize=14)
ax1.set_title('Error as a function of epochs', fontsize=16)
#ax1.legend(title="Hidden Units", fontsize=12, title_fontsize=13)
ax1.grid(True)
ax1.tick_params(labelsize=12)  # Adjust font size for ticks

# Right subplot (Placeholder - you can replace this with your actual plotting code)
# For demonstration, this will just plot a simple line - replace with your desired content
ax2.plot(param_counts, time_bucket, marker='o') 
ax2.set_xlabel('Number of parameters', fontsize=14)
ax2.set_ylabel('CPU-time(s)', fontsize=14)
ax2.set_title(f'Time complexity O(n)', fontsize=16)
ax2.grid(True)
ax2.tick_params(labelsize=12)
print(param_counts)
print(time_bucket)
for k in range(len(n_hidden_units)):
    ax3.plot(mse_errors[k,:],time_p3_bucket[k,:])
ax3.set_xlabel('Error', fontsize=14)
#ax3.set_xscale('log')

ax3.set_xscale('log')
ax3.set_ylabel('CPU-time', fontsize=14)
ax3.set_title(f'CPU time as a function of errors', fontsize=16)
ax3.grid(True)
ax3.tick_params(labelsize=12)
ax3.set_ylim(0,13)
print(time_p3_bucket[k,:])
# Position the legend centrally below the subplots
handles, labels = ax1.get_legend_handles_labels()  # Get handles and labels for the legend

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.3), ncol=3, fontsize=12)
plt.tight_layout(rect=[0, 0.1, 1, 0.95], pad=0.5, h_pad=0.5, w_pad=0.5)
plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Adjust these parameters to control space between subplots
plt.savefig("opgaver/_static/Algoefficiency.svg", bbox_inches='tight')
plt.clf()

