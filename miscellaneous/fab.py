import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from softadapt import *
import numpy as np
import scipy.io
import os
from mpl_toolkits.mplot3d import Axes3D

# c er bare en eller anden konstant
c = 0.5
x0 = 0
y0 = 0
z0 = 0
N = 20

# Define boundary conditions
t0 = 0.0
t_final = 5
x_left = 0.0
x_right = 5
y_left = 0
y_right = 5
z_left = 0
z_right = 5

# Create input data
X_vals = torch.linspace(x_left, x_right, N, requires_grad=True)
Y_vals = torch.linspace(y_left, y_right, N, requires_grad=True)
Z_vals = torch.linspace(z_left, z_right, N, requires_grad=True)
t_vals = torch.linspace(t0, t_final, N, requires_grad=True)
X_train, Y_train, Z_train, t_train = torch.meshgrid(X_vals, Y_vals, Z_vals, t_vals, indexing="xy")
X_train = X_train.unsqueeze(-1)
Y_train = Y_train.unsqueeze(-1)
Z_train = Z_train.unsqueeze(-1)
t_train = t_train.unsqueeze(-1)

# Define initial and boundary conditions
h = lambda x, y, z: torch.sin(x + y + z)  
g1 = lambda t: torch.zeros_like(t)  
g2 = lambda t: torch.zeros_like(t)  

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn_approx = nn.Sequential(
            nn.Linear(4,16),
            nn.Tanh(),
            nn.Linear(16,8),
            nn.Tanh(),
            nn.Linear(8,16),
            nn.Tanh(),
            nn.Linear(16,1)
        )

    def forward(self, x, y, z, t):
        x_combined = torch.cat((x, y, z, t),dim=4)
        logits = self.fn_approx(x_combined)
        return logits
   
model = NeuralNetwork()

# Forward pass
u_prediction = model(X_train, Y_train, Z_train, t_train)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Tager sådan 25 min
num_epochs = 2000

softadapt_object = SoftAdapt(beta=0.1)

epochs_to_make_updates = 5

values_of_component_1 = []
values_of_component_2 = []
values_of_component_3 = []

adapt_weights = torch.tensor([1,1,1])

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(X_train, Y_train, Z_train, t_train)
    # Compute the first derivatives
    du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    du_dy = torch.autograd.grad(u_prediction, Y_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    du_dz = torch.autograd.grad(u_prediction, Z_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

    du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    
    d2u_dx2 = torch.autograd.grad(du_dx, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    d2u_dy2 = torch.autograd.grad(du_dy, Y_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    d2u_dz2 = torch.autograd.grad(du_dz, Z_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

    d2u_dt2 = torch.autograd.grad(du_dt, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    # Compute the loss
    loss_PDE = criterion(c**2 * (d2u_dx2 + d2u_dy2 + d2u_dz2) , d2u_dt2)
    loss_boundary = criterion(model(x0*torch.ones_like(X_train), y0*torch.ones_like(X_train), z0*torch.ones_like(X_train), t_train), g1(t_train)) \
              + criterion(model(x_right*torch.ones_like(X_train), y_right*torch.ones_like(X_train), z_right*torch.ones_like(X_train), t_train), g2(t_train))
    loss_IC = criterion(model(X_train, Y_train, Z_train, t0*torch.ones_like(t_train)), h(X_train, Y_train, Z_train))
    loss = (loss_PDE + loss_boundary + loss_IC)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    values_of_component_1.append(loss_PDE)
    values_of_component_2.append(loss_boundary)
    values_of_component_3.append(loss_IC)


    if epoch % epochs_to_make_updates == 0 and epoch != 0:
        adapt_weights = softadapt_object.get_component_weights(
        torch.tensor(values_of_component_1), 
        torch.tensor(values_of_component_2), 
        torch.tensor(values_of_component_3),
        verbose=False,
        )
                                                            
        # Resetting the lists to start fresh (this part is optional)
        values_of_component_1 = []
        values_of_component_2 = []
        values_of_component_3 = []

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    u_pred = model(X_train, Y_train, Z_train, t_train)
    u_pred =  u_pred.squeeze(-1).numpy()
    X_train = X_train.squeeze(-1).numpy()
    Y_train = Y_train.squeeze(-1).numpy()
    Z_train = Z_train.squeeze(-1).numpy()
    t_train = t_train.squeeze(-1).numpy()

#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X_train, Y_train, Z_train, c=u_pred, cmap='viridis', linewidth=0.5)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.show()