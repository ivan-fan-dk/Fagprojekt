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




N = int(20_000**0.5)

# Define boundary conditions
t0 = 0.0
t_final = torch.pi/2
x_left = -5.
x_right = 5.

# Create input data
c1 = torch.linspace(0.1,3, 10, requires_grad=True)
c2 = torch.linspace(0.1,3, 10, requires_grad=True)
#since B = sqrt((c2 * A**2) / (2c1)) we have to change init conds! (phi!)

X_vals = torch.linspace(x_left, x_right, N, requires_grad=True)
t_vals = torch.linspace(t0, t_final, N, requires_grad=True)
X_train, t_train, c1_train, c2_train = torch.meshgrid(X_vals, t_vals, c1, c2, indexing="xy")

X_train = X_train.unsqueeze(-1)
t_train = t_train.unsqueeze(-1)
c1_train = c1_train.unsqueeze(-1)
c2_train = c2_train.unsqueeze(-1)

X_vals_ = X_vals.view(-1,1,1,1,1)
t_vals_ = t_vals.view(-1,1,1,1,1)
c1_vals_ = c1.view(-1,1,1,1,1)
c2_vals_ = c2.view(-1,1,1,1,1)


# Define functions h(x), u(x)
A = 2
phi = lambda x, c1, c2: A*(torch.cosh(torch.sqrt((c1 * A**2) * (2 * c2)**(-1)) * x))**(-1)
 

hidden_units = 2**7
#to simulate a complex output we make it spit out two things like this [real, imaginary]
#the only change is now we have two inputs more, c1 and c2!
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
    
model = NeuralNetwork()
#model.apply(NeuralNetwork.init_weights)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 200

#setup for softadapt:

softadapt_object = SoftAdapt(beta=0.1)


epochs_to_make_updates = 5

values_of_component_1 = []
values_of_component_2 = []
values_of_component_3 = []
values_of_component_4 = []
values_of_component_5 = []

# Initializing adaptive weights to all ones.
adapt_weights = torch.tensor([1,1,1,1,1])

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(X_train, t_train, c1_train, c2_train)

    u_real = u_prediction[:,:,:,:,0].unsqueeze(-1)
    u_imag = u_prediction[:,:,:,:,1].unsqueeze(-1)

    #this is really bad code, it could be optimized by making another meshgrid with lijke the xleft and xright entries
    #so the dimension is reduced, but since its just MSE it doesnt matter cause its divided by n anyways but really slow!
    ########################################################################################################################
    u_left = model(x_left*torch.ones_like(X_train),t_train, c1_train, c2_train)
    u_right = model(x_right*torch.ones_like(X_train),t_train, c1_train, c2_train)
    
    u_ic_real = model(X_train, torch.zeros_like(t_train), c1_train, c2_train)[:,:,:,:,0].unsqueeze(-1)
    u_ic_imag = model(X_train, torch.zeros_like(t_train), c1_train, c2_train)[:,:,:,:,1].unsqueeze(-1) 

    ########################################################################################################################
    
    # Compute the derivatives
    du_dx_real = torch.autograd.grad(u_real, X_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
    du_dx_imag = torch.autograd.grad(u_imag, X_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]

    du_dt_real = torch.autograd.grad(u_real, t_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
    du_dt_imag = torch.autograd.grad(u_imag, t_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]

    d2u_dx2_real = torch.autograd.grad(du_dx_real, X_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
    d2u_dx2_imag = torch.autograd.grad(du_dx_imag, X_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]
    
    
    bound_left_real = du_dx_real[:,0,:,:,0].unsqueeze(-1)
    bound_left_imag = du_dx_imag[:,0,:,:,0].unsqueeze(-1)

    bound_right_real = du_dx_real[:,-1,:,:,0].unsqueeze(-1)
    bound_right_imag = du_dx_imag[:,-1,:,:,0].unsqueeze(-1)

    # Compute the loss for the nonlinear schrodinger eq:
    loss_PDE_real = criterion(-du_dt_imag + c1_train * d2u_dx2_real + c2_train * (u_real**2 + u_imag**2) * u_real, torch.zeros_like(u_real))
    loss_PDE_imag = criterion(du_dt_real +  c1_train * d2u_dx2_imag + c2_train * (u_real**2 + u_imag**2) * u_imag, torch.zeros_like(u_imag))
    #loss_PDE = loss_PDE_real + loss_PDE_imag

    loss_IC = criterion(u_ic_real, phi(X_train, c1_train, c2_train))+criterion(u_ic_imag, torch.zeros_like(X_train))

    loss_boundary_2 = criterion(bound_left_real, bound_right_real)+criterion(bound_left_imag, bound_right_imag)

    loss_boundary_1 = criterion(u_left, u_right)

    # Backward pass and optimization
    optimizer.zero_grad()

    #softadapt weights update stuff:
    values_of_component_1.append(loss_PDE_real)
    values_of_component_2.append(loss_PDE_imag)
    values_of_component_3.append(loss_boundary_1)
    values_of_component_4.append(loss_boundary_2)
    values_of_component_5.append(loss_IC)


    if epoch % epochs_to_make_updates == 0 and epoch != 0:
        adapt_weights = softadapt_object.get_component_weights(
        torch.tensor(values_of_component_1), 
        torch.tensor(values_of_component_2), 
        torch.tensor(values_of_component_3),
        torch.tensor(values_of_component_4),
        torch.tensor(values_of_component_5),
        verbose=False,
        )
                                                            
      
          # Resetting the lists to start fresh (this part is optional)
        values_of_component_1 = []
        values_of_component_2 = []
        values_of_component_3 = []
        values_of_component_4 = []
        values_of_component_5 = []

      # Change 5: Update the loss function with the linear combination of all components.
    loss = 1000*(adapt_weights[0] * loss_PDE_real + adapt_weights[1] * loss_PDE_imag + adapt_weights[2] * loss_boundary_1 + adapt_weights[3] * loss_boundary_2 + adapt_weights[4]*loss_IC)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')


# Define a closure function to reevaluate the model and loss
def closure():
    lbfgs.zero_grad()
    # Forward pass
    u_prediction = model(X_train, t_train, c1_train, c2_train)

    u_real = u_prediction[:,:,:,:,0].unsqueeze(-1)
    u_imag = u_prediction[:,:,:,:,1].unsqueeze(-1)

    #this is really bad code, it could be optimized by making another meshgrid with lijke the xleft and xright entries
    #so the dimension is reduced, but since its just MSE it doesnt matter cause its divided by n anyways but really slow!
    ########################################################################################################################
    u_left = model(x_left*torch.ones_like(X_train),t_train, c1_train, c2_train)
    u_right = model(x_right*torch.ones_like(X_train),t_train, c1_train, c2_train)
    
    u_ic_real = model(X_train, torch.zeros_like(t_train), c1_train, c2_train)[:,:,:,:,0].unsqueeze(-1)
    u_ic_imag = model(X_train, torch.zeros_like(t_train), c1_train, c2_train)[:,:,:,:,1].unsqueeze(-1) 

    ########################################################################################################################
    
    # Compute the derivatives
    du_dx_real = torch.autograd.grad(u_real, X_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
    du_dx_imag = torch.autograd.grad(u_imag, X_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]

    du_dt_real = torch.autograd.grad(u_real, t_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
    du_dt_imag = torch.autograd.grad(u_imag, t_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]

    d2u_dx2_real = torch.autograd.grad(du_dx_real, X_train, create_graph=True, grad_outputs=torch.ones_like(u_real))[0]
    d2u_dx2_imag = torch.autograd.grad(du_dx_imag, X_train, create_graph=True, grad_outputs=torch.ones_like(u_imag))[0]
    
    
    bound_left_real = du_dx_real[:,0,:,:,0].unsqueeze(-1)
    bound_left_imag = du_dx_imag[:,0,:,:,0].unsqueeze(-1)

    bound_right_real = du_dx_real[:,-1,:,:,0].unsqueeze(-1)
    bound_right_imag = du_dx_imag[:,-1,:,:,0].unsqueeze(-1)

    # Compute the loss for the nonlinear schrodinger eq:
    loss_PDE_real = criterion(-du_dt_imag + c1_train * d2u_dx2_real + c2_train * (u_real**2 + u_imag**2) * u_real, torch.zeros_like(u_real))
    loss_PDE_imag = criterion(du_dt_real +  c1_train * d2u_dx2_imag + c2_train * (u_real**2 + u_imag**2) * u_imag, torch.zeros_like(u_imag))
    #loss_PDE = loss_PDE_real + loss_PDE_imag

    loss_IC = criterion(u_ic_real, phi(X_train, c1_train, c2_train))+criterion(u_ic_imag, torch.zeros_like(X_train))

    loss_boundary_2 = criterion(bound_left_real, bound_right_real)+criterion(bound_left_imag, bound_right_imag)

    loss_boundary_1 = criterion(u_left, u_right)

      # Change 5: Update the loss function with the linear combination of all components.
    loss = 1000*(loss_PDE_real + loss_PDE_imag + loss_boundary_1 + loss_boundary_2 + loss_IC)
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
    loss.backward()
    return loss

lbfgs = optim.LBFGS(model.parameters(), max_iter=200)
lbfgs.step(closure)

#plotting stuff:
with torch.no_grad():
    #for the 3d plot we import the matplotlib extension:
    from mpl_toolkits.mplot3d import Axes3D
    u_pred = model(X_vals_, 0.79*torch.ones_like(t_vals_), 0.5*torch.ones_like(X_vals_), torch.ones_like(X_vals_))
    u_pred = u_pred[:,:,0,0,:].squeeze(-1).numpy()
    X_vals = X_vals.squeeze(-1).numpy()
    t_vals = t_vals.squeeze(-1).numpy()

# loadmat
data = scipy.io.loadmat("opgaver/_static/NLS.mat")

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

u_pred_abs = np.sqrt(u_pred[:,:,0]**2+u_pred[:,:,1]**2)
plt.plot(X_vals, u_pred_abs, label='Prediction for t=0.79')
plt.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')
plt.tight_layout()
plt.savefig("schrodinger_plot_c1_c2.png")

plt.clf()

#save the model:
torch.save(model.state_dict(), "schrodinger_model_c1c2.pth")
