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

N = int(200)

# Define boundary conditions
t0 = 0.0
t_final = torch.pi/2
x_left = -5.
x_right = 5.

# Create input data
X_vals = torch.linspace(x_left, x_right, N, requires_grad=True)
t_vals = torch.linspace(t0, t_final, N, requires_grad=True)
X_train, t_train = torch.meshgrid(X_vals, t_vals, indexing="xy")
X_train = X_train.unsqueeze(-1)
t_train = t_train.unsqueeze(-1)

X_vals_ = X_vals.view(-1,1,1)
t_vals_ = t_vals.view(-1,1,1)

#print(X_vals.view(-1,1,1).shape, torch.ones_like(X_vals).view(-1,1,1).shape)


# Define functions h(x), u(x)
phi = lambda x: 2/torch.cosh(x)

hidden_units = 100
#to simulate a complex output we make it spit out two things like this [real, imaginary]
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(2,hidden_units),
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

    def forward(self, x, y):
        x_combined = torch.cat((x, y),dim=2)
        logits = self.fn_approx(x_combined)
        return logits

    
model = NeuralNetwork()
#model.apply(NeuralNetwork.init_weights)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

num_epochs_adam = 1

for epoch in range(num_epochs_adam):
    # Forward pass
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
    #loss_PDE = loss_PDE_real + loss_PDE_imag

    loss_IC = criterion(u_ic_real, phi(X_vals_))+criterion(u_ic_imag, torch.zeros_like(X_vals_))

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

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs_adam}], Loss: {loss.item():.4f}', end='\r')

# Define a closure function to reevaluate the model and loss
def closure():
    lbfgs.zero_grad()
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
    #loss_PDE = loss_PDE_real + loss_PDE_imag

    loss_IC = criterion(u_ic_real, phi(X_vals_))+criterion(u_ic_imag, torch.zeros_like(X_vals_))

    loss_boundary_2 = criterion(bound_left_real, bound_right_real)+criterion(bound_left_imag, bound_right_imag)

    loss_boundary_1 = criterion(u_left, u_right)

    # ... rest of your forward pass code ...
    loss = 1000*(loss_PDE_real + loss_PDE_imag + loss_boundary_1 + loss_boundary_2 + loss_IC)
    print(f'Loss: {loss.item():.4f}', end='\r')
    loss.backward()
    return loss

lbfgs = optim.LBFGS(model.parameters(), max_iter=1)
lbfgs.step(closure)

#save the model:
torch.save(model.state_dict(), "schrodinger_model.pth")





######################################################################
############################# Plotting ###############################
######################################################################    

# X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
# X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
# X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
# X_u_train = np.vstack([X0, X_lb, X_ub])
fig, ax = plt.subplots()
ax.axis('off')

# ####### Row 0: h(t,x) ##################    
# gs0 = gridspec.GridSpec(1, 2)
# gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
# ax = plt.subplot(gs0[:, :])

# h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
#               extent=[lb[1], ub[1], lb[0], ub[0]], 
#               origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(h, cax=cax)

# ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)

# line = np.linspace(x.min(), x.max(), 2)[:,None]
# ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
# ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
# ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    

# ax.set_xlabel('$t$')
# ax.set_ylabel('$x$')
# leg = ax.legend(frameon=False, loc = 'best')
# #   plt.setp(leg.get_texts(), color='w')
# ax.set_title('$|h(t,x)|$', fontsize = 10)

#with torch.no_grad():
    #for the 3d plot we import the matplotlib extension:
from mpl_toolkits.mplot3d import Axes3D
u_pred = lambda tt: model(X_vals.view(-1,1,1), tt*torch.ones_like(X_vals).view(-1,1,1)).squeeze(-1).detach().numpy()
X_vals_plot = X_vals.squeeze(-1).detach().numpy()
    # t_vals_plot = t_vals.squeeze(-1).detach().numpy()

# loadmat
data = scipy.io.loadmat("opgaver/_static/NLS.mat")

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

u_pred_abs = lambda tt: np.sqrt(u_pred(tt)[:,:,0]**2+u_pred(tt)[:,:,1]**2)
# plt.plot(X_vals, u_pred_abs(), label='Prediction for t=0.79')
# plt.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')
# plt.tight_layout()
# plt.savefig("schrodinger_plot.png")

# plt.clf()

####### Row 1: h(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)



ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(X_vals_plot,u_pred_abs(0.59), 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$|h(t,x)|$')    
ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-0.1,5.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(X_vals_plot,u_pred_abs(0.79), 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$|h(t,x)|$')
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-0.1,5.1])
ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(X_vals_plot,u_pred_abs(0.98), 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$|h(t,x)|$')
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-0.1,5.1])    
ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)

plt.savefig('NLS.png')