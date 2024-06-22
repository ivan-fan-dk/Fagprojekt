import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from softadapt import *

N = 20

# Define boundary conditions
t0 = 0.0
t_final = 1.
x_left = -1.
x_right = 1.

# Create input data
X_vals = torch.linspace(x_left, x_right, N, requires_grad=True)
t_vals = torch.linspace(t0, t_final, N, requires_grad=True)
X_train, t_train = torch.meshgrid(X_vals, t_vals, indexing="xy")
X_train = X_train.unsqueeze(-1)
t_train = t_train.unsqueeze(-1)

print(X_vals.view(-1,1,1).shape, torch.ones_like(X_vals).view(-1,1,1).shape)


# Define functions h(x), u(x)
phi = lambda x: -torch.sin(torch.pi*x)
psi_minus = lambda t: torch.zeros_like(t_train)
psi_plus = lambda t: torch.zeros_like(t_train)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(2,128),
            nn.Tanh(),
            nn.Linear(128,312),
            nn.Tanh(),
            nn.Linear(312,1)
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
model.apply(NeuralNetwork.init_weights)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

#setup for softadapt:

softadapt_object = SoftAdapt(beta=0.1)


epochs_to_make_updates = 5

values_of_component_1 = []
values_of_component_2 = []
values_of_component_3 = []


# Initializing adaptive weights to all ones.
adapt_weights = torch.tensor([1,1,1])

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(X_train, t_train)
    # Compute the first derivatives
    du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    d2u_dx2 = torch.autograd.grad(du_dx, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    
    # Compute the loss
    loss_PDE = criterion(du_dt + u_prediction*du_dx, 0.01/torch.pi * d2u_dx2)
    loss_boundary = criterion(model(x_left*torch.ones_like(X_train),t_train), psi_minus(t_train)) + criterion(model(x_right*torch.ones_like(X_train),t_train), psi_plus(t_train))
    loss_IC = criterion(model(X_train, t0*torch.ones_like(t_train)), phi(X_train))
    loss = loss_PDE + loss_boundary + loss_IC

    # Backward pass and optimization
    optimizer.zero_grad()

    #softadapt weights update stuff:
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

      # Change 5: Update the loss function with the linear combination of all components.
    loss = adapt_weights[0] * loss_PDE + adapt_weights[1] * loss_boundary + adapt_weights[2]*loss_IC

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    #for the 3d plot we import the matplotlib extension:
    from mpl_toolkits.mplot3d import Axes3D
    u_pred = model(X_vals.view(-1,1,1), 0.25*torch.ones_like(X_vals).view(-1,1,1))
    u_pred =  u_pred.squeeze(-1).numpy()
    X_vals = X_vals.squeeze(-1).numpy()
    t_vals = t_vals.squeeze(-1).numpy()


plt.plot(X_vals, u_pred, label='Prediction for t=0.25')
plt.tight_layout()
plt.savefig("/workspaces/Fagprojekt/opgaver/_static/i_plot_chris_test.png")