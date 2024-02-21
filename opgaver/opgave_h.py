import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import functions as fc
# Meshgrid giver en userwarning som vi ik gider
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# Define the exact solution and its derivative
u0 = 1.0
lam_range = torch.linspace(-1, 1, steps=100)  # lambda in the range [-1, 1]
t_range = torch.linspace(0, 2, steps=100)  # time in the range [0, 2]
u = lambda t, lam: u0 * torch.exp(lam * t)

# Generate training data
T, LAM = torch.meshgrid(t_range, lam_range)  # create a grid of (t, lambda)
X_train = torch.stack((T.reshape(-1), LAM.reshape(-1)), dim=1)  # flatten and stack to get the coordinates (t, lambda)
y_train = u(X_train[:, 0], X_train[:, 1])  # compute the exact solution at each (t, lambda)
y_train = y_train.view(-1, 1)  # reshape y_train to [10000, 1]

N_f = len(X_train)  # number of points for the ODE loss
N_i = len(X_train[X_train[:, 0] == 0])  # number of points for the initial condition loss


# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 10, bias=True)  # input dimension is 2 because we have two inputs: t and lambda
        self.layer2 = nn.Linear(10, 50, bias=True)
        self.layer3 = nn.Linear(50, 30, bias=True)
        self.layer4 = nn.Linear(30, 10, bias=True)
        self.layer5 = nn.Linear(10, 1, bias=True)  # output dimension is 1 because we want to predict a single value: u

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

# Create an instance of the model, define loss and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training the model
num_epochs = 10000
def ode_loss(outputs, X_train, N_f):
    # Compute the ODE loss here
    F_N = X_train[:, 1] * outputs  # Compute F(N(t_i^f; Θ)) = λu
    N_prime = torch.autograd.grad(outputs, X_train, grad_outputs=torch.ones_like(outputs), create_graph=True)[0][:, 0]  # Compute N'(t_i^f)
    return torch.sum((F_N - N_prime)**2) / N_f

def ic_loss(outputs, labels, N_i):
    # Compute the initial condition loss here
    N_t0 = outputs[X_train[:, 0] == 0]  # Compute N(t_0^i; Θ)
    u_t0 = labels[X_train[:, 0] == 0]  # Compute u(t_0)
    return torch.sum((N_t0 - u_t0)**2) / N_i

X_train.requires_grad = True

# Training the model (tager lang tid)
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    
    # Compute the losses
    loss_ode = ode_loss(outputs, X_train, N_f)
    loss_ic = ic_loss(outputs, y_train, N_i)
    
    # Combine the losses
    loss = loss_ode + loss_ic

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    predictions = model(X_train)


fc.plot_comparison(X_train[:, 0].detach().numpy(), y_train.detach().numpy(), predictions.detach().numpy(), plot_title ="IVP", filename="h)_plot_h")


