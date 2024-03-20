import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import functions as fc

# Define the exact solution and its derivative
u0 = 1.0
t0 = 0.0
lam_steps = 5
lam_range = torch.linspace(-1, 1, steps=lam_steps, requires_grad=True)  # lambda in the range [-1, 1]
t_range = torch.linspace(0, 2, steps=10, requires_grad=True)  # time in the range [0, 2]
u = lambda t, lam: u0 * torch.exp(lam * t)

# Generate training data
T, LAM = torch.meshgrid(t_range, lam_range, indexing="xy")  # create a grid of (t, lambda)

# y_train = u(T, LAM)  # compute the exact solution at each (t, lambda)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(2,8),
            nn.Tanh(),
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, x, y):
        if x.dim() != y.dim():
            raise AssertionError(f"x and y must have the same number of dimensions, but got x.dim() == {x.dim()} and y.dim() == {y.dim()}")
        x_combined = torch.stack((x, y), dim=x.dim())
        logits = self.fn_approx(x_combined).squeeze()
        return logits

# Create an instance of the model, define loss and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Training the model
num_epochs = int(1e4)

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(T, LAM)
    
    du_dt = torch.autograd.grad(u_prediction, T, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

    # Compute the losses
    loss_ode = criterion(LAM*u_prediction, du_dt)
    loss_ic = criterion(model(t0*torch.ones_like(lam_range), lam_range), u0*torch.ones_like(lam_range))
    
    # Combine the losses
    loss = loss_ode + loss_ic

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

torch.save(model, "opgaver/model_h")

with torch.no_grad():
    fig, axs = plt.subplots(1, lam_steps, sharey="row", figsize=(8*lam_steps, 8))
    for i, lam in enumerate(lam_range):
        predictions = model(t_range, lam*torch.ones_like(t_range))

        axs[i].plot(t_range, u(t_range, lam), '-o',label="true value")
        axs[i].plot(t_range, predictions.numpy(), '-o',label="predicted value")
        axs[i].legend()
        #axs[i].gca().set_ylim([-1,8])
        axs[i].set_title(f"lambda = {lam}")
    plt.tight_layout()
    plt.savefig(f"opgaver/_static/h.png")    
    plt.clf()