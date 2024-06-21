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
lam_range_test = torch.tensor([-0.75,0.65,1.5])
lam_steps_test = len(lam_range_test)
t_range = torch.linspace(0, 2, steps=200, requires_grad=True)  # time in the range [0, 2]
u = lambda t, lam: u0 * torch.exp(lam * t)

# Generate training data
T, LAM = torch.meshgrid(t_range, lam_range, indexing="xy")  # create a grid of (t, lambda)
# y_train = u(T, LAM)  # compute the exact solution at each (t, lambda)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(2,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1)
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4e}', end='\r')
print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4e}')
torch.save(model, "opgaver/model_h")

# Set global font sizes
plt.rcParams.update({'font.size': 16})  # Increase the base font size

# Alternatively, you can specify individual font sizes directly
title_fontsize = 18
label_fontsize = 16
legend_fontsize = 14
labels = [-0.75,0.65,1.5]

with torch.no_grad():
    fig, axs = plt.subplots(1, lam_steps_test, sharey="row", figsize=(8*lam_steps_test, 8))
    for i, lam in enumerate(lam_range_test):
        predictions = model(t_range, lam*torch.ones_like(t_range))

        axs[i].plot(t_range, u(t_range, lam), '-o', label="true value")
        axs[i].plot(t_range, predictions.numpy(), '--',linewidth=5, label="predicted value")
        axs[i].legend(fontsize=legend_fontsize)
        axs[i].set_title(f"lambda = {labels[i]}", fontsize=title_fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=label_fontsize)  # Adjust tick label size


   
    plt.tight_layout()
    plt.savefig(f"opgaver/_static/h.svg", bbox_inches='tight')
    plt.clf()

 