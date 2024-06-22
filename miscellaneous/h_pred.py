import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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
        # print("logits.shape == ", logits.shape)
        return logits

model = torch.load("opgaver/model_h")
t_range = torch.linspace(-3, 3, steps=100)
lam_steps = 10
lam_range = torch.linspace(-5, 5, steps=lam_steps)  # lambda in the range [-1, 1]
u0 = 1.0
u = lambda t, lam: u0 * torch.exp(lam * t)

with torch.no_grad():
    fig, axs = plt.subplots(1, lam_steps, figsize=(8*lam_steps, 8))
    for i, lam in enumerate(lam_range):
        predictions = model(t_range, lam*torch.ones_like(t_range))

        axs[i].plot(t_range, u(t_range, lam), '-o',label="true value")
        axs[i].plot(t_range, predictions.numpy(), '-o',label="predicted value")
        axs[i].legend()
        #axs[i].gca().set_ylim([-1,8])
        axs[i].set_title(f"lambda = {lam}")
    plt.tight_layout()
    plt.savefig(f"opgaver/_static/h_pred.svg", format="svg")    
    plt.clf()