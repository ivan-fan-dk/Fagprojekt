import torch
import torch.nn as nn
import numpy as np
import functions as fc
from softadapt import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 15
x_left = -1.
x_right = 1.

t0 = 0
u0 = 1.0
lam_range = torch.linspace(x_left, x_right, steps=N, requires_grad=True)  # lambda in the range [-1, 1]
t_range = torch.linspace(t0, 2, steps=N, requires_grad=True)  # time in the range [0, 2]
u = lambda t, lam: u0 * torch.exp(lam * t)

print(lam_range)
# Generate training data
grid_X, grid_Y = torch.meshgrid(t_range, lam_range, indexing="xy")

grid_X = grid_X.unsqueeze(2)
grid_Y = grid_Y.unsqueeze(2)

print(u(grid_X,grid_Y).shape)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

    def forward(self, x, y):
        x_combined = torch.cat((x, y), dim=2)
        logits = self.fn_approx(x_combined)
        return logits

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2000

 
#setup for softadapt:
# Change 1: Create a SoftAdapt object (with your desired variant)
softadapt_object = NormalizedSoftAdapt(beta=0.1)

# Change 2: Define how often SoftAdapt calculate weights for the loss components
epochs_to_make_updates = 5

# Change 3: Initialize lists to keep track of loss values over the epochs we defined above
values_of_component_1 = []
values_of_component_2 = []

# Initializing adaptive weights to all ones.
adapt_weights = torch.tensor([1,1])

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(grid_X, grid_Y)

    # Compute the first derivatives
    df_dx = torch.autograd.grad(u_prediction, grid_X, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

    #hope boundary is nice!
    # Assuming t0, lam_range, and u0 are your tensors and N is the size
    bound_cond = criterion(model(t0 * torch.ones(N).view(-1,1,1), lam_range.view(-1,1,1)), u0*torch.ones(N).view(-1,1,1))
    #DE criterion:
    DE_loss = criterion(df_dx, grid_Y * u_prediction)

    # Backward pass and optimization
    optimizer.zero_grad()

    values_of_component_1.append(DE_loss)
    values_of_component_2.append(bound_cond)

    # Compute the loss

    # Change 4: Append the loss values to the lists
    if epoch % epochs_to_make_updates == 0 and epoch != 0:
        adapt_weights = softadapt_object.get_component_weights(
        torch.tensor(values_of_component_1), 
        torch.tensor(values_of_component_2), 
        verbose=False,
        )
                                                            
      
          # Resetting the lists to start fresh (this part is optional)
        values_of_component_1 = []
        values_of_component_2 = []


      # Change 5: Update the loss function with the linear combination of all components.
    loss = adapt_weights[0] * DE_loss + 2 * adapt_weights[1] * bound_cond

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
print(f"Mean Squared Error on trained data: {loss.item():.4f}")

# Plot the results
# Compute model predictions
with torch.no_grad():
    predictions = model(torch.linspace(0,5,N).view(-1,1,1), (0.5*torch.ones(N)).view(-1,1,1))


print(predictions)
predictions_np = predictions.squeeze(-1).detach().numpy()  # Convert tensor to numpy array
t_range_np = torch.linspace(0,5,N).detach().numpy()  # Convert tensor to numpy array

plt.figure(figsize=(10, 5))

# Plotting predictions
plt.plot(t_range_np, predictions_np, label='Predictions', color='blue')

# Plotting training points
plt.scatter(t_range_np, u(torch.linspace(0,5,N), (0.5*torch.ones(N))).detach().numpy(), label='Training Points', color='red')

plt.legend(loc='upper left')
plt.savefig("opgaver/_static/h)_plot.png")
