import torch
import torch.nn as nn
import numpy as np
import functions as fc
from softadapt import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 100
# Define your functions u(x) and f(x)
# \Delta u = f, u(-5) = g_left, u(5) = g_right
u = lambda x,y: x**2+y**2
f = lambda x,y: 4. * torch.ones((x.shape[0],x.shape[1],1))


# Define boundary conditions
x_left = -1.
x_right = 1.



# Create input data
X_train = torch.linspace(x_left, x_right, N, requires_grad=True)
Y_train = torch.linspace(x_left, x_right, N, requires_grad=True)


grid_X, grid_Y = torch.meshgrid(X_train, Y_train, indexing="xy")

grid_X = grid_X.unsqueeze(2)
grid_Y = grid_Y.unsqueeze(2)

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

num_epochs = 500

 
#setup for softadapt:
# Change 1: Create a SoftAdapt object (with your desired variant)
softadapt_object = NormalizedSoftAdapt(beta=0.1)

# Change 2: Define how often SoftAdapt calculate weights for the loss components
epochs_to_make_updates = 5

# Change 3: Initialize lists to keep track of loss values over the epochs we defined above
values_of_component_1 = []
values_of_component_2 = []
values_of_component_3 = []
values_of_component_4 = []
values_of_component_5 = []

# Initializing adaptive weights to all ones.
adapt_weights = torch.tensor([1,1,1,1,1])

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(grid_X, grid_Y)

    # Compute the first derivatives
    df_dx = torch.autograd.grad(u_prediction,grid_X, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    df_dy = torch.autograd.grad(u_prediction,grid_Y, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

    # Compute the second derivatives
    d2f_dx2 = torch.autograd.grad(df_dx, grid_X, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]
    d2f_dy2 = torch.autograd.grad(df_dy, grid_Y, create_graph=True, grad_outputs=torch.ones_like(df_dy))[0]




    #seperate Ohm and dOhm:
    #hope boundary is nice!
    #function is u=x^2+y^2 and the boundary is a centered square from x=-1 to x=1 and so on for y, please fill out the rest of the sides:
    x_side1 = criterion(model(grid_X, grid_Y)[:, -1, :], (torch.ones(N) + X_train**2).view(-1,1))
    x_side2 = criterion(model(grid_X, grid_Y)[:, 0, :],  (torch.ones(N) + X_train**2).view(-1,1))
    y_side1 = criterion(model(grid_X, grid_Y)[-1, :, :], (torch.ones(N) + Y_train**2).view(-1,1))
    y_side2 = criterion(model(grid_X, grid_Y)[0, :, :],  (torch.ones(N) + Y_train**2).view(-1,1))

    #DE criterion:
    DE_loss = criterion(d2f_dx2[1:-1,1:-1]+d2f_dy2[1:-1,1:-1], f(grid_X[1:-1,1:-1],grid_Y[1:-1,1:-1]))

    # Backward pass and optimization
    optimizer.zero_grad()

    values_of_component_1.append(x_side1)
    values_of_component_2.append(x_side2)
    values_of_component_3.append(y_side1)
    values_of_component_4.append(y_side2)
    values_of_component_5.append(DE_loss)

    # Compute the loss
    #loss = criterion(d2f_dx2[1:-1,1:-1]+d2f_dy2[1:-1,1:-1], f(grid_X[1:-1,1:-1],grid_Y[1:-1,1:-1]))+x_side1+x_side2+y_side1+y_side2

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
    loss = adapt_weights[0] * x_side1 + adapt_weights[1]*x_side2 + adapt_weights[2]*y_side1 + adapt_weights[3]*y_side2 + adapt_weights[4]*DE_loss

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
print(f"Mean Squared Error on trained data: {loss.item():.4f}")

# Create a grid of (x, y) coordinates
#X_test = torch.linspace(x_left, x_right, N, requires_grad=True)
#Y_test = torch.linspace(x_left, x_right, N, requires_grad=True)
#grid_X_test, grid_Y_test = torch.meshgrid(X_test, Y_test)

# Compute model predictions
with torch.no_grad():
    predictions = model(grid_X, grid_Y)

# Convert tensors to numpy arrays for plotting
grid_X = grid_X.squeeze(2).detach().numpy()
grid_Y = grid_Y.squeeze(2).detach().numpy()
predictions = predictions.squeeze(2).detach().numpy()

print(predictions.shape, grid_X.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(grid_X, grid_Y, predictions, alpha=0.3, rstride=100, cstride=100,color="blue")

# Also plot the training points
ax.plot_surface(grid_X, grid_Y, u(grid_X,grid_Y), alpha=0.3, rstride=100, cstride=100, color="red")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Predictions')

plt.savefig("opgaver/_static/erik2d_plot.png")
