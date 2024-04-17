# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


import numpy as np

# Parameters
Nx = 40  # Number of grid points in x-direction
Ny = 40  # Number of grid points in y-direction
Nt = 1  # Number of time steps
delta_x = 1  # Grid spacing in x-direction
delta_y = 1  # Grid spacing in y-direction
delta_t = 0.1  # Time step size
epsilon = 0.001  # Parameter epsilon

# Initialize grid
u = np.random.rand(Nx, Ny)  # Initial condition
res = np.zeros((Nx, Ny, Nt))

# # Function to update the grid for each time step
# def update_grid(u):
#     u_new = u.copy()
#     for i in range(1, Nx - 1):
#         for j in range(1, Ny - 1):
#             laplacian = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]) / (delta_x ** 2)
#             u_new[i, j] = 0.1 * (laplacian + u[i, j] - u[i, j] ** 3) * delta_t + u[i, j]

#     # Update boundary points to be the same as their closest neighbors
#     u_new[0, :] = u_new[1, :]
#     u_new[-1, :] = u_new[-2, :]
#     u_new[:, 0] = u_new[:, 1]
#     u_new[:, -1] = u_new[:, -2]

#     return u_new

# # Iterate over time steps
# for t in range(Nt):
#     u = update_grid(u)
#     res[:, :, t] = u.copy()

    
# # Assuming u is your grid data

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 3 subplots side-by-side

# # Plot the final state of the grid in the first subplot
# times = [0, 50, 99]
# for i in range(3):
#     axs[i].imshow(res[:,:,times[i]])
#     axs[i].set_title('Allen-Cahn Equation')
#     axs[i].set_xlabel('x')
#     axs[i].set_ylabel('y')
#     axs[i].set_aspect('equal')  # Ensure aspect ratio is equal


# # You can plot additional data in the other subplots if needed
# plt.show()


# %%
# import numpy as np

# def divide_to_subdomains(image):
#     """
#     Divide the input image into four equally sized subparts.

#     Parameters:
#         image (numpy.ndarray): Input image.

#     Returns:
#         subimages (list): List containing four equally sized subparts of the image.
#     """

#     # Get the dimensions of the image
#     height, width = domain.shape[:2]

#     # Calculate the midpoint of the domain dimensions
#     mid_height = height // 2
#     mid_width = width // 2

#     # Divide the domain into four subparts
#     subdomain1 = domain[:mid_height, :mid_width]
#     subdomain2 = domain[:mid_height, mid_width:]
#     subdomain3 = domain[mid_height:, :mid_width]
#     subdomain4 = domain[mid_height:, mid_width:]

#     # Return the four subparts
#     return [subdomain1, subdomain2, subdomain3, subdomain4]

# # Example usage:
# subimages = divide_to_subdomains(image)

# test = divide_to_subdomains(res[:,:,99])

# test[1].shape


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Define boundary conditions
Nx = 50  # Number of grid points in x-direction
Ny = 50 # Number of grid points in y-direction
Nt = 50  # Number of time steps

# Create grid points along x and y axes
x = torch.arange(0, Nx, 0.5, dtype=torch.float32, requires_grad=True)
y = torch.arange(0, Ny, 0.5, dtype=torch.float32, requires_grad=True)
t = torch.arange(0, Nt, 1, dtype=torch.float32, requires_grad=True)

def PINN(x,y,t,num_epochs,learning_rate):
    # Create meshgrid from X and Y
    X, Y, T = torch.meshgrid(x, y, t, indexing = "xy")

    class NeuralNetwork(nn.Module):

        def __init__(self):
            super().__init__()
            
            self.fn_approx = nn.Sequential(
                nn.Linear(3,81),
                nn.Tanh(),
                nn.Linear(81,27),
                nn.Tanh(),
                nn.Linear(27,1)
            )

        def forward(self, x, y, t):
            x_combined = torch.stack((x, y, t), dim=x.dim())
            logits = self.fn_approx(x_combined).squeeze()
            return logits

    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    model(X, Y, T)

    for epoch in range(num_epochs):
        # Forward pass
        u_prediction = model(X, Y, T)
        # Compute the first derivatives
        du_dx = torch.autograd.grad(u_prediction, X, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
        du_dy = torch.autograd.grad(u_prediction, Y, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
        du_dt = torch.autograd.grad(u_prediction, T, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

        du_dxx = torch.autograd.grad(du_dx, X, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
        du_dyy = torch.autograd.grad(du_dy, Y, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
        du_dtt = torch.autograd.grad(du_dt, T, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

        # Compute the loss
        loss_PDE = criterion(du_dt, - du_dxx - du_dyy - du_dtt + u_prediction - u_prediction**3)

        # Boundary conditions values

        pred = model(X, Y, T)
        pred_bound_left = criterion(pred[:, 0, :], torch.ones_like(pred[:, -1, :]))
        pred_bound_right = criterion(pred[:, -1, :],  torch.ones_like(pred[:, -1, :]))
        pred_bound_bottom = criterion(pred[0, :, :], torch.ones_like(pred[:, -1, :]))
        pred_bound_top = criterion(pred[-1, :, :],  torch.ones_like(pred[:, -1, :]))

        # Summing up all boundary losses
        loss_boundary = pred_bound_left + pred_bound_right + pred_bound_bottom + pred_bound_top

        # Total loss
        loss = loss_PDE + loss_boundary

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        #for the 3d plot we import the matplotlib extension:
        from mpl_toolkits.mplot3d import Axes3D
        u_pred = model(X, Y, T)
        u_pred =  u_pred.numpy()
        X = X.numpy()
        Y = Y.numpy()
        T = T.numpy()

        u_pred = u_pred[:,:,40]

        # Create a heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(u_pred, cmap='viridis')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Heatmap of u_pred')
        plt.show()





PINN(x,y,t,500,0.001)