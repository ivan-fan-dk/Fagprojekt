# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
import time



def finite_difference_allen_cahn(Nx, Ny, Nt):


    delta_x = 1  # Grid spacing in x-direction
    delta_y = 1  # Grid spacing in y-direction
    delta_t = 0.1  # Time step size
    epsilon = 0.001  # Parameter epsilon

    # Initialize grid
    u = np.random.rand(Nx, Ny)  # Initial condition
    res = np.zeros((Nx, Ny, Nt))


    # Function to update the grid for each time step
    def update_grid(u):
        u_new = u.copy()
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                laplacian = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]) / (delta_x ** 2)
                u_new[i, j] = 0.1 * (laplacian + u[i, j] - u[i, j] ** 3) * delta_t + u[i, j]

        # Update boundary points to be the same as their closest neighbors
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = u_new[-2, :]
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]

        return u_new

    # Iterate over time steps
    for t in range(Nt):
        u = update_grid(u)
        res[:, :, t] = u.copy()

        
    return  res


#%% 
def PINN(num_epochs,learning_rate, Nx, Ny, Nt):

    # Create grid points along x and y axes
    x = torch.arange(0, Nx, 1, dtype=torch.float32, requires_grad=True)
    y = torch.arange(0, Ny, 1, dtype=torch.float32, requires_grad=True)
    t = torch.arange(0, Nt, 1, dtype=torch.float32, requires_grad=True)
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

        # pred = model(X, Y, T)
        # pred_bound_left = criterion(pred[:, 0, :], torch.ones_like(pred[:, -1, :]))
        # pred_bound_right = criterion(pred[:, -1, :],  torch.ones_like(pred[:, -1, :]))
        # pred_bound_bottom = criterion(pred[0, :, :], torch.ones_like(pred[:, -1, :]))
        # pred_bound_top = criterion(pred[-1, :, :],  torch.ones_like(pred[:, -1, :]))

        # # Summing up all boundary losses
        # loss_boundary = pred_bound_left + pred_bound_right + pred_bound_bottom + pred_bound_top

        # Total loss
        loss = loss_PDE# + loss_boundary

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return loss.item()
    # with torch.no_grad():
    #     #for the 3d plot we import the matplotlib extension:
    #     from mpl_toolkits.mplot3d import Axes3D
    #     u_pred = model(X, Y, T)
    #     u_pred =  u_pred.numpy()
    #     X = X.numpy()
    #     Y = Y.numpy()
    #     T = T.numpy()
    #     print(u_pred.shape)
    #     u_pred = u_pred[:,:,9]
       

    #     # Create a heatmap
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(u_pred, cmap='viridis')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Heatmap of u_pred')
    #     plt.show()





#PINN(500,0.001)


def split_grid(X, n):
    """
    Split a grid X into n equally large squares.

    Parameters:
        X (torch.Tensor): Grid to be split.
        n (int): Number of squares to split into (must be a perfect square).

    Returns:
        List of subgrids.
    """
    # Calculate the size of each square
    size_x = X.shape[0] // int(n ** 0.5)
    size_y = X.shape[1] // int(n ** 0.5)

    # Initialize list to store subgrids
    subgrids = []

    # Iterate over the range of n for x dimension
    for i in range(int(n ** 0.5)):
        # Calculate boundaries for x dimension
        x_start = i * size_x
        x_end = (i + 1) * size_x

        # Iterate over the range of n for y dimension
        for j in range(int(n ** 0.5)):
            # Calculate boundaries for y dimension
            y_start = j * size_y
            y_end = (j + 1) * size_y

            # Extract the subgrid
            subgrid = X[x_start:x_end, y_start:y_end]

            # Append the subgrid to the list
            subgrids.append(subgrid)

    return subgrids




def merge_grids(subgrids):
    """
    Merge a list of subgrids into a single grid.

    Parameters:
        subgrids (List[torch.Tensor]): List of subgrids to be merged.

    Returns:
        torch.Tensor: Merged grid.
    """
    # Get the dimensions of the first subgrid
    rows = subgrids[0].shape[0]
    cols = subgrids[0].shape[1]  # Corrected

    # Calculate the total number of subgrids
    num_subgrids = len(subgrids)

    # Calculate the size of the merged grid
    merged_rows = rows * 2
    merged_cols = cols * 2

    # Initialize the merged grid
    merged_grid = torch.zeros((merged_rows, merged_cols))

    # Iterate over the subgrids and place them in the merged grid
    for i, subgrid in enumerate(subgrids):
        # Calculate the position for the current subgrid
        x_start = (i // 2) * rows
        x_end = x_start + rows
        y_start = (i % 2) * cols
        y_end = y_start + cols

        # Convert NumPy array to PyTorch tensor
        subgrid = torch.tensor(subgrid)

        # Place the subgrid in the merged grid
        merged_grid[x_start:x_end, y_start:y_end] = subgrid

    return merged_grid



def XPINN(num_epochs, learning_rate, number_of_subdomains, Nx, Ny, Nt):
   

    simulated = torch.tensor(finite_difference_allen_cahn(Nx, Ny, Nt),requires_grad=True, dtype=torch.float32).unsqueeze(-1)
    print(simulated.shape)

    x = torch.arange(0, Nx, 1, dtype=torch.float32, requires_grad=True)
    y = torch.arange(0, Ny, 1, dtype=torch.float32, requires_grad=True)
    t = torch.arange(0, Nt, 1, dtype=torch.float32, requires_grad=True)

    # Create meshgrid from X and Y
    X, Y, T = torch.meshgrid(x, y, t, indexing = "xy")

    squares_X = split_grid(X, number_of_subdomains)
    squares_Y = split_grid(Y, number_of_subdomains)
    squares_T = split_grid(T, number_of_subdomains)
    square_sol = split_grid(simulated, number_of_subdomains)

    print(squares_X[0].shape, square_sol[0][:,:,:,0].shape)

    #subdomain grid
    res = []
    loss_list = []
    for i in range(len(squares_X)):
        class NNU(nn.Module):

            def __init__(self):
                super().__init__()
                
                self.fn_approx = nn.Sequential(
                    nn.Linear(3,20),
                    nn.Tanh(),
                    nn.Linear(20,20),
                    nn.Tanh(),
                    nn.Linear(20,20),
                    nn.Tanh(),
                    nn.Linear(20,20),
                    nn.Tanh(),
                    nn.Linear(20,20),
                    nn.Tanh(),
                    nn.Linear(20,1)
                )

            def forward(self, x, y, t):
                x_combined = torch.stack((x, y, t), dim=x.dim())
                logits = self.fn_approx(x_combined).squeeze()
                return logits
        
        class NNF(nn.Module):

            def __init__(self):
                super().__init__()
                
                self.fn_approx = nn.Sequential(
                    nn.Linear(3,20),
                    nn.Tanh(),
                    nn.Linear(20,20),
                    nn.Tanh(),
                    nn.Linear(20,20),
                    nn.Tanh(),
                    nn.Linear(20,1)
                )

            def forward(self, x, y, t):
                x_combined = torch.stack((x, y, t), dim=x.dim())
                logits = self.fn_approx(x_combined).squeeze()
                return logits

        modelU = NNU()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(modelU.parameters(), lr=learning_rate)


        for epoch in range(num_epochs):
            X = squares_X[i]
            Y = squares_Y[i]
            T = squares_T[i]
            # First we predict U 
            u_prediction = modelU(X, Y, T)
            U_loss = criterion(u_prediction, square_sol[i][:,:,:,0])

            # Now we want to predict th residual
            # Compute the first derivatives
            du_dx = torch.autograd.grad(u_prediction, X, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
            du_dy = torch.autograd.grad(u_prediction, Y, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
            du_dt = torch.autograd.grad(u_prediction, T, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

            du_dxx = torch.autograd.grad(du_dx, X, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
            du_dyy = torch.autograd.grad(du_dy, Y, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
            du_dtt = torch.autograd.grad(du_dt, T, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

            # Compute the loss
            loss_PDE = criterion(du_dt, - du_dxx - du_dyy - du_dtt + u_prediction - u_prediction**3)

            # Total loss
            loss = loss_PDE + U_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        loss_list.append(loss.item())
        with torch.no_grad():

            u_pred = modelU(X, Y, T)
            u_pred =  u_pred.numpy()

            res.append(u_pred)
        
    loss_list = np.array(loss_list)
    mean_loss = np.mean(loss_list)

    print(f"This took: {training_time} seconds to train")

    return res, mean_loss


def testing_epochs(learning_rate, subdomains, Nx, Ny, Nt):
    import time 
    t1 = time.time()
    t2 = time.time()
    training_time = t2-t1

    epochs_to_test = [10, 50, 100, 200, 500, 1000]

    mse_xpinn = []
    mse_pinn  = [] 

    for t_epoch in epochs_to_test:
        mse_xpinn = XPINN(t_epoch, learning_rate, subdomains, Nx, Ny, Nt)
        mse_pinn  = PINN(t_epoch,learning_rate, Nx, Ny, Nt)



XPINN(100, 0.001, 4, 20, 20, 20)