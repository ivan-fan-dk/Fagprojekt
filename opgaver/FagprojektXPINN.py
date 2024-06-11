
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
import time 

def PINN(plot_it:bool, num_epochs):
    t_start = time.time()
    # Define functions h(x), u(x)
    c = 1
    h = lambda x: torch.sin(x)
    u = lambda x, t: h(x-c*t)
    g = lambda t: h(-c*t)

    # Define boundary conditions
    t0 = 0.0
    t_final = 10.0
    x_left = 0.0
    x_right = 10.0

    # Create input data
    N = 20
    X_vals = torch.linspace(x_left, x_right, N, requires_grad=True)
    t_vals = torch.linspace(t0, t_final, N, requires_grad=True)
    X_train, t_train = torch.meshgrid(X_vals, t_vals, indexing="xy")
    X_train = X_train.unsqueeze(-1)
    t_train = t_train.unsqueeze(-1)

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
            x_combined = torch.cat((x, y),dim=2)
            logits = self.fn_approx(x_combined)
            return logits

        
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model(X_train, t_train)

    for epoch in range(num_epochs):
        # Forward pass
        u_prediction = model(X_train, t_train)
        # Compute the first derivatives
        du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
        du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

        # Compute the loss
        loss_PDE = criterion(-c*du_dx, du_dt)
        loss_boundary = criterion(model(torch.zeros_like(X_train), t_train), g(t_train))
        loss_IC = criterion(model(X_train, torch.zeros_like(t_train)), h(X_train))
        loss = loss_PDE  + loss_IC + loss_boundary

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
        u_pred = model(X_train, t_train)
        u_pred =  u_pred.squeeze(-1).numpy()
        X_train = X_train.squeeze(-1).numpy()
        t_train = t_train.squeeze(-1).numpy()
        if plot_it:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X_train, t_train, u_pred, cmap='viridis')
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('u(x,t)')
            plt.show()
    
    t_end = time.time()

    time_run = t_end - t_start
    return u_pred, time_run, loss.item()


def XPINN():
    # Define functions h(x), u(x)
    c = 1
    h = lambda x: torch.sin(x)
    u = lambda x, t: h(x - c * t)
    g = lambda t: h(-c * t)


    # Dividing the area to make four subdomains
    X_vals_1 = torch.linspace(0, 5, 10, requires_grad=True)
    X_vals_2 = torch.linspace(5, 10, 10, requires_grad=True)

    t_vals_1 = torch.linspace(0, 5, 10, requires_grad=True)
    t_vals_2 = torch.linspace(5, 10, 10, requires_grad=True)

    # Making subgrids for each subdomain
    X_train_1, t_train_1 = torch.meshgrid(X_vals_1, t_vals_1, indexing="xy")
    X_train_1_2, t_train_2 = torch.meshgrid(X_vals_1, t_vals_2, indexing="xy")
    X_train_2, t_train_1_2 = torch.meshgrid(X_vals_2, t_vals_1, indexing="xy")
    X_train_2_2, t_train_2_2 = torch.meshgrid(X_vals_2, t_vals_2, indexing="xy")

    X_train_1 = X_train_1.unsqueeze(-1)
    X_train_2 = X_train_2.unsqueeze(-1)
    t_train_1 = t_train_1.unsqueeze(-1)
    t_train_2 = t_train_2.unsqueeze(-1)
    X_train_1_2 = X_train_1_2.unsqueeze(-1)
    X_train_2_2 = X_train_2_2.unsqueeze(-1)
    t_train_1_2 = t_train_1_2.unsqueeze(-1)
    t_train_2_2 = t_train_2_2.unsqueeze(-1)

    # Simulated true values for the entire domain (for demonstration purposes)
    true_values = np.random.rand(20, 20)  # Replace this with actual true values

    domain = [
        [X_train_1, t_train_1],
        [X_train_1_2, t_train_2],
        [X_train_2, t_train_1_2],
        [X_train_2_2, t_train_2_2]
    ]

    # Initialize arrays to save the merged results
    all_X_train = []
    all_t_train = []
    all_u_pred = []

    # Define the Neural Network model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fn_approx = nn.Sequential(
                nn.Linear(2, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )

        def forward(self, x, y):
            x_combined = torch.cat((x, y), dim=2)
            logits = self.fn_approx(x_combined)
            return logits

    # Training the model for each subdomain
    models = []  # to store models for each subdomain
    for i, subdomain in enumerate(domain):
        X_train = subdomain[0]
        t_train = subdomain[1]

        model = NeuralNetwork()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 1000

        for epoch in range(num_epochs):
            # Forward pass
            u_prediction = model(X_train, t_train)
            # Compute the first derivatives
            du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
            du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

            # Compute the loss
            loss_PDE = criterion(-c * du_dx, du_dt)
            loss_boundary = criterion(model(torch.zeros_like(X_train), t_train), g(t_train))
            loss_IC = criterion(model(X_train, torch.zeros_like(t_train)), h(X_train))

            # Interface loss
            if i > 0:  # For subdomains other than the first one
                X_interface = X_train[:, 0:1, :]
                t_interface = t_train[:, 0:1, :]
                u_pred_interface = model(X_interface, t_interface)

                prev_model = models[i-1]
                with torch.no_grad():
                    u_prev_interface = prev_model(X_interface, t_interface)
                
                avg_pred = (u_pred_interface + u_prev_interface) / 2
                true_interface = torch.tensor(true_values[X_interface[:, 0, 0].long(), t_interface[:, 0, 0].long()], dtype=torch.float32).unsqueeze(-1)

                loss_interface = criterion(avg_pred, true_interface)
                loss = loss_PDE + loss_boundary + loss_IC + loss_interface
            else:
                loss = loss_PDE + loss_boundary + loss_IC

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

        with torch.no_grad():
            u_pred = model(X_train, t_train)
            u_pred = u_pred.squeeze(-1).numpy()
            X_train = X_train.squeeze(-1).numpy()
            t_train = t_train.squeeze(-1).numpy()

            all_X_train.append(X_train)
            all_t_train.append(t_train)
            all_u_pred.append(u_pred)

        models.append(model)

    # Merging the results
    X_train = np.concatenate(all_X_train, axis=0)
    t_train = np.concatenate(all_t_train, axis=0)
    u_pred = np.concatenate(all_u_pred, axis=0)

    # Plotting the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_train, t_train, u_pred, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    plt.show()

def XPINN(true_values):
    # Define functions h(x), u(x)
    c = 1
    h = lambda x: torch.sin(x)
    u = lambda x, t: h(x - c * t)
    g = lambda t: h(-c * t)

    # Create input data
    N = 20
    # Dividing the area to make four subdomains
    X_vals_1 = torch.linspace(0, 5, 10, requires_grad=True)
    X_vals_2 = torch.linspace(5, 10, 10, requires_grad=True)

    t_vals_1 = torch.linspace(0, 5, 10, requires_grad=True)
    t_vals_2 = torch.linspace(5, 10, 10, requires_grad=True)

    # Making subgrids for each subdomain
    X_train_1, t_train_1 = torch.meshgrid(X_vals_1, t_vals_1, indexing="xy")
    X_train_1_2, t_train_2 = torch.meshgrid(X_vals_1, t_vals_2, indexing="xy")
    X_train_2, t_train_1_2 = torch.meshgrid(X_vals_2, t_vals_1, indexing="xy")
    X_train_2_2, t_train_2_2 = torch.meshgrid(X_vals_2, t_vals_2, indexing="xy")

    X_train_1 = X_train_1.unsqueeze(-1)
    X_train_2 = X_train_2.unsqueeze(-1)
    t_train_1 = t_train_1.unsqueeze(-1)
    t_train_2 = t_train_2.unsqueeze(-1)

    X_train_1_2 = X_train_1_2.unsqueeze(-1)
    X_train_2_2 = X_train_2_2.unsqueeze(-1)
    t_train_1_2 = t_train_1_2.unsqueeze(-1)
    t_train_2_2 = t_train_2_2.unsqueeze(-1)


    domain = [
        [X_train_1, t_train_1],
        [X_train_2, t_train_1_2],
        [X_train_1_2, t_train_2],
        [X_train_2_2, t_train_2_2]
    ]

    # Initialize arrays to save the merged results
    all_X_train = []
    all_t_train = []
    all_u_pred = []

    # Define the Neural Network model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fn_approx = nn.Sequential(
                nn.Linear(2, 20),
                nn.Tanh(),
                nn.Linear(20, 20),
                nn.Tanh(),
                nn.Linear(20, 1)
            )

        def forward(self, x, y):
            x_combined = torch.cat((x, y), dim=2)
            logits = self.fn_approx(x_combined)
            return logits

    # Create a list of models, optimizers, and losses for each subdomain
    models = [NeuralNetwork() for _ in domain]
    optimizers = [optim.Adam(model.parameters(), lr=0.0001) for model in models]
    criterion = nn.MSELoss()

    num_epochs = 2000

    for epoch in range(num_epochs):
        total_loss = 0

        for i, (model, optimizer, subdomain) in enumerate(zip(models, optimizers, domain)):
            X_train, t_train = subdomain

            # Forward pass
            u_prediction = model(X_train, t_train)
            # Compute the first derivatives
            du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
            du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

            #compute the loss
            loss_PDE = criterion(-c * du_dx, du_dt)
            loss_IC = criterion(model(X_train, torch.zeros_like(t_train)), h(X_train))


            X_interface = torch.tensor([5.0], requires_grad=True).expand_as(t_train)
            t_interface = t_train
            
            u_pred_interface = model(X_interface, t_interface)
            #Loss along X-interfaces
            if i == 0:
                next_model = models[1]  # Interface between subdomain 1 and 2
            elif i == 1:
                next_model = models[0]  
            elif i == 2:
                next_model = models[3] 
            elif i == 3:
                next_model = models[2]  
            u_next_interface = next_model(X_interface, t_interface)

            avg_pred = (u_pred_interface + u_next_interface) / 2
            t_indices = t_interface.squeeze().long()
            true_interface = torch.tensor(true_values[10,t_indices], dtype=torch.float32).unsqueeze(-1)

            #Make sure the boundary values match at the interface
            loss_interface_x = criterion(u_pred_interface, true_interface) + criterion(torch.abs(u_pred_interface - u_next_interface),torch.zeros_like(u_pred_interface))

            X_interface = X_train
            t_interface = torch.tensor([5.0], requires_grad=True).expand_as(X_train)

            u_pred_interface = model(X_interface, t_interface)
            #Loss along t-interface
            if i == 0:
                next_model = models[2] 
            elif i == 1:
                next_model = models[3]  
            elif i == 2:
                next_model = models[0] 
            elif i == 3:
                next_model = models[1]  
        

            u_next_interface = next_model(X_interface, t_interface)

            avg_pred = (u_pred_interface + u_next_interface) / 2
            X_indices = X_interface.squeeze().long()
            true_interface = torch.tensor(true_values[X_indices,10], dtype=torch.float32).unsqueeze(-1)
            # Ensure the boundary values match at the interface
            loss_interface_t = criterion(u_pred_interface, true_interface) + criterion(torch.abs(u_pred_interface - u_next_interface),torch.zeros_like(u_pred_interface))

            loss = loss_PDE  + loss_IC + loss_interface_t + loss_interface_x

            total_loss += loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item():.4f}', end='\r')

    for model, subdomain in zip(models, domain):
        X_train, t_train = subdomain
        with torch.no_grad():
            u_pred = model(X_train, t_train)
            u_pred = u_pred.squeeze(-1).numpy()
            X_train = X_train.squeeze(-1).numpy()
            t_train = t_train.squeeze(-1).numpy()

            all_X_train.append(X_train)
            all_t_train.append(t_train)
            all_u_pred.append(u_pred)

    # Merging the results
    X_train = np.concatenate((np.concatenate((all_X_train[0],all_X_train[1]),axis=1),np.concatenate((all_X_train[2],all_X_train[3]),axis=1)), axis=0)
    t_train = np.concatenate((np.concatenate((all_t_train[0],all_t_train[1]),axis=1),np.concatenate((all_t_train[2],all_t_train[3]),axis=1)), axis=0)
    u_pred = np.concatenate((np.concatenate((all_u_pred[0],all_u_pred[1]),axis=1),np.concatenate((all_u_pred[2],all_u_pred[3]),axis=1)), axis=0)
    print(X_train.shape,t_train.shape,u_pred.shape,true_values.shape)


    # Plotting the results
    fig = plt.figure(figsize=(12, 6))   
    # First subplot with u_pred
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_train, t_train, u_pred, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u_pred(x,t)')
    ax1.set_title('Prediction')

    # Second subplot with true_values
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_train, t_train, true_values, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('true_values(x,t)')
    ax2.set_title('True Values')

    plt.tight_layout()
    plt.show()




def XPINN(true_values,plot_it:bool, num_epochs):
    t_start = time.time()
    # Define functions h(x), u(x)
    c = 1
    h = lambda x: torch.sin(x)
    u = lambda x, t: h(x - c * t)
    g = lambda t: h(-c * t)

    # Create input data
    N = 20
    # Dividing the area to make four subdomains
    X_vals_1 = torch.linspace(0, 5, 10, requires_grad=True)
    X_vals_2 = torch.linspace(5, 10, 10, requires_grad=True)

    t_vals_1 = torch.linspace(0, 5, 10, requires_grad=True)
    t_vals_2 = torch.linspace(5, 10, 10, requires_grad=True)

    # Making subgrids for each subdomain
    X_train_1, t_train_1 = torch.meshgrid(X_vals_1, t_vals_1, indexing="xy")
    X_train_1_2, t_train_2 = torch.meshgrid(X_vals_1, t_vals_2, indexing="xy")
    X_train_2, t_train_1_2 = torch.meshgrid(X_vals_2, t_vals_1, indexing="xy")
    X_train_2_2, t_train_2_2 = torch.meshgrid(X_vals_2, t_vals_2, indexing="xy")

    X_train_1 = X_train_1.unsqueeze(-1)
    X_train_2 = X_train_2.unsqueeze(-1)
    t_train_1 = t_train_1.unsqueeze(-1)
    t_train_2 = t_train_2.unsqueeze(-1)

    X_train_1_2 = X_train_1_2.unsqueeze(-1)
    X_train_2_2 = X_train_2_2.unsqueeze(-1)
    t_train_1_2 = t_train_1_2.unsqueeze(-1)
    t_train_2_2 = t_train_2_2.unsqueeze(-1)


    domain = [
        [X_train_1, t_train_1],
        [X_train_2, t_train_1_2],
        [X_train_1_2, t_train_2],
        [X_train_2_2, t_train_2_2]
    ]

    # Initialize arrays to save the merged results
    all_X_train = []
    all_t_train = []
    all_u_pred = []

    # Define the Neural Network model
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

    # Create a list of models, optimizers, and losses for each subdomain
    models = [NeuralNetwork() for _ in domain]
    optimizers = [optim.Adam(model.parameters(), lr=0.002) for model in models]
    criterion = nn.MSELoss()


    for epoch in range(num_epochs):
        total_loss = 0

        for i, (model, optimizer, subdomain) in enumerate(zip(models, optimizers, domain)):
            X_train, t_train = subdomain

            # Forward pass
            u_prediction = model(X_train, t_train)
            # Compute the first derivatives
            du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
            du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

            #compute the loss
            loss_PDE = criterion(-c * du_dx, du_dt)
            loss_IC = criterion(model(X_train, torch.zeros_like(t_train)), h(X_train))
            loss_boundary = criterion(model(torch.zeros_like(X_train), t_train), g(t_train))
            loss_boundary_end = criterion(model(10*torch.ones_like(X_train), t_train), g(t_train))

            X_interface = torch.tensor([5.0], requires_grad=True).expand_as(t_train)
            t_interface = t_train

            u_pred_interface = model(X_interface, t_interface)

            #Loss along X-interfaces
            if i == 0:
                next_model = models[1]  # Interface between subdomain 1 and 2
            elif i == 1:
                next_model = models[0]  
            elif i == 2:
                next_model = models[3] 
            elif i == 3:
                next_model = models[2]  
            u_next_interface = next_model(X_interface, t_interface)

            avg_pred = (u_pred_interface + u_next_interface) / 2
            t_indices = t_interface.squeeze().long()
            true_interface = torch.tensor(true_values[10,t_indices], dtype=torch.float32).unsqueeze(-1)

            #Make sure the boundary values match at the interface
            loss_interface_x =  criterion(torch.abs(u_pred_interface - u_next_interface),torch.zeros_like(u_pred_interface))# + criterion(avg_pred, true_interface)

            X_interface = X_train
            t_interface = torch.tensor([5.0], requires_grad=True).expand_as(X_train)

            u_pred_interface = model(X_interface, t_interface)
            #Loss along t-interface
            if i == 0:
                next_model = models[2] 
            elif i == 1:
                next_model = models[3]  
            elif i == 2:
                next_model = models[0] 
            elif i == 3:
                next_model = models[1]  
            

            u_next_interface = next_model(X_interface, t_interface)

            avg_pred = (u_pred_interface + u_next_interface) / 2
            X_indices = X_interface.squeeze().long()
            true_interface = torch.tensor(true_values[X_indices,10], dtype=torch.float32).unsqueeze(-1)

            # Ensure the boundary values match at the interface
            loss_interface_t = criterion(torch.abs(u_pred_interface - u_next_interface),torch.zeros_like(u_pred_interface)) + criterion(avg_pred, true_interface)
            
            loss = loss_PDE  + loss_IC  + loss_interface_t + loss_interface_x 

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {loss.item():.4f}', end='\r')
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    for model, subdomain in zip(models, domain):
        X_train, t_train = subdomain
        with torch.no_grad():
            u_pred = model(X_train, t_train)
            u_pred = u_pred.squeeze(-1).numpy()
            X_train = X_train.squeeze(-1).numpy()
            t_train = t_train.squeeze(-1).numpy()

            all_X_train.append(X_train)
            all_t_train.append(t_train)
            all_u_pred.append(u_pred)

    # Merging the results
    X_train = np.concatenate((np.concatenate((all_X_train[0],all_X_train[1]),axis=1),np.concatenate((all_X_train[2],all_X_train[3]),axis=1)), axis=0)
    t_train = np.concatenate((np.concatenate((all_t_train[0],all_t_train[1]),axis=1),np.concatenate((all_t_train[2],all_t_train[3]),axis=1)), axis=0)
    u_pred = np.concatenate((np.concatenate((all_u_pred[0],all_u_pred[1]),axis=1),np.concatenate((all_u_pred[2],all_u_pred[3]),axis=1)), axis=0)


    if plot_it:
        #Analytical solution
        # Parameters
        c = 1  # Wave speed
        x = np.linspace(0, 10, 20)  # Spatial domain
        t = np.linspace(0, 10, 20)  # Time domain
        X, T = np.meshgrid(x, t)

        # Function definition
        def u(x, t, c):
            return np.sin(x - c * t)

        U = u(X, T, c)


        fig = plt.figure(figsize=(18, 6))  # Increased figure width for better layout

        # First subplot with u_pred
        ax1 = fig.add_subplot(131, projection='3d')  # Changed to 131 for 1 row, 3 columns, 1st subplot
        ax1.plot_surface(X_train, t_train, u_pred, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_zlabel('u(x,t)')
        ax1.set_title('XPINN')

        # Second subplot with true_values for PINN
        ax2 = fig.add_subplot(132, projection='3d')  # Changed to 132 for 1 row, 3 columns, 2nd subplot
        ax2.plot_surface(X_train, t_train, true_values, cmap='viridis')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('u(x,t)')
        ax2.set_title('PINN')

        # Third subplot with true_values for True values
        ax3 = fig.add_subplot(133, projection='3d')  # Changed to 133 for 1 row, 3 columns, 3rd subplot
        ax3.plot_surface(X, T, U, cmap='viridis')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('u(x,t)')
        ax3.set_title('True values')

        plt.tight_layout()
        plt.show()
    t_end = time.time()
    time_run = t_end - t_start
    return time_run, loss.item()



# def XPINN(true_values, plot_it: bool, num_epochs):
#     t_start = time.time()
#     c = 1
#     h = lambda x: torch.sin(x)
#     g = lambda t: h(-c * t)

#     N = 20
#     X_vals_1 = torch.linspace(0, 5, 10, requires_grad=True)
#     X_vals_2 = torch.linspace(5, 10, 10, requires_grad=True)
#     t_vals_1 = torch.linspace(0, 5, 10, requires_grad=True)
#     t_vals_2 = torch.linspace(5, 10, 10, requires_grad=True)

#     X_train_1, t_train_1 = torch.meshgrid(X_vals_1, t_vals_1, indexing="xy")
#     X_train_1_2, t_train_2 = torch.meshgrid(X_vals_1, t_vals_2, indexing="xy")
#     X_train_2, t_train_1_2 = torch.meshgrid(X_vals_2, t_vals_1, indexing="xy")
#     X_train_2_2, t_train_2_2 = torch.meshgrid(X_vals_2, t_vals_2, indexing="xy")

#     X_train_1 = X_train_1.unsqueeze(-1)
#     X_train_2 = X_train_2.unsqueeze(-1)
#     t_train_1 = t_train_1.unsqueeze(-1)
#     t_train_2 = t_train_2.unsqueeze(-1)

#     X_train_1_2 = X_train_1_2.unsqueeze(-1)
#     X_train_2_2 = X_train_2_2.unsqueeze(-1)
#     t_train_1_2 = t_train_1_2.unsqueeze(-1)
#     t_train_2_2 = t_train_2_2.unsqueeze(-1)

#     domain = [
#         [X_train_1, t_train_1],
#         [X_train_2, t_train_1_2],
#         [X_train_1_2, t_train_2],
#         [X_train_2_2, t_train_2_2]
#     ]

#     all_X_train = []
#     all_t_train = []
#     all_u_pred = []

#     class NeuralNetwork(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.fn_approx = nn.Sequential(
#                 nn.Linear(2, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 32),
#                 nn.Tanh(),
#                 nn.Linear(32, 1)
#             )

#         def forward(self, x, y):
#             x_combined = torch.cat((x, y), dim=2)
#             logits = self.fn_approx(x_combined)
#             return logits

#     models = [NeuralNetwork() for _ in domain]
#     optimizers = [optim.Adam(model.parameters(), lr=0.002) for model in models]
#     criterion = nn.MSELoss()

#     for epoch in range(num_epochs):
#         total_loss = 0

#         for i, (model, optimizer, subdomain) in enumerate(zip(models, optimizers, domain)):
#             X_train, t_train = subdomain

#             u_prediction = model(X_train, t_train)
#             du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
#             du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

#             loss_PDE = criterion(-c * du_dx, du_dt)
#             loss_IC = criterion(model(X_train, torch.zeros_like(t_train)), h(X_train))

#             # Loss along X-interfaces
#             if i == 0:
#                 next_model = models[1]
#             elif i == 1:
#                 next_model = models[0]
#             elif i == 2:
#                 next_model = models[3]
#             elif i == 3:
#                 next_model = models[2]
            
#             X_interface = torch.tensor([5.0], requires_grad=True).expand_as(t_train)
#             t_interface = t_train
#             u_pred_interface = model(X_interface, t_interface)
#             u_next_interface = next_model(X_interface, t_interface)

#             loss_interface_x = criterion(torch.abs(u_pred_interface - u_next_interface), torch.zeros_like(u_pred_interface))

#             # Loss along t-interface
#             if i == 0:
#                 next_model = models[2]
#             elif i == 1:
#                 next_model = models[3]
#             elif i == 2:
#                 next_model = models[0]
#             elif i == 3:
#                 next_model = models[1]

#             X_interface = X_train
#             t_interface = torch.tensor([5.0], requires_grad=True).expand_as(X_train)
#             u_pred_interface = model(X_interface, t_interface)
#             u_next_interface = next_model(X_interface, t_interface)

#             loss_interface_t = criterion(torch.abs(u_pred_interface - u_next_interface), torch.zeros_like(u_pred_interface))

#             loss = loss_PDE + loss_IC + loss_interface_t + loss_interface_x

#             # Apply boundary condition only to the physical boundaries
#             if i in [0, 2]:
#                 loss_boundary = criterion(model(torch.zeros_like(X_train), t_train), g(t_train))
#                 loss += loss_boundary

#             total_loss += loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         if (epoch + 1) % 1 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item():.4f}', end='\r')

#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

#     for model, subdomain in zip(models, domain):
#         X_train, t_train = subdomain
#         with torch.no_grad():
#             u_pred = model(X_train, t_train)
#             u_pred = u_pred.squeeze(-1).numpy()
#             X_train = X_train.squeeze(-1).numpy()
#             t_train = t_train.squeeze(-1).numpy()

#             all_X_train.append(X_train)
#             all_t_train.append(t_train)
#             all_u_pred.append(u_pred)

#     X_train = np.concatenate((np.concatenate((all_X_train[0], all_X_train[1]), axis=1), np.concatenate((all_X_train[2], all_X_train[3]), axis=1)), axis=0)
#     t_train = np.concatenate((np.concatenate((all_t_train[0], all_t_train[1]), axis=1), np.concatenate((all_t_train[2], all_t_train[3]), axis=1)), axis=0)
#     u_pred = np.concatenate((np.concatenate((all_u_pred[0], all_u_pred[1]), axis=1), np.concatenate((all_u_pred[2], all_u_pred[3]), axis=1)), axis=0)



#     if plot_it:
#         #Analytical solution
#         # Parameters
#         c = 1  # Wave speed
#         x = np.linspace(0, 10, 20)  # Spatial domain
#         t = np.linspace(0, 10, 20)  # Time domain
#         X, T = np.meshgrid(x, t)

#         # Function definition
#         def u(x, t, c):
#             return np.sin(x - c * t)

#         U = u(X, T, c)


#         fig = plt.figure(figsize=(18, 6))  # Increased figure width for better layout

#         # First subplot with u_pred
#         ax1 = fig.add_subplot(131, projection='3d')  # Changed to 131 for 1 row, 3 columns, 1st subplot
#         ax1.plot_surface(X_train, t_train, u_pred, cmap='viridis')
#         ax1.set_xlabel('x')
#         ax1.set_ylabel('t')
#         ax1.set_zlabel('u(x,t)')
#         ax1.set_title('XPINN')

#         # Second subplot with true_values for PINN
#         ax2 = fig.add_subplot(132, projection='3d')  # Changed to 132 for 1 row, 3 columns, 2nd subplot
#         ax2.plot_surface(X_train, t_train, true_values, cmap='viridis')
#         ax2.set_xlabel('x')
#         ax2.set_ylabel('t')
#         ax2.set_zlabel('u(x,t)')
#         ax2.set_title('PINN')

#         # Third subplot with true_values for True values
#         ax3 = fig.add_subplot(133, projection='3d')  # Changed to 133 for 1 row, 3 columns, 3rd subplot
#         ax3.plot_surface(X, T, U, cmap='viridis')
#         ax3.set_xlabel('x')
#         ax3.set_ylabel('t')
#         ax3.set_zlabel('u(x,t)')
#         ax3.set_title('True values')

#         plt.tight_layout()
#         plt.show()
#     t_end = time.time()
#     time_run = t_end - t_start
#     return time_run, total_loss.item()

#Running the XPINN function
simulated = PINN(False,1000)
print(simulated[1])
XPINN(simulated[0],True,1000)

def testing():
    import numpy as np
    import matplotlib.pyplot as plt
    
    epoch_vals = [10, 50, 100, 200, 500, 1000, 2000]
    time_vals_PINN = []
    loss_vals_PINN = []

    time_vals_XPINN = []
    loss_vals_XPINN = []
    
    for val in epoch_vals:
        test = PINN(False, val)
        time_vals_PINN.append(test[1])
        loss_vals_PINN.append(test[2])

        test_X = XPINN(test[0],False, val)
        time_vals_XPINN.append(test_X[0])
        loss_vals_XPINN.append(test_X[1])
    
    epoch_vals = np.array(epoch_vals)
    time_vals_PINN = np.array(time_vals_PINN)
    loss_vals_PINN = np.array(loss_vals_PINN)
    time_vals_XPINN = np.array(time_vals_XPINN)
    loss_vals_XPINN = np.array(loss_vals_XPINN)
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for Loss vs. Epochs
    axs[0].plot(epoch_vals, loss_vals_PINN, marker='o', color="blue", label="PINN loss")
    axs[0].plot(epoch_vals, loss_vals_XPINN, marker='o', color="red", label="XPINN loss")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss vs. Epochs')
    axs[0].grid(True)
    axs[0].legend()

    # Plot for Epochs vs. Time
    axs[1].plot(epoch_vals, time_vals_PINN, marker='o', color="blue", label="PINN")
    axs[1].plot(epoch_vals, time_vals_XPINN, marker='o', color="red", label="XPINN")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Time')
    axs[1].set_title('Epochs vs. Time')
    axs[1].grid(True)
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()
#testing()
