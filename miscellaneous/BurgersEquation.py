import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time 

# Define functions h(x), u(x)
h = lambda x: torch.sin(x)
u = lambda x, t: -torch.sin(x*torch.pi)


# Define boundary conditions
t0 = 0.0
t_final = 1.0
x_left = -1.0
x_right = 1.0
loss_list = []
time_list = []
# Create input data
N = 100
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

num_epochs = 5000
model(X_train, t_train)
total_time = 0
for epoch in range(num_epochs):
    time0 = time.time()
    # Forward pass
    u_prediction = model(X_train, t_train)
    # Compute the first derivatives
    du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    du_dxx = torch.autograd.grad(du_dx, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

    # Compute the loss
    loss_PDE = criterion(du_dt+u_prediction*du_dx-(0.01/torch.pi)*du_dxx,torch.zeros_like(u_prediction))
    loss_IC = criterion(model(X_train,torch.zeros_like(t_train)),-torch.sin(torch.pi*X_train))
    loss_boundary = criterion(model(-1*torch.ones_like(X_train),t_train), model(1*torch.ones_like(X_train),t_train))


    loss = loss_PDE + loss_boundary + loss_IC

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
        loss_list.append(loss.item())
    time1 = time.time()
    run_time = time1-time0
    total_time += run_time
    time_list.append(total_time)

print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    #for the 3d plot we import the matplotlib extension:
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    u_pred = model(X_train, t_train)
    u_pred =  u_pred.squeeze(-1).numpy()
    X_train = X_train.squeeze(-1).numpy()
    t_train = t_train.squeeze(-1).numpy()

    epoch_vals = np.array([i for i in range(num_epochs)])
    time_vals_PINN = np.array(time_list)
    loss_vals_PINN = np.array(np.log(loss_list))

    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for Loss vs. Epochs
    axs[0].plot(epoch_vals, loss_vals_PINN, marker='o', color="blue", label="PINN loss")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('log(Loss)')
    axs[0].set_title('log(Loss) vs. Epochs')
    axs[0].grid(True)
    axs[0].legend()

    # Plot for Epochs vs. Time
    axs[1].plot(epoch_vals, time_vals_PINN, marker='o', color="blue", label="PINN")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Time')
    axs[1].set_title('Epochs vs. Time')
    axs[1].grid(True)
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_train, t_train, u_pred, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    plt.show()