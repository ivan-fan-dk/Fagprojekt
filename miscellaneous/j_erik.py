import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

c = 0.5
x0 = 0.
N = 20

# Define boundary conditions
t0 = 0.0
t_final = 10.0
x_left = 0.0
x_right = 10.0

# Create input data
X_vals = torch.linspace(x_left, x_right, N, requires_grad=True)
t_vals = torch.linspace(t0, t_final, N, requires_grad=True)
X_train, t_train = torch.meshgrid(X_vals, t_vals, indexing="xy")
X_train = X_train.unsqueeze(-1)
t_train = t_train.unsqueeze(-1)

# Define functions h(x), u(x)
h = lambda x: 0.5*c/((torch.cosh(0.5*c**(0.5) * (x-torch.ones_like(X_train)*x0)))**2)
u = lambda x, t: h(x-c*t)
g = lambda t: h(torch.ones_like(X_train)*x0-c*t)


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


for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(X_train, t_train)
    # Compute the first derivatives
    du_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    du_dt = torch.autograd.grad(u_prediction, t_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    d2u_dx2 = torch.autograd.grad(du_dx, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    d3u_dx3 = torch.autograd.grad(d2u_dx2, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

    # Compute the loss
    loss_PDE = criterion(-6*u_prediction*du_dx - d3u_dx3, du_dt)
    loss_boundary = criterion(model(x0*torch.ones_like(X_train), t_train), g(t_train))
    loss_IC = criterion(model(X_train, t0*torch.ones_like(t_train)), h(X_train))
    loss = 100*(loss_PDE + 3*loss_boundary + 4*loss_IC)

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
    u_true = u(X_train, t_train)
    u_pred =  u_pred.squeeze(-1).numpy()
    X_train = X_train.squeeze(-1).numpy()
    t_train = t_train.squeeze(-1).numpy()

fig, axs = plt.subplots(2, 1, figsize=(10, 10), subplot_kw={'projection': '3d'})

# Plot the predicted surface
axs[0].view_init(30, 30)
axs[0].plot_surface(X_train, t_train, u_pred, cmap='viridis', alpha=0.5)
axs[0].set_title('Predicted')
axs[0].set_xlabel('x')
axs[0].set_ylabel('t')
axs[0].set_zlabel('u(x,t)')

# Plot the true surface
axs[1].view_init(30, 30)
axs[1].plot_surface(X_train, t_train, u_true.squeeze(-1).numpy(), cmap='terrain', alpha=0.5)
axs[1].set_title('True')
axs[1].set_xlabel('x')
axs[1].set_ylabel('t')
axs[1].set_zlabel('u(x,t)')

plt.tight_layout()
plt.savefig("opgaver/_static/j_3d_plot.png")