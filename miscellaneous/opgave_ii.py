import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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


X_test = torch.linspace(-10, 10, N, requires_grad=True).view(-1, 1)

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
    loss = loss_PDE + loss_boundary + loss_IC

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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_train, t_train, u_pred, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    plt.savefig("opgaver/_static/ii_3d_plot.png")