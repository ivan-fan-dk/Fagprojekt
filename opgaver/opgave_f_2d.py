import torch
import torch.nn as nn
import numpy as np
import functions as fc


# Define your functions u(x) and f(x)
u = lambda x, y: x**2+y**2
f = lambda x: 4


# Define boundary conditions
x_left = -5.0
x_right = 5.0
y_left = -5.0
y_right = 5.0


nx, ny = (11, 11)
x = torch.linspace(-5.0, 5.0, nx, requires_grad=True)
y = torch.linspace(-5.0, 5.0, ny, requires_grad=True)
xv, yv = torch.meshgrid(x, y)
xv,yv = xv.reshape(-1,1), yv.reshape(-1,1)
xy = torch.stack((xv,yv),dim=1).squeeze()
x = torch.linspace(-5.0, 5.0, nx, requires_grad=True).view(-1,1)
y = torch.linspace(-5.0, 5.0, ny, requires_grad=True).view(-1,1)
nones = torch.ones(len(x)).view(-1,1)*(-1)
ones = torch.ones(len(x)).view(-1,1)

z_y1 = u(x,-1)
z_y2 = u(x,1)
z_x1 = u(-1,y)
z_x2 = u(1,y)
z = u(xv,yv).view(-1,1)

# Create input data
N = 500

#X_train = torch.linspace(x_left, x_right, N, requires_grad=True).view(-1, 1)
#X_test = torch.linspace(-10, 10, N, requires_grad=True).view(-1, 1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 20, bias=True)
        self.layer2 = nn.Linear(20, 20, bias=True)
        self.layer3 = nn.Linear(20, 20, bias=True)
        self.layer4 = nn.Linear(20, 20, bias=True)
        self.layer5 = nn.Linear(20, 1, bias=True)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

num_epochs = 20000

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(xy)
    # Compute the first derivatives
    #df_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    
    # Compute the second derivatives
    #d2f_dx2 = torch.autograd.grad(df_dx, X_train, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]

    # Compute the loss
    loss = criterion(u_prediction, z) + criterion(z_y1, model(torch.stack((x,nones),dim=1).squeeze())) + criterion(z_y2, model(torch.stack((x,ones),dim=1).squeeze()))+criterion(z_x1, model(torch.stack((nones,y),dim=1).squeeze()))+criterion(z_x2, model(torch.stack((ones,y),dim=1).squeeze()))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
print(f"Mean Squared Error on trained data: {loss.item():.4f}")

# Evaluate the model (you can use X_test for evaluation)
# Compute the first derivatives
#df_dx = torch.autograd.grad(model(X_test), X_test, create_graph=True, grad_outputs=torch.ones_like(model(X_test)))[0]

# Compute the second derivatives
#d2f_dx2 = torch.autograd.grad(df_dx, X_test, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]

# Compute the loss
loss = criterion(u_prediction, z) + criterion(z_y1, model(x,torch.ones(len(z_y1))*(-1))) + criterion(z_y2, model(x,torch.ones(len(z_y1))))+criterion(z_x1, model(torch.ones(len(z_y1))*(-1),y))+criterion(z_x2, model(torch.ones(len(z_y2)),y))
print(f"Mean Squared Error on training data: {loss.item():.4f}")

# plot the comparison between labels and predictions
#fc.plot_comparison(X_train, u(X_train), model(X_train), "u", "f)_plot_u.png")
#fc.plot_comparison(X_test, u(X_test), model(X_test), "u", "f)_plot_u_test.png")

# plot second derivative

#fc.plot_comparison(X_train, f(X_train), d2f_dx2, "f", "f)_plot_f.png")