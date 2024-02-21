import torch
import torch.nn as nn
import numpy as np
import functions as fc


# Define your functions u(x) and f(x)
# \Delta u = f, u(-5) = g_left, u(5) = g_right
u = lambda x: x**3
f = lambda x: 6*x


# Define boundary conditions
x_left = -5.0
x_right = 5.0
g_left = u(x_left)
g_right = u(x_right)

# Create input data
N = 500
X_train = torch.linspace(x_left, x_right, N, requires_grad=True).view(-1, 1)
X_test = torch.linspace(-10, 10, N, requires_grad=True).view(-1, 1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(3, 20, bias=True)
        self.layer2 = nn.Linear(20, 20, bias=True)
        self.layer3 = nn.Linear(20, 20, bias=True)
        self.layer4 = nn.Linear(20, 20, bias=True)
        self.layer5 = nn.Linear(20, 1, bias=True)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = self.layer5(x)
        return x

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

num_epochs = 20000

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(X_train)
    # Compute the first derivatives
    df_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    
    # Compute the second derivatives
    d2f_dx2 = torch.autograd.grad(df_dx, X_train, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]

    # Compute the loss
    loss = criterion(d2f_dx2, f(X_train)) + criterion(model(torch.tensor([x_left])), torch.tensor([g_left])) + criterion(model(torch.tensor([x_right])), torch.tensor([g_right]))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
print(f"Mean Squared Error on trained data: {loss.item():.4f}")

# Evaluate the model (you can use X_test for evaluation)
# Compute the first derivatives
df_dx = torch.autograd.grad(model(X_test), X_test, create_graph=True, grad_outputs=torch.ones_like(model(X_test)))[0]

# Compute the second derivatives
d2f_dx2 = torch.autograd.grad(df_dx, X_test, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]

# Compute the loss
loss = criterion(d2f_dx2[1:-1, 0], f(X_test)[1:-1, 0]) # + criterion(model(torch.tensor([x_left])), g_left) + criterion(model(torch.tensor([x_right])), g_right)
print(f"Mean Squared Error on test data: {loss.item():.4f}")

# plot the comparison between labels and predictions
fc.plot_comparison(X_train, u(X_train), model(X_train), "u", "f)_plot_u.png")
fc.plot_comparison(X_test, u(X_test), model(X_test), "u", "f)_plot_u_test.png")

# plot second derivative
fc.plot_comparison(X_train, f(X_train), d2f_dx2, "f", "f)_plot_f.png")