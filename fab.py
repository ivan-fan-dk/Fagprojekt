# Enten 'cubic', 'sine', 'abosulute' 'exp_sin'

# Kræver import torch
#import torch.nn as nn
#import torch.optim as optim
#import numpy as np
#import matplotlib.pyplot as plt

def neural_approx(choice):

    functions = {
        'cubic': (lambda x: x**3 + x**2 + x, lambda x: 3*x**2 + 2*x + 1),
        'sine': (np.sin, np.cos),
        'absolute': (np.abs, np.sign),
        'exp_sin': (lambda x: np.exp(np.sin(x)), lambda x: np.cos(x) * np.exp(np.sin(x)))
    }

    func, dfunc = functions[choice]


    class Net(nn.Module):
        def __init__(self, activation):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(1, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.activation = layer

        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)
            return x

    # Choose the activation function
    if choice == 'sine' or choice == 'exp_sin':
        layer = nn.Tanh()
    else:
        layer = nn.ReLU()

    # Initialize the network and optimizer
    net = Net(layer)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Generate some data
    x = torch.linspace(-10, 10, 1000).reshape(-1, 1)
    y = func(x)

    # Train the network
    for epoch in range(5000):
        output = net(x)
        loss = nn.MSELoss()(output, y)
        if epoch % 100 == 99:
            print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test the network
    x_test = torch.tensor([2.0]).reshape(-1, 1)
    y_test = net(x_test)

    # Compute the derivative
    x_test.requires_grad = True
    y_test = net(x_test)
    y_test.backward()

    print(f"Function output: {y_test.detach().numpy()[0][0]}")
    print(f"Derivative: {x_test.grad.detach().numpy()[0][0]}")


    x_test_plt = x.clone().reshape(-1, 1)
    y_test_plt = net(x_test_plt).detach().numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(x_test_plt.detach().numpy(), func(x_test_plt.detach().numpy()), label='True function')
    plt.plot(x_test_plt.detach().numpy(), y_test_plt, label='Neural network approximation')
    plt.legend()
    plt.show()

    true_derivative = dfunc(x_test_plt.detach().numpy())

    x_test_plt.requires_grad = True
    y_values = net(x_test_plt)
    v = torch.ones(y_test_plt.shape)
    y_values.backward(v)
    approx_derivative = x_test_plt.grad.detach().numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(x_test_plt.detach().numpy(), true_derivative, label='True derivative')
    plt.plot(x_test_plt.detach().numpy(), approx_derivative, label='Approximated derivative')
    plt.legend()
    plt.show()

    return 'Double dab'