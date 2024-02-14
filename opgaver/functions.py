import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import functions.py as fc

def f(f, N, lower_bound, upper_bound):
    """
    input: function's expression f, and input data X
    output: return labels y
    """
    X = np.linspace(lower_bound, upper_bound, N)
    y = f(X)
    return torch.tensor(X, dtype=torch.float32).view(-1, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

X = torch.tensor(X, dtype=torch.float32).view(-1, 1)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 10, bias=True)
        self.layer2 = nn.Linear(10, 50, bias=True)
        self.layer3 = nn.Linear(50, 30, bias=True)
        self.layer4 = nn.Linear(30, 10, bias=True)
        self.layer5 = nn.Linear(10, 1, bias=True)


    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

# Create an instance of the model
model = NeuralNetwork()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 20000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    predictions = model(X)
    loss = criterion(predictions, y)
    print(f"Mean Squared Error on test data: {loss.item():.4f}")

# Convert predictions and input data back to numpy for visualization
predictions = predictions.numpy()

X = np.linspace(-30, 30, N)
y = np.sin(X)*X+np.cos(X)
#y = X**3+X**2


# Visualize the results
plt.scatter(X, y, label='Actual Data', color="green")
#plt.scatter(k, g, label="Actual Function", color="grey")
plt.plot(X, predictions, label='Predictions', color='red')

plt.legend()
plt.show()

## Testing outside the data scope
# Generate new data points outside the training range
new_X = np.linspace(-50, 50, 200)
new_X_tensor = torch.tensor(new_X, dtype=torch.float32).view(-1, 1)

# Make predictions for the new data points
with torch.no_grad():
    new_predictions = model(new_X_tensor)

# Convert predictions and input data back to numpy for visualization
new_predictions = new_predictions.numpy()

# Visualize the results, including the predictions for new data points
plt.scatter(X, y, label='Training Data')
plt.plot(X, predictions, label='Predictions on Training Data', color='red')
plt.plot(new_X, new_predictions, label='Predictions on New Data', linestyle='dashed', color='green')
plt.legend()
plt.show()


## Automatic differentiation

import torch
from torch.autograd import Variable
import numpy as np

# Define a range of x values
x_values = torch.arange(-15, 15, step=0.1, dtype=torch.float32)

# Create an empty list to store y values
y_values = []

# Perform automatic differentiation for each x value
for x_val in x_values:
    # Define a variable with requires_grad=True
    x = Variable(torch.tensor([x_val]), requires_grad=True)
    
    # Define a mathematical expression
    y = torch.sin(2*x)+torch.cos(x)
    
    # Compute the gradient
    y.backward()
    
    # Append the y value to the list
    y_values.append(x.grad.item())

# Print the list of y values
print("List of y values:", y_values)


# Define the function y = x^2 + 3x + 1
y_func = lambda x: 2*torch.cos(2*x)-torch.sin(x)

# Calculate y values for each x
y_values_func = [y_func(x) for x in x_values]


plt.scatter(x_values.detach().numpy(), y_values, label='Actual Data')
plt.plot(x_values.detach().numpy(), y_values_func, label='Predictions', color='red')
plt.legend()
plt.show()