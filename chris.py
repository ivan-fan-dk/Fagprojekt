import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
N = 1000
X = np.linspace(-30, 30, N)
y = np.sin(X)*X+np.cos(X)
#y = X**3+X**2

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
