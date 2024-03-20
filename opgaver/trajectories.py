import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Physics-Informed Neural Network (PINNS) model
class PINNSModel(nn.Module):
    def __init__(self):
        super(PINNSModel, self).__init__()
        self.fc1 = nn.Linear(2, 50)  # Input: (time, position)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)  # Output: mass

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate sample trajectory data (time, position, mass)
# For simplicity, let's assume a projectile launched from the ground
# with an initial velocity, and we collect data on its position at
# different time intervals.
def generate_data(num_samples):
    times = torch.linspace(0, 5, num_samples).reshape(-1, 1)  # Time samples
    positions = 4.9 * times ** 2  # Assuming projectile motion under gravity
    mass = torch.tensor([[10.0]])  # True mass of the object
    return torch.cat((times, positions, mass.expand(num_samples, -1)), dim=1)

# Train the PINNS model
def train_model(model, data, num_epochs=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        mass_pred = model(data[:, :2])  # Predict mass based on time and position
        loss = criterion(mass_pred, data[:, 2].unsqueeze(1))  # True mass is the third column
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Main
if __name__ == "__main__":
    # Generate sample trajectory data
    num_samples = 50
    trajectory_data = generate_data(num_samples)

    # Initialize PINNS model
    model = PINNSModel()

    # Train the PINNS model using trajectory data
    train_model(model, trajectory_data)

    # Estimate the mass of the object
    estimated_mass = model(torch.tensor([[0.0, 0.0]]))  # Assuming initial position
    print(f"Estimated mass: {estimated_mass.item()} kg")
