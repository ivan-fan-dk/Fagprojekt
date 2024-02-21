import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import functions as fc

N = 1000
X,y = fc.data1D(lambda x,: np.sin(x)*x+np.cos(x),-30,30,N)
#y = X**3+X**2

X = torch.tensor(X, dtype=torch.float32).view(-1, 1)    
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

#this is just defining the NN:
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

# Create an instance of the model, define loss and optimizer
model = NeuralNetwork()
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
## How is this using test data?
with torch.no_grad():
    predictions = model(X)
    loss = criterion(predictions, y)
    print(f"Mean Squared Error on test data: {loss.item():.4f}")

#now that we have the model we can find the approximated f'(x_0) for some x_0 
#this is an approximated derivative from the .backward() method! 


d = torch.tensor([2.5], requires_grad=True)

# Forward pass through the network
z = model(d)

# Backward pass to compute gradients
z.backward()

# The gradient at x = 2.5 is stored in x.grad
print(d.grad, "her!")