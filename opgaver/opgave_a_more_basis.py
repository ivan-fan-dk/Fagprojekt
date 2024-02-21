import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import functions as fc

f = lambda x: np.sin(x)*x+np.cos(x)
X,y = fc.data(lambda x,: np.sin(x)*x+np.cos(x),-5,5,100,1)

X = torch.column_stack((X, torch.cos(X), torch.sin(X), torch.multiply(X, torch.sin(X))))#, torch.dot(X,torch.sin(X))))

X_test = torch.arange(-20,20,0.1).view(-1,1)
X_test = torch.column_stack((X_test, torch.cos(X_test), torch.sin(X_test), torch.multiply(X_test, torch.sin(X_test)))) #, torch.dot(X,torch.sin(X))))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(X.shape[1], 20, bias=True)
        self.layer2 = nn.Linear(20, 20, bias=True)
        self.layer3 = nn.Linear(20, 20, bias=True)
        self.layer4 = nn.Linear(20, 20, bias=True)
        self.layer5 = nn.Linear(20, 1, bias=True)


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
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

# Evaluate the model
with torch.no_grad():
    predictions = model(X)
    loss = criterion(predictions, y)
    print(f"Mean Squared Error on test data: {loss.item():.4f}")

# Visualize the results
plt.scatter(X[:,0], y, label='Actual Data', color="green")
#plt.scatter(k, g, label="Actual Function", color="grey")
plt.plot(X[:,0], predictions, label='Predictions', color='red')

plt.legend()
plt.savefig("opgaver/_static/a)_plot_more_basis.png")
plt.clf()

# Visualize the results
plt.scatter(X_test.detach().numpy()[:,0], f(X_test.detach().numpy()[:,0]), label='Actual Data', color="green")
#plt.scatter(k, g, label="Actual Function", color="grey")
plt.plot(X_test.detach().numpy()[:,0], model(X_test).detach().numpy(), label='Predictions', color='red')

plt.legend()
plt.savefig("opgaver/_static/a)_plot_test_more_basis.png")