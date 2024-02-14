#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import functions as fc

f = lambda x: np.sin(x)*x+np.cos(x)
df = lambda x: np.cos(x)*x
N = 100
X1,y1 = fc.data(f,0,10,N,1)
X2,y2 = fc.data(df,0,10,N,1)
X = torch.cat((X1,X2),0)
y = torch.cat((y1,y2),0)

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
num_epochs = 2000
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

#%%

plt.scatter(X[:N].detach().numpy(), y[:N], label='Actual Data for f')
plt.plot(X[:N].detach().numpy(), predictions[:N], label='Predictions for f', color='red')
plt.legend()
plt.savefig("opgaver/_static/c)_plot_f.png")
plt.clf()

plt.scatter(X[N:].detach().numpy(), y[N:], label='Actual Data for df')
plt.plot(X[N:].detach().numpy(), predictions[N:], label='Predictions for df', color='red')
plt.legend()
plt.savefig("opgaver/_static/c)_plot_df.png")