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
X,y = fc.data(f,-10,10,N,1)
dy = fc.data(df,-10,10,N,1)[1]

labels = torch.stack((y,dy),dim=1).squeeze()
print(X.shape)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 10, bias=True)
        self.layer2 = nn.Linear(10, 50, bias=True)
        self.layer3 = nn.Linear(50, 30, bias=True)
        self.layer4 = nn.Linear(30, 10, bias=True)
        self.layer5 = nn.Linear(10, 2, bias=True)


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
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

# Evaluate the model
with torch.no_grad():
    predictions = model(X)
    loss = criterion(predictions, labels)
    print(f"Mean Squared Error on test data: {loss.item():.4f}")

#%%

plt.scatter(X.detach().numpy(), y, label='Actual Data for f')
plt.plot(X.detach().numpy(), predictions[:,0], label='Predictions for f', color='red')
plt.legend()
plt.savefig("opgaver/_static/c)_plot_f.png")
plt.clf()

plt.scatter(X.detach().numpy(), dy, label='Actual Data for df')
plt.plot(X.detach().numpy(), predictions[:,1], label='Predictions for df', color='red')
plt.legend()
plt.savefig("opgaver/_static/c)_plot_df.png")
plt.clf()

X_test = torch.arange(-20,20.).view(-1,1)
print(X_test.shape)
plt.scatter(X_test.numpy(), f(X_test.numpy()), label='Actual Data for f')
plt.plot(X_test.numpy(), model(X_test).detach().numpy()[:,0], label='Predictions for f', color='red')
plt.legend()
plt.savefig("opgaver/_static/c)_plot_f_test.png")
plt.clf()

plt.scatter(X_test.numpy(), df(X_test.numpy()), label='Actual Data for df')
plt.plot(X_test.numpy(), model(X_test).detach().numpy()[:,1], label='Predictions for df', color='red')
plt.legend()
plt.savefig("opgaver/_static/c)_plot_df_test.png")