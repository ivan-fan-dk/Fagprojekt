import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import miscellaneous.functions as fc

f = lambda x: np.sin(x)*x+np.cos(x)
df = lambda x: np.cos(x)*x
N = 1000
X,y = fc.data(f,-10,10,N,1)
dy = fc.data(df,-10,10,N,1)[1]

# Alternative: generate random data within -10,10
# X = torch.sort(torch.rand(N)*20-10)[0].view(-1,1)
# y, dy = f(X), df(X)

labels = torch.stack((y,dy),dim=1).squeeze()

X_test = torch.arange(-20.,20.).view(-1,1)

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

# regularizer on weight_decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training the model
num_epochs = 10000

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


fc.plot_comparison(X, y, predictions[:,0], "f", "c)_plot_f.png")
fc.plot_comparison(X, dy, predictions[:,1], "df", "c)_plot_df.png")
fc.plot_comparison(X_test, f(X_test), model(X_test)[:,0], "f", "c)_plot_f_test.png")
fc.plot_comparison(X_test, df(X_test), model(X_test)[:,1], "df", "c)_plot_df_test.png")