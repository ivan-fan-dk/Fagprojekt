import torch
import torch.nn as nn
import numpy as np
import functions as fc
import matplotlib.pyplot as plt
import os

# Define your functions u(x) and f(x)
u = lambda x, y: x**2+y**2
f = lambda x: 4+x-x


# Define boundary conditions
x_left = -5.0
x_right = 5.0
y_left = -5.0
y_right = 5.0

#Number of inputs
nx, ny = (50,50)

#Create inputs of squares
x = torch.linspace(-5.0, 5.0, nx, requires_grad=True)
y = torch.linspace(-5.0, 5.0, ny, requires_grad=True)
xv, yv = torch.meshgrid(x, y)
xv,yv = xv.reshape(-1,1), yv.reshape(-1,1)
#Model input
X_train = torch.stack((xv,yv),dim=1).squeeze()
#Recreate x,y values as cols to use for boundaries
x = torch.linspace(-5.0, 5.0, nx, requires_grad=True).view(-1,1)
y = torch.linspace(-5.0, 5.0, ny, requires_grad=True).view(-1,1)

#Tensors which are used for boundaries 
nones = torch.ones(len(x)).view(-1,1)*(-1)
ones = torch.ones(len(x)).view(-1,1)
nones_long = torch.ones(len(x)**2).view(-1,1)*(-1)
ones_long = torch.ones(len(x)**2).view(-1,1)

#Labels - boundaries and all input
z_y1 = u(x,-1)
z_y2 = u(x,1)
z_x1 = u(-1,y)
z_x2 = u(1,y)
z = u(xv,yv).view(-1,1)

#Create test data
x_t = torch.linspace(-5, 5, nx, requires_grad=True)
y_t = torch.linspace(-5, 5, ny, requires_grad=True)
xv_t, yv_t = torch.meshgrid(x_t, y_t)
xv_t,yv_t = xv_t.reshape(-1,1), yv_t.reshape(-1,1)
#Model input
X_test = torch.stack((xv_t,yv_t),dim=1).squeeze()
Label_test = u(xv_t,yv_t).view(-1,1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 20, bias=True)
        self.layer2 = nn.Linear(20, 20, bias=True)
        self.layer3 = nn.Linear(20, 20, bias=True)
        self.layer4 = nn.Linear(20, 20, bias=True)
        self.layer5 = nn.Linear(20, 1, bias=True)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

num_epochs = 5000

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(X_train)
    #Compute the first derivatives
    df_dx = torch.autograd.grad(u_prediction, X_train, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    
    # Compute the second derivatives
    d2f_dx2 = torch.autograd.grad(df_dx, X_train, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]

    # Compute the loss
    loss =criterion(z_y1, model(torch.stack((x,nones),dim=1).squeeze())) + criterion(z_y2, model(torch.stack((x,ones),dim=1).squeeze()))+criterion(z_x1, model(torch.stack((nones,y),dim=1).squeeze()))+criterion(z_x2, model(torch.stack((ones,y),dim=1).squeeze()))+criterion(torch.stack((ones_long*4,ones_long*4),dim=1).squeeze(), d2f_dx2)+ criterion(u_prediction, z) 

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
print(f"Mean Squared Error on trained data: {loss.item():.4f}")
    # Forward pass
u_test = model(X_test)
#Compute the first derivatives
df_dx = torch.autograd.grad(u_test, X_test, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]

# Compute the second derivatives
d2f_dx2 = torch.autograd.grad(df_dx, X_test, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]

# Compute the loss
loss = criterion(d2f_dx2,f(X_test))

# Evaluate the model (you can use X_test for evaluation)
# Compute the first derivatives
#df_dx = torch.autograd.grad(model(X_test), X_test, create_graph=True, grad_outputs=torch.ones_like(model(X_test)))[0]

# Compute the second derivatives
#d2f_dx2 = torch.autograd.grad(df_dx, X_test, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]

# Compute the loss
for i in range(len(X_test)):
    print("\n",X_test[i].detach(),u_test[i].detach(),Label_test[i].detach())
    print(d2f_dx2[i].detach())
print(f"Mean Squared Error on test data: {loss.item():.4f}")
X = X_test[:,0].detach().numpy()
Y = X_test[:,1].detach().numpy()
Z = u_test.detach().numpy()[:,0]

# Create a figure and an axes object
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Create a figure and an axes object
ax.plot_trisurf(X, Y, Z, color="red")
ax.plot_trisurf(X, Y, Z, color="red")

# Plot the surface
#ax.plot_surface(xv.detach().numpy()[:,0], yv.detach().numpy()[:,0], Z, alpha=0.3, rstride=100, cstride=100, color="red")
ax.set_zlim(0,50)
# Add some labels and a title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Prediction")
ax.set_title("3D Surface Plot")
plt.savefig(os.path.dirname(__file__) + f"/_static/2d_Harris")
# Show the plot
print(X,Y,Z)