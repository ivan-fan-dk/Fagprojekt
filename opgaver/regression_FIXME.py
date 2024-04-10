# FIXME
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = torch.linspace(-10,10,5000)
t = torch.linspace(0,5,5000)

A = torch.tensor([2])
v = torch.tensor([1])
c = torch.tensor([3])
c1 = torch.tensor([3])
c2 = torch.tensor([0.5])
a =  torch.tensor([v/(2*c1)])
x0 = torch.tensor([1])

B = torch.sqrt(c2*A**2 / (2*c1))
b = c1*(B**2 - a**2)
xi = x-x0*torch.ones_like(x)-v*t

rho = A * 1/torch.cosh(B*xi)
phi = a*x+b*t+c*torch.ones_like(x)

true_vals_real = rho * torch.cos(phi)
true_vals_imag = rho * torch.sin(phi)
true_vals = torch.stack((true_vals_real, true_vals_imag), dim=x.dim())

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(2,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,6),
        )
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x, t):
        if x.dim() != t.dim():
            raise AssertionError(f"x and y must have the same number of dimensions, but got x.dim() == {x.dim()} and y.dim() == {y.dim()}")
        x_combined = torch.stack((x, t),dim=x.dim())
        vars = self.fn_approx(x_combined).squeeze()
        self.A = torch.mean(vars[:, 0])
        self.c1 = torch.mean(vars[:, 1])
        self.c2 = torch.mean(vars[:, 2])
        self.v = torch.mean(vars[:, 3])
        self.c = torch.mean(vars[:, 4])
        self.x0 = torch.mean(vars[:, 5])
        
        self.B = torch.sqrt(self.c1*(self.A)**2 / (2*self.c1))
        self.a = self.v/(2*self.c1)
        self.b = self.c1*((self.B)**2-(self.a)**2)

        self.xi = x-self.x0*torch.ones_like(x)-self.v*t
        self.rho = self.A * 1/torch.cosh(self.B*xi)
        self.phi = self.a*x+self.b*t+self.c*torch.ones_like(x)

        self.real = self.rho * torch.cos(self.phi)
        self.im = self.rho * torch.sin(self.phi)
        out = torch.stack((self.real, self.im), dim=x.dim())

        return out

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = int(5e3)
for epoch in range(num_epochs):
    loss = criterion(model(x,t), true_vals)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.7f}', end='\r')

print(f"Mean Squared Error on trained data: {loss.item():.4f}")
print(torch.tensor([A,c1,c2,v,c,x0]))
print(torch.tensor([model.A,model.c1,model.c2,model.v,model.c,model.x0]))

quit()
X, T = torch.meshgrid(x, t)

xi = X-x0*torch.ones(X.shape)-v*T
rho = A * 1/torch.cosh(B*xi)
phi = a*X+b*T+c*torch.ones(X.shape)

true_vals_real_num = (rho * torch.cos(phi)).numpy()
true_vals_imag_num = (rho * torch.sin(phi)).numpy()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.numpy(), T.numpy(), np.sqrt(true_vals_real_num**2 + true_vals_imag_num**2), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
plt.savefig("opgaver/_static/regression.png")