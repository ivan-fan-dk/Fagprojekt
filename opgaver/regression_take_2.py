#schrodinger regression thing take 2 - we will start by just trying to approximate c1 and c2!!!!!
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

hidden_units = 50

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        #create the parameter for adaptive activation function thing:
        self.n = 10.5
        self.alpha = nn.Parameter(torch.tensor(1/self.n)) #initialize so that na=1
        self.lin1 = nn.Linear(2,hidden_units)
        self.lin2 = nn.Linear(hidden_units,hidden_units)
        self.lin3 = nn.Linear(hidden_units,hidden_units)
        self.lin4 = nn.Linear(hidden_units,hidden_units)
        self.out = nn.Linear(hidden_units,2)

    def forward(self, x, y):
        x_combined = torch.cat((x, y),dim=1)
        #do the activation function thing
        x1 = torch.tanh(self.n * self.alpha * self.lin1(x_combined))
        x2 = torch.tanh(self.n * self.alpha * self.lin2(x1))
        x3 = torch.tanh(self.n * self.alpha * self.lin3(x2))
        x4 = torch.tanh(self.n * self.alpha * self.lin4(x3))
        logits = torch.abs(self.out(x4))+0.01
        return logits
    
#standard loss setup: they use MSE + ADAM, so we do too, might be worth looking into LBFGS
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)  #their lr is 0.0008


N = 300
#true values:
x = torch.linspace(-5,5,N).reshape(-1,1)
t = torch.linspace(0,torch.pi/2,N).reshape(-1,1)

c1 = 1/2 * torch.ones_like(x)
c2 = 1 * torch.ones_like(x)

#################################
#setting up a fat juicer function to put stuff into:
def f(x,t,c1,c2):
    A = torch.tensor([2])
    v = torch.tensor([1])
    c = torch.tensor([3])
    x0 = torch.tensor([1])

    a = 1/2 * v * c1**(-1)
    B = torch.sqrt(1/2 * c2 * A**2 * c1**(-1))
    b = c1*(B**2 - a**2)
    xi = x-x0*torch.ones_like(x)-v*t

    rho = A * 1/torch.cosh(B*xi)
    phi = a*x+b*t+c*torch.ones_like(x)

    true_vals_real = rho * torch.cos(phi)
    true_vals_imag = rho * torch.sin(phi)

    true_vals = torch.stack((true_vals_real, true_vals_imag), dim=1)
    return true_vals

#training loop

epochs_to_make_updates = 10
num_epochs_adam = 22_000 


for epoch in range(num_epochs_adam):
    optimizer.zero_grad()
    c1_m = model(x,t)[:,0].reshape(-1,1)
    c2_m = model(x,t)[:,1].reshape(-1,1)
    #train so that c1 and c2 are guessed correctly
    loss = criterion(f(x,t,c1_m,c2_m),f(x,t,c1,c2))

    loss.backward()
    optimizer.step()

    if (epoch + 1) % epochs_to_make_updates == 0:
        print(f'Epoch [{epoch+1}/{num_epochs_adam}], Loss: {loss.item():.8f}', end='\r')
        #print(torch.mean(model(x,t)[:,0]))
        #print(torch.mean(model(x,t)[:,1]))

#print(torch.mean(model(x,t)[:,0]))
#print(torch.mean(model(x,t)[:,1]))

#plot the two different models:
c1_NN = torch.mean(model(x,t)[:,0])
c2_NN = torch.mean(model(x,t)[:,1])
print(c1_NN, c2_NN)
f_NN = f(x,0.79*torch.ones_like(t),c1_NN,c2_NN).detach().numpy()
f_true = f(x,0.79*torch.ones_like(t),c1[0],c2[0]).detach().numpy()
abs_vals_NN = np.sqrt(f_NN[:,0]**2 + f_NN[:,1]**2)
abs_vals_f = np.sqrt(f_true[:,0]**2 + f_true[:,1]**2)



plt.plot(x,abs_vals_NN,label='True', color='blue')
plt.plot(x,abs_vals_f,label='NN', linestyle='--', color='red')
plt.legend()
plt.savefig("NN_vs_true_schrod.png")

