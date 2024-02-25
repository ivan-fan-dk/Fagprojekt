import torch
import torch.nn as nn
import numpy as np
import functions as fc
from softadapt import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

N = 100
x0 = 1
v0 = 0
w0 = 3
l0 = 0.5
t0 = 0
t_max = 10

# Generate training data
t_range = torch.linspace(t0, t_max, steps=N, requires_grad=True)  
grid_T = t_range.unsqueeze(1)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

    def forward(self, x):
        logits = self.fn_approx(x)
        return logits

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20000

 
#setup for softadapt:
# Change 1: Create a SoftAdapt object (with your desired variant)
softadapt_object = NormalizedSoftAdapt(beta=0.1)

# Change 2: Define how often SoftAdapt calculate weights for the loss components
epochs_to_make_updates = 5

# Change 3: Initialize lists to keep track of loss values over the epochs we defined above
values_of_component_1 = []
values_of_component_2 = []
values_of_component_3 = []

# Initializing adaptive weights to all ones.
adapt_weights = torch.tensor([1,1,1])

for epoch in range(num_epochs):
    # Forward pass
    u_prediction = model(grid_T)

    # Compute the first derivatives
    df_dx = torch.autograd.grad(u_prediction, grid_T, create_graph=True, grad_outputs=torch.ones_like(u_prediction))[0]
    df_df_dx = torch.autograd.grad(df_dx, grid_T, create_graph=True, grad_outputs=torch.ones_like(df_dx))[0]
    
    # Adding initial conditions
    ivp_cond_x = criterion(model(t0 * torch.ones(N).view(-1,1,1)),  x0*torch.ones(N).view(-1,1,1))
    ivp_cond_v = criterion(model(t0 * torch.ones(N).view(-1,1,1)),  v0*torch.ones(N).view(-1,1,1))

    #DE criterion:

    DE_loss = criterion(df_df_dx, -w0**2*model(grid_T)-l0*df_dx)

    # Backward pass and optimization
    optimizer.zero_grad()

    values_of_component_1.append(DE_loss)
    values_of_component_2.append(ivp_cond_x)
    values_of_component_3.append(ivp_cond_v)

    # Compute the loss

    # Change 4: Append the loss values to the lists
    if epoch % epochs_to_make_updates == 0 and epoch != 0:
        adapt_weights = softadapt_object.get_component_weights(
        torch.tensor(values_of_component_1), 
        torch.tensor(values_of_component_2), 
        torch.tensor(values_of_component_3), 
        verbose=False,
        )
                                                            
      
          # Resetting the lists to start fresh (this part is optional)
        values_of_component_1 = []
        values_of_component_2 = []
        values_of_component_3 = []


      # Change 5: Update the loss function with the linear combination of all components.
    loss = adapt_weights[0] * DE_loss + 4*adapt_weights[1] * ivp_cond_x + adapt_weights[2] * ivp_cond_v

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
print(f"Mean Squared Error on trained data: {loss.item():.4f}")

# Plot the results
# Compute model predictions
with torch.no_grad():
    predictions = model(torch.linspace(t0,t_max,N).view(-1,1,1))


predictions_np = predictions.squeeze(-1).detach().numpy()  # Convert tensor to numpy array
t_range_np = t_range.detach().numpy()  # Convert tensor to numpy array


### SOLVE NUMERICALLY with runge Kutta

# Define the differential equation
def runge_kutta_solver(func, y0, t_span, h, omega_0, damping):
    t_values = torch.arange(t_span[0], t_span[1] + h, h)
    n = len(t_values)
    y_values = torch.zeros((n, len(y0)))
    y_values[0] = y0

    for i in range(1, n):
        k1 = h * func(t_values[i - 1], y_values[i - 1], omega_0, damping)
        k2 = h * func(t_values[i - 1] + h/2, y_values[i - 1] + k1/2, omega_0, damping)
        k3 = h * func(t_values[i - 1] + h/2, y_values[i - 1] + k2/2, omega_0, damping)
        k4 = h * func(t_values[i - 1] + h, y_values[i - 1] + k3, omega_0, damping)

        y_values[i] = y_values[i - 1] + (k1 + 2*k2 + 2*k3 + k4)/6

    return t_values, y_values

# Damped harmonic oscillator equation
def damped_harmonic_oscillator(t, y, omega_0, damping):
    x, v = y
    dxdt = v
    dvdt = - omega_0**2 * x - damping * v
    return torch.tensor([dxdt, dvdt])

# Runge-Kutta numerical solution
t_values_rk, y_values_rk = runge_kutta_solver(damped_harmonic_oscillator, torch.tensor([x0, v0]), [t0, t_max], (t_max - t0) / (N - 1), w0, l0)


#Analytical solution
u = lambda t: np.exp(-0.25*t)*(0.0836242*np.sin(2.9*t)+np.cos(2.9*t)) 

plt.figure(figsize=(10, 5))

# Plotting predictions

plt.scatter(t_range_np, u(t_range_np), label='Analytical Data')
plt.plot(t_range_np, predictions_np, label='Predictions', color='red')
plt.legend()
plt.title("Test")
filename = "i_plot"
plt.savefig(os.path.dirname(__file__) + f"/_static/{filename}")

