import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def data(f, lower_bound, upper_bound, N, inputdims):
    """
    input: function's expression f (use lambda function), bounds,number of data points,inputdims
    output: return labels y
    example with 4 dims: 
    X0,X1,X2,X3,y = data(lambda x0,x1,x2,x3: np.sin(x0)*x1+np.cos(x2)+x3,0,10,100,4)

    """
    X = []
    for i in range(inputdims):
        #Create input data
        X.append(np.linspace(lower_bound, upper_bound, N))
    #Calculate label and return input and output
    if inputdims == 1:
        z = f(X[0])
        return torch.tensor(X[0], dtype=torch.float32).view(-1, 1), torch.tensor(z, dtype=torch.float32).view(-1, 1)
    elif inputdims == 2:
        z = f(X[0],X[1])
        return torch.tensor(X[0], dtype=torch.float32).view(-1, 1),torch.tensor(X[1], dtype=torch.float32).view(-1, 1), torch.tensor(z, dtype=torch.float32).view(-1, 1)
    elif inputdims == 3: 
        z = f(X[0],X[1],X[2])
        return torch.tensor(X[0], dtype=torch.float32).view(-1, 1),torch.tensor(X[1], dtype=torch.float32).view(-1, 1),torch.tensor(X[2], dtype=torch.float32).view(-1, 1), torch.tensor(z, dtype=torch.float32).view(-1, 1)
    elif inputdims == 4: 
        z = f(X[0],X[1],X[2],X[3])
        return torch.tensor(X[0], dtype=torch.float32).view(-1, 1),torch.tensor(X[1], dtype=torch.float32).view(-1, 1),torch.tensor(X[2], dtype=torch.float32).view(-1, 1),torch.tensor(X[3], dtype=torch.float32).view(-1, 1),torch.tensor(z, dtype=torch.float32).view(-1, 1)
    elif inputdims == 5: 
        z = f(X[0],X[1],X[2],X[3],X[4])
        return torch.tensor(X[0], dtype=torch.float32).view(-1, 1),torch.tensor(X[1], dtype=torch.float32).view(-1, 1),torch.tensor(X[2], dtype=torch.float32).view(-1, 1),torch.tensor(X[3], dtype=torch.float32).view(-1, 1),torch.tensor(X[4], dtype=torch.float32).view(-1, 1),torch.tensor(z, dtype=torch.float32).view(-1, 1)
    else: return print("Dim not supported")

def plot_comparison(X, y, y_prediction, plot_title:str ="a title", filename:str ="filename"):
    """
    input:
    X, y, y_prediction, both tensors and numpy array are accepted
    plot_title is the title on the graph
    filename 
    output: plot the comparison between labels and predictions, save it in _static/[filename]
    """
    # convert tensor to array
    if torch.is_tensor(X):
        X = X.detach().numpy()
    if torch.is_tensor(y):
        y = y.detach().numpy()
    if torch.is_tensor(y_prediction):
        y_prediction = y_prediction.detach().numpy()

    plt.scatter(X, y, label='Actual Data')
    plt.plot(X, y_prediction, label='Predictions', color='red')
    plt.legend()
    plt.title(plot_title)
    plt.savefig(os.path.dirname(__file__) + f"/_static/{filename}")
    # clear figure
    plt.clf()