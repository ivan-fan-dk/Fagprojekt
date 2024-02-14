import numpy as np
import torch

def data(f, lower_bound, upper_bound,N,inputdims):
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
X0,X1,X2,X3,y = data(lambda x0,x1,x2,x3: np.sin(x0)*x1+np.cos(x2)+x3,0,10,100,4)