import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#datasets:
from sklearn import datasets
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#prep data:

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=2)


X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1)
X_test =   torch.from_numpy(X_test.astype(np.float32)).view(-1,1)
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1,1)
y_test =   torch.from_numpy(y_test.astype(np.float32)).view(-1,1)

#model:

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.basic_btch = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )   
    
    def forward(self, x):
        logits = self.basic_btch(x)
        return logits

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

#initializing the model and training it:
model = NeuralNet()
