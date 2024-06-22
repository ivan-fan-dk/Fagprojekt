import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

# We redefine the loss components above for the sake of completeness.
loss_component_1 = torch.tensor([1, 2, 3, 4, 5])
loss_component_2 = torch.tensor([150, 100, 50, 10, 0.1])
loss_component_3 = torch.tensor([1500, 1000, 500, 100, 1])

# Here we define the different SoftAdapt objects
softadapt_object  = SoftAdapt(beta=0.1)
normalized_softadapt_object  = NormalizedSoftAdapt(beta=0.1)
loss_weighted_softadapt_object  = LossWeightedSoftAdapt(beta=0.1)

print("1")
exit()

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
