import torch
import torch.nn as nn

hidden_units = 2**7
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fn_approx = nn.Sequential(
            nn.Linear(4,hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units,hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units,hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units,2)
        )

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, y, c1, c2):
        x_combined = torch.cat((x, y, c1, c2),dim=4)
        logits = self.fn_approx(x_combined)
        return logits
    
model = NeuralNetwork()


# Initialize the model
model = NeuralNetwork()

# Load the state dict previously saved
state_dict = torch.load("schrodinger_model.pth")

# Load the state dict to the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()
