import torch
from torch import nn

class PolicyNetwork(nn.Module):
    
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=4):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LogSoftmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)


class ValueNetwork(nn.Module):
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
        super(ValueNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        return self.model(x)
