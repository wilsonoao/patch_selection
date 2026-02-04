import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, in_channel=768):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)