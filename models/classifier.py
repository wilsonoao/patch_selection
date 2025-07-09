import torch.nn as nn

class TwoLayerClassifier(nn.Module):
    def __init__(self, in_channel=768):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, a=0.01)  # 可換成其他如 normal_
        if m.bias is not None:
            init.constant_(m.bias, 0)