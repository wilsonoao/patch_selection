import torch.nn as nn
import torch.nn.init as init


class TwoLayerClassifier(nn.Module):
    def __init__(self, in_channel=768):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0.01)
        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        
        init.kaiming_uniform_(self.fc2.weight, a=0.01)
        if self.fc2.bias is not None:
            init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)