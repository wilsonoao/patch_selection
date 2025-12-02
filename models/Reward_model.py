import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, use_bn=False, p_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else None

        # training 時不使用 dropout（p=0）
        self.dropout = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(hidden_dim, 2)  # two reward outputs

    def set_dropout_rate(self, p):
        """動態設定 dropout rate（inference 時使用）"""
        self.dropout.p = p

    def forward(self, x):
        h = self.fc1(x)
        if self.bn1 is not None:
            h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        r = self.fc2(h)
        return r