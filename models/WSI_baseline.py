import torch
import torch.nn as nn
import torch.nn.functional as F

class WSI_classifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=2):
        super(WSI_classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一層 + ReLU
        x = self.fc2(x)          # 第二層，沒有 activation，適用於分類 logits 輸出
        return x

