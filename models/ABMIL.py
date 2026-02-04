import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMILPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim)
        self.U = nn.Linear(in_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, feats):
        """
        feats: [B, L, D]
        return: pooled [B, D]
        """
        output = {}
        # attention score before mask: [B, L, 1]
        H = torch.tanh(self.V(feats)) * torch.sigmoid(self.U(feats))
        A = self.w(H).squeeze(-1)  # [B, L]
        # print(A.shape)

        # softmax over L dimension
        output["A"] = torch.softmax(A, dim=1)  # [B, L]
        
        # weighted sum
        output["BagEmbedding"] = torch.bmm(output["A"].unsqueeze(1), feats).squeeze(1)  # [B, D]
        output["logits"] = self.classifier(output["BagEmbedding"])

        
        return output
