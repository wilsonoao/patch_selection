from torch import Tensor
import torch
import torch.nn as nn
import torchvision.models as models
import random
import loralib as lora

class DropoutOrIdentity(nn.Module):
    def __init__(self, p):
        super(DropoutOrIdentity, self).__init__()
        if p is None or p == 0:
            self.layer = nn.Identity()
        else:
            self.layer = nn.Dropout(p)
    def forward(self, x):
        return self.layer(x)
class MILAttention(nn.Module):
    def __init__(self, featureLength = 768, featureInside = 256,dropout=None):
        '''
        Parameters:
            featureLength: Length of feature passed in from feature extractor(encoder)
            featureInside: Length of feature in MIL linear
            dropout: dropout rate. If None, no dropout
        Output: tensor
            weight of the features
        '''
        super(MILAttention, self).__init__()
        self.featureLength = featureLength
        self.featureInside = featureInside

        self.attention_V = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias=True),
            nn.Tanh(),
            DropoutOrIdentity(dropout)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias=True),
            nn.Sigmoid(),
            DropoutOrIdentity(dropout)
        )
        self.attention_weights = nn.Linear(self.featureInside, 1, bias=True)
        self.softmax_0 = nn.Softmax(dim=0)
        self.softmax_1 = nn.Softmax(dim=1)

    def forward(self, x: Tensor, nonpad = None) -> Tensor:
        bz, pz, fz = x.shape if len(x.shape) == 3 else (1, *x.shape)
        # x = x.view(bz*pz, fz)
        att_v = self.attention_V(x)
        # print("att_v", att_v.shape)
        att_u = self.attention_U(x)
        # print("att_u", att_u.shape)
        att_v = att_v.view(bz * pz, -1)
        att_u = att_u.view(bz * pz, -1)

        att = self.attention_weights(att_u * att_v)
        # print("att", att.shape)
        weight = att.view(bz, pz, 1)

        if nonpad is not None:
            for idx, i in enumerate(weight):
                weight[idx][:nonpad[idx]] = self.softmax_0(weight[idx][:nonpad[idx]])
                weight[idx][nonpad[idx]:] = 0
        else:
            weight = self.softmax_1(att)
        weight = weight.view(bz, 1, pz)

        return weight


class MILNet(nn.Module):
    def __init__(self, featureLength = 768, linearLength = 256, dropout=None):
        '''
        Parameters:
            featureLength: Length of feature from resnet18
            linearLength:  Length of feature for MIL attention
        Forward:
            weight sum of the features
        '''
        super(MILNet, self).__init__()
        flatten = nn.Flatten(start_dim = 1)
        fc = nn.Linear(512, featureLength, bias=True)

        self.attentionlayer = MILAttention(featureLength, linearLength,dropout=dropout)

    def forward(self, x, lengths):
        if len(x.shape) == 2:
            batch_size = 1
            num_patches, feature_dim = x.shape
        else:
            batch_size, num_patches, feature_dim = x.shape

        weight = self.attentionlayer(x, lengths)
        # print(weight)
        x = x.view(batch_size * num_patches, -1)
        weight = weight.view(batch_size * num_patches, 1)
        x = weight * x
        x = x.view(batch_size, num_patches, -1)
        x = torch.sum(x, dim=1)

        return x

class ClfNet(nn.Module):
    def __init__(self, featureLength=768, classes=2, dropout=None):
        super(ClfNet, self).__init__()

        self.featureExtractor = MILNet(featureLength=featureLength,dropout=None)
        # self.featureLength = featureLength
        self.fc_target = nn.Sequential(
            nn.Linear(featureLength, 256, bias=True),
            DropoutOrIdentity(dropout),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128, bias=True),
            DropoutOrIdentity(dropout),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, classes, bias=True),
        )

    def forward(self, x, lengths=None):
        #features = self.featureExtractor(x.squeeze(0), lengths)
        features = self.featureExtractor(x, lengths)
        preds = self.fc_target(features)
        return preds

class MLP(nn.Module):
    def __init__(self, featureLength=768, classes=2, ft=False,dropout=None):
        super(MLP, self).__init__()

        # self.featureLength = featureLength
        self.fc_target = nn.Sequential(
            nn.Linear(featureLength, 256, bias=True),
            DropoutOrIdentity(dropout),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128, bias=True),
            DropoutOrIdentity(dropout),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, classes, bias=True),
        )

    def forward(self, features, race, lengths=None):
        # features = self.featureExtractor(x.squeeze(0), lengths)
        preds = self.fc_target(features)
        return preds

class WeibullModel(nn.Module):
    def __init__(self, featureLength=768, dropout=None):
        super(WeibullModel, self).__init__()
        self.featureExtractor = MILNet(featureLength=768,dropout=dropout)
        self.featureLength = featureLength
        self.fc_target = nn.Sequential(
            nn.Linear(featureLength, 256, bias=True),
            nn.ReLU(),
            DropoutOrIdentity(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            DropoutOrIdentity(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, 2, bias=True),
        )
        self.softplus = nn.Softplus()
        nn.init.xavier_uniform_(self.fc_target[0].weight)

    def forward(self, x, lengths):
        features = self.featureExtractor(x.squeeze(0), lengths)
        x = self.fc_target(features)
        shape_scale = self.activate(x)
        return shape_scale

    def activate(self, x):
        a = torch.exp(x[:, 0])
        b = self.softplus(x[:, 1])
        a = torch.reshape(a, (a.size()[0], 1))
        b = torch.reshape(b, (b.size()[0], 1))
        return torch.cat((a, b), dim=1)
