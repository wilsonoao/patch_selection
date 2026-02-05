import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F

import os
# import re
# import csv
# import yaml
# import json
# import glob
import shutil
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,accuracy_score, confusion_matrix
import heapq
import statistics
import matplotlib.pyplot as plt


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42,type=int)
    parser.add_argument('--num_epochs', default=300,type=int)
    parser.add_argument('--lr', default=0.00005,type=float)
    parser.add_argument('--group_lr', default=0.00005,type=float)
    parser.add_argument('--n_classes', default=2,type=int)
    parser.add_argument('--in_channel', default=768,type=int)
    parser.add_argument('--hidden_dim', default=256,type=int)

    parser.add_argument('--theta_start', default=2,type=float)
    parser.add_argument('--theta_end', default=1,type=float)
    parser.add_argument('--k', default=4,type=float)
    parser.add_argument('--dirichlet_weight', default=8,type=float)
    parser.add_argument('--log_name', default="",type=str)


    parser.add_argument('--h5_dir',default=None,type=str)
    parser.add_argument('--csv_dir', default=None,type=str)
    parser.add_argument('--feature_dir', default=None,type=str)
    parser.add_argument('--clinical_pkl_path', default=None,type=str)
    parser.add_argument('--cluster_pkl_dir', default=None,type=str)
    parser.add_argument('--save_dir', type=str, default=None,help='')
    parser.add_argument('--csv_saveName', type=str, default=None,help='')
    parser.add_argument('--test_dir', default=None,help='',type=str)
    parser.add_argument('--baseline_dir', default=None,help='',type=str)
    
    parser.add_argument('--patience', type=int, default=10, help = '')


    parser.add_argument('--train', default=None)
    parser.add_argument('--use_wandb', default=None)
    parser.add_argument('--eval_static', default=None)
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config file",
    )
    args = parser.parse_args()
    return args

import torch
import torch.nn.functional as F

def pred_label_process(memory, true_label, cls_pred):

    """
    Args:
        token_preds: Tensor of shape [T], 每輪 token 預測值（如 sigmoid 後機率）
        cls_pred: float or Tensor scalar, CLS token 的預測值（如 sigmoid output）
        true_label: int or scalar Tensor, ground truth label (0 or 1)
    
    Returns:
        final_pred: float, 最終融合預測值
    """
    T = len(memory.results_dict)
    pred = [results_dict['Y_prob'][0][true_label].item() for results_dict in memory.results_dict]
    topk = heapq.nlargest(min(T, 5), pred)  # 前 K 大值
    max_pred = topk[0]
    avg_top3 = statistics.mean(topk[:3])
    avg_top5 = statistics.mean(topk[:5])

    if true_label == 1:
        final_pred = (max_pred + avg_top3 + avg_top5 + cls_pred) / 4
    else:
        final_pred = cls_pred

    return final_pred


def calculate_metrics(targets, probs):
    threshold = 0.5  # You can adjust this threshold as needed
    predictions = (probs[:, 1] >= threshold).astype(int)
 
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions) 
    auc = roc_auc_score(targets, probs[:, 1])  
    accuracy = accuracy_score(targets, predictions)
    # print(f"precision: {precision}")
    # print(f"recall: {recall}")
    # print(f"auc: {auc}")
    # print(f"accuracy: {accuracy}")
    # print(f"f1: {f1}")
    # print("confusion matrix:")
    # print(confusion_matrix(targets, predictions))

    return precision, recall, f1, auc, accuracy

 

def cat_msg2cluster_group(x_groups,msg_tokens):
    x_groups_cated = []
    for x in x_groups:  
        x = x.unsqueeze(dim=0)  
        try:
            temp = torch.cat((msg_tokens,x),dim=2)
        except Exception as e:
            print('Error when cat msg tokens to sub-bags')
        x_groups_cated.append(temp)

    return x_groups_cated



def plot_group_attn_magnitude_distribution(
    attn_list,
    bins=None,
    save_path="group_attn_magnitude.png",
    title="Group Attention Magnitude Distribution",
):
    """
    attn_list: list[Tensor] or Tensor (B, N)
    """

    if bins is None:
        bins = [
            (1.0, 1e-1),
            (1e-1, 1e-2),
            (1e-2, 1e-3),
            (1e-3, 1e-4),
            (1e-4, 0.0),
        ]

    # ---- collect all attention ----
    if isinstance(attn_list, torch.Tensor):
        x = attn_list.detach().flatten().abs()
    else:
        x = torch.cat([a.detach().flatten().abs() for a in attn_list], dim=0)

    labels, counts = [], []

    for high, low in bins:
        mask = (x <= high) & (x > low)
        counts.append(mask.sum().item())
        labels.append(f"{high:g} ~ {low:g}")

    # ---- plot ----
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, counts)

    plt.xlabel("Value Range")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45)

    # ---- annotate value on each bar ----
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def attention_entropy_norm(A, eps=1e-12):
    """
    A: attention tensor, shape [1, N] or [N]
    return: scalar H_norm
    """
    if A.dim() == 2:
        A = A.squeeze(0)   # [N]

    A = A.clamp(min=eps)
    N = A.numel()

    entropy = -torch.sum(A * torch.log(A))
    H_norm = entropy / torch.log(torch.tensor(N, device=A.device, dtype=A.dtype))

    return H_norm



def attention_entropy(A, eps=1e-12):
    """
    A: attention tensor, shape [1, N] or [N]
    return: scalar H_norm
    """
    if A.dim() == 2:
        A = A.squeeze(0)   # [N]

    A = A.clamp(min=eps)
    N = A.numel()

    entropy = -torch.sum(A * torch.log(A))
    

    return entropy


def chief_wsi_embedding(chief_model, feature, type="model"):

    if type == "model":
        anatomical=13
        with torch.no_grad():
            x,tmp_z = feature,anatomical
            result = chief_model(x, torch.tensor([tmp_z]))
            wsi_feature_emb = result['WSI_feature']  ###[1,768]
            # print(wsi_feature_emb.size())
    elif type == "mean-pooling":
        wsi_feature_emb = feature.mean(dim=0, keepdim=True)
    elif type == "max-pooling":
        wsi_feature_emb, _ = feature.max(dim=0, keepdim=True)

    return wsi_feature_emb