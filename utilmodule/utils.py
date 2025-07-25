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


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='tcga',type=str)  
    parser.add_argument('--mode', default='rlselect',type=str)
    parser.add_argument('--seed', default=2021,type=int)
    parser.add_argument('--num_epochs', default=300,type=int)
    parser.add_argument('--lr', default=0.00001,type=int)
    

    parser.add_argument('--in_chans', default=1024,type=int)
  
    parser.add_argument('--embed_dim', default=768,type=int)
    parser.add_argument('--attn', default='normal',type=str)
    parser.add_argument('--gm', default='cluster',type=str)
    parser.add_argument('--cls', default=True,type=bool)
    parser.add_argument('--num_msg', default=1,type=int)
    parser.add_argument('--ape', default=True,type=bool)
    parser.add_argument('--n_classes', default=2,type=int)
    parser.add_argument('--num_layers', default=2,type=int) 

    parser.add_argument('--instaceclass', default=True,type=bool,help='') 
    parser.add_argument('--CE_CL', default=True,type=bool,help='')
    parser.add_argument('--ape_class', default=False,type=bool,help='') 


    parser.add_argument('--test_h5', default='/work/data/TCGA-LUAD-FS/CHIEF/20X/h5_files(stain_norm)',type=str)
    parser.add_argument('--train_h5',default='/work/data/TCGA-LUAD-FS/CHIEF/20X/h5_files(stain_norm)',type=str)
    parser.add_argument('--csv', default='/work/PAMIL_GIGAPATH_CHIEF/4_fold/LUAD/CSMD3/dataset_fold_0.csv',type=str)
    parser.add_argument('--chief_feature_dir', default='/work/data/TCGA-LUAD-FS/CHIEF/20X/pt_files(stain_norm)',type=str)
    parser.add_argument('--gigapath_feature_dir', default='/work/data/TCGA-LUAD-FS/GIGAPATH/20X/pt_files(stain_norm)',type=str)
 
    
    parser.add_argument('--policy_hidden_dim', type=int, default=768)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--state_dim', type=int, default=768)
    parser.add_argument('--action_size', type=int, default=30) 
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    
    parser.add_argument('--test_total_T', type=int, default=3)
    parser.add_argument('--train_total_T', type=int, default=3)

    parser.add_argument('--reward_rule', type=str, default="cl",help=' ')
    parser.add_argument('--save_dir', type=str, default="/work/result_two_round_60/result_two_round_60/CSMD3",help='')
    parser.add_argument('--patience', type=int, default=20, help = '')

    parser.add_argument('--test_dir', default="/work/PAMIL_GIGAPATH_CHIEF/test",help='')

    
    
    args = parser.parse_args()
    return args

import torch
import torch.nn.functional as F


def calculate_metrics(targets, probs):
    threshold = 0.5  # You can adjust this threshold as needed
    predictions = (probs[:, 1] >= threshold).astype(int)
 
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions) 
    auc = roc_auc_score(targets, probs[:, 1])  
    accuracy = accuracy_score(targets, predictions)
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"auc: {auc}")
    print(f"accuracy: {accuracy}")
    print(f"f1: {f1}")
    print("confusion matrix:")
    print(confusion_matrix(targets, predictions))

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
