import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
# from gigapath import slide_encoder
# from gigapath.pipeline import run_inference_with_slide_encoder
from huggingface_hub import login
login("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")
 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.createmode import create_model
from utilmodule.core import train ,seed_torch
from torch.utils.data import DataLoader, random_split
from datasets.load_datasets import h5file_Dataset
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import wandb
from models.Reward_model import RewardMLP 
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

class TwoLayerClassifier(nn.Module):
    def __init__(self, in_channel=768, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)   # 新增 Dropout，預設 p=0.5
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))          # 加在 ReLU 後面
        return self.fc2(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, a=0.01)  # 可換成其他如 normal_
        if m.bias is not None:
            init.constant_(m.bias, 0)


def main(args):
 
    seed_torch(args.seed)

    chief_ppo, chief_memory = create_model(args)
    gigapath_ppo, gigapath_memory = create_model(args)

    data_csv_dir = args.csv
    chief_feature_dir = args.chief_feature_dir
    gigapath_feature_dir = args.gigapath_feature_dir
    h5file_dir = args.train_h5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier_chief = RewardMLP(input_dim=768, hidden_dim=1024, use_bn=False, p_dropout=None).to(device)
    classifier_chief.apply(init_weights)

    classifier_giga = TwoLayerClassifier().to(device)
    classifier_giga.apply(init_weights)

    chief_rewardModels = RewardMLP(input_dim=768, hidden_dim=1024, use_bn=False, p_dropout=None).to(device)
    gigapath_rewardModels = [RewardMLP(input_dim=1024, hidden_dim=1024, use_bn=False, p_dropout=0.5).to(device), RewardMLP(input_dim=768, hidden_dim=512, use_bn=False, p_dropout=0.3).to(device)]

    train_dataset = h5file_Dataset(data_csv_dir,h5file_dir,chief_feature_dir, gigapath_feature_dir,'train')
    train_all_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # 各自建立 DataLoader
    validation_dataset = h5file_Dataset(data_csv_dir,h5file_dir,chief_feature_dir, gigapath_feature_dir,'val')
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_dataset = h5file_Dataset(data_csv_dir,h5file_dir,chief_feature_dir, gigapath_feature_dir,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    run_name = f"{args.csv.split('/')[-1].split('.')[0]}"
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    wandb.login(key="6c2e984aee5341ab06b1d26cefdb654ffea09bc7")
    wandb.init(
        project=args.csv.split('/')[-3]+"_SingleCHIEF_NoisetwoModel_RNNagent"+args.save_dir.split("/")[-1],      # 可以在網站上看到
        name=run_name+"_intermediate",      # optional，可用於區分實驗
        config=vars(args)                    # optional，紀錄一些超參數
    )

    train(args,chief_ppo, gigapath_ppo, classifier_chief, classifier_giga, chief_rewardModels, gigapath_rewardModels, None, chief_memory, gigapath_memory, train_all_dataloader, validation_dataloader, test_dataloader, wandb)

if __name__ == "__main__":

    args = make_parse()
    main(args)