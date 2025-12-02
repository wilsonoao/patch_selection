import os
import sys
# from gigapath import slide_encoder
# from gigapath.pipeline import run_inference_with_slide_encoder
from models.CHIEF import CHIEF
from CHIEF_network import ClfNet
from models.Reward_model import RewardMLP 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import test ,seed_torch, cluster_chief_data
from torch.utils.data import DataLoader
from datasets.load_datasets import h5file_Dataset
import torch
import numpy as np
from utilmodule.createmode import create_model
import pandas as pd
from models.classifier import TwoLayerClassifier
import pickle

def main(args):
 
    seed_torch(args.seed)
    res_list = []
    
    chief_ppo, chief_memory = create_model(args)
    gigapath_ppo, gigapath_memory = create_model(args)

    data_csv_dir = args.csv
    chief_feature_dir = args.chief_feature_dir
    gigapath_feature_dir = args.gigapath_feature_dir
    h5file_dir = args.train_h5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chief_ppo_weight = torch.load(os.path.join(args.test_dir, 'ppo_chief.pth'), map_location=device)
    chief_ppo.policy_old.load_state_dict(chief_ppo_weight)

    # gigapath_ppo_weight = torch.load(os.path.join(args.test_dir, 'ppo_gigapath.pth'), map_location=device)
    # gigapath_ppo.policy_old.load_state_dict(gigapath_ppo_weight)
    
    classifier_chiefs = [RewardMLP(input_dim=768, hidden_dim=1024, use_bn=False, p_dropout=0).to(device) for _ in range(1)]
    for i, classifier_chief  in enumerate(classifier_chiefs):
        classifier_chief_weight = torch.load(os.path.join(args.test_dir, f'classifier_chief.pth'), map_location=device)
        classifier_chief.load_state_dict(classifier_chief_weight)

    classifier_giga = TwoLayerClassifier().to(device)
    # classifier_giga_weight = torch.load(os.path.join(args.test_dir, 'classifier_gigapath.pth'), map_location=device)
    # classifier_giga.load_state_dict(classifier_giga_weight)

    test_dataset = h5file_Dataset(data_csv_dir,h5file_dir,chief_feature_dir, gigapath_feature_dir,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # # gigapath wsi model
    # gigapath_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536).to(device)
    # gigapath_model.eval()

    # chief wsi model
    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()

    classifier_chief.eval()
    classifier_giga.eval()
    gigapath_ppo.policy.eval()
    chief_ppo.policy_old.eval()
    
    # print("val")
    # precision, recall, f1, auc, accuracy = test(args,basedmodel,ppo,classifymodel,FusionHisF,memory,val_dataloader )

    # print("test")
    pkl_path = os.path.join(args.chief_feature_dir, "cluster_record_bisect.pkl")
    with open(pkl_path, "rb") as f:
        cluster_list_test = pickle.load(f)
    test(args,chief_ppo,classifier_chiefs,chief_memory, cluster_list_test, test_dataloader, chief_model, "test", epoch=0, wandb=None, run_time_test=False, record_csv=True)

if __name__ == "__main__":

    args = make_parse()
    main(args)
