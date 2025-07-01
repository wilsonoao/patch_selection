import os
import sys
 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import test ,seed_torch
from torch.utils.data import DataLoader
from datasets.load_datasets import h5file_Dataset
import torch
import numpy as np
from utilmodule.createmode import create_model
import pandas as pd





def main(args):
 
    seed_torch(2021)
    res_list = []
    
    basedmodel,ppo,classifymodel,memory,FusionHisF = create_model(args)
    data_csv_dir = args.csv
    feature_dir = args.feature_dir
    h5file_dir = args.train_h5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    basedmodel_weight = torch.load(os.path.join(args.test_dir, 'basedmodel.pth'), map_location=device)
    basedmodel.load_state_dict(basedmodel_weight)

    FusionHisF_weight = torch.load(os.path.join(args.test_dir, 'FusionHisF.pth'), map_location=device)
    FusionHisF.load_state_dict(FusionHisF_weight)

    classifymodel_weight = torch.load(os.path.join(args.test_dir, 'classifymodel.pth'), map_location=device)
    classifymodel.load_state_dict(classifymodel_weight)

    ppo_weight = torch.load(os.path.join(args.test_dir, 'ppo.pth'), map_location=device)
    ppo.policy.load_state_dict(ppo_weight)

    basedmodel.eval()
    classifymodel.eval()
    FusionHisF.eval()
    ppo.policy.eval() 
    
    val_dataset = h5file_Dataset(data_csv_dir,h5file_dir,feature_dir,'val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataset = h5file_Dataset(data_csv_dir,h5file_dir,feature_dir,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("val")
    precision, recall, f1, auc, accuracy = test(args,basedmodel,ppo,classifymodel,FusionHisF,memory,val_dataloader )

    print("test")
    precision, recall, f1, auc, accuracy = test(args,basedmodel,ppo,classifymodel,FusionHisF,memory,test_dataloader )
    





if __name__ == "__main__":

    args = make_parse()
    main(args)
