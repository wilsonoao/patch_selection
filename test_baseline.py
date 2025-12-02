import os
import sys
 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import train_baseline ,seed_torch, test_baseline
from torch.utils.data import DataLoader
from datasets.load_datasets import h5file_Dataset
import torch
import torch.nn as nn
import numpy as np
from utilmodule.createmode import create_model
import pandas as pd
from models.WSI_baseline import WSI_classifier
from models.CHIEF import CHIEF
from CHIEF_network import ClfNet


class TwoLayerClassifier(nn.Module):
    def __init__(self, in_channel=768, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def variable_length_collate_fn(batch):
    coords, chief_data, giga_data, label, _ = zip(*batch)  # 分別是 tuple of Tensors
    return list(coords), list(chief_data), list(giga_data), torch.tensor(label), None


def main(args):
 
    seed_torch(2021)
    res_list = []
    
    classifymodel = TwoLayerClassifier(in_channel=1024)
    data_csv_dir = args.csv
    chief_feature_dir = args.chief_feature_dir
    gigapath_feature_dir = args.gigapath_feature_dir
    h5file_dir = args.train_h5

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifymodel = classifymodel.to(device)

    classifymodel_weight = torch.load(os.path.join(args.test_dir, 'classifymodel.pth'), map_location=device)
    classifymodel.load_state_dict(classifymodel_weight)


    #train_dataset = h5file_Dataset(data_csv_dir,h5file_dir,feature_dir,'train')
    #train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_dataset = h5file_Dataset(data_csv_dir,h5file_dir,chief_feature_dir, gigapath_feature_dir,'val')
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True,collate_fn=variable_length_collate_fn)
    test_dataset = h5file_Dataset(data_csv_dir,h5file_dir,chief_feature_dir, gigapath_feature_dir,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn=variable_length_collate_fn)

    # print("Validaion\n")
    # test_baseline(args,None,None,classifymodel,None,None,validation_dataloader)

    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()
    
    print("Test\n")
    test_baseline(args,chief_model,classifymodel,None,None,test_dataloader)



if __name__ == "__main__":

    args = make_parse()
    main(args)
