import os
import sys
from gigapath import slide_encoder
from gigapath.pipeline import run_inference_with_slide_encoder
from models.CHIEF import CHIEF
from CHIEF_network import ClfNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import test ,seed_torch
from torch.utils.data import DataLoader
from datasets.load_datasets import h5file_Dataset
import torch
import numpy as np
from utilmodule.createmode import create_model
import pandas as pd
from models.classifier import TwoLayerClassifier

def main(args):
 
    seed_torch(2021)
    res_list = []
    
    basedmodel,ppo,_,memory,FusionHisF = create_model(args)
    data_csv_dir = args.csv
    chief_feature_dir = args.chief_feature_dir
    gigapath_feature_dir = args.gigapath_feature_dir
    h5file_dir = args.train_h5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_weight = torch.load(os.path.join(args.test_dir, 'ppo.pth'), map_location=device)
    ppo.policy.load_state_dict(ppo_weight)
    
    classifier_chief = TwoLayerClassifier().to(device)
    classifier_chief_weight = torch.load(os.path.join(args.test_dir, 'classifier_chief.pth'), map_location=device)
    classifier_chief.load_state_dict(classifier_chief_weight)

    classifier_giga = TwoLayerClassifier().to(device)
    classifier_giga_weight = torch.load(os.path.join(args.test_dir, 'classifier_gigapath.pth'), map_location=device)
    classifier_giga.load_state_dict(classifier_giga_weight)

    test_dataset = h5file_Dataset(data_csv_dir,h5file_dir,chief_feature_dir, gigapath_feature_dir,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # gigapath wsi model
    gigapath_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536).to(device)
    gigapath_model.eval()

    # chief wsi model
    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()

    classifier_chief.eval()
    classifier_giga.eval()
    ppo.policy.eval()
    
    # print("val")
    # precision, recall, f1, auc, accuracy = test(args,basedmodel,ppo,classifymodel,FusionHisF,memory,val_dataloader )

    # print("test")
    precision, recall, f1, auc, accuracy = test(args,ppo,classifier_chief, classifier_giga,memory,test_dataloader, chief_model, gigapath_model, "test", epoch=0, wandb=None, run_time_test=False, record_csv=True)
    


if __name__ == "__main__":

    args = make_parse()
    main(args)
