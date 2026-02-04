import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import seed_torch, eval_baseline_test, train_baseline
from torch.utils.data import DataLoader
from dataset.load_datasets import h5file_Dataset
from dataset.dataset_split import build_fold_csv 
from utils.utils import load_yaml_config
import torch
import torch.nn as nn
import wandb
from models.Classifier import MLP
from pathlib import Path
from datetime import datetime

    
        
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_token = "6c2e984aee5341ab06b1d26cefdb654ffea09bc7"
    save_dir = Path(args.save_dir) if args.save_dir != "" else None
    seed_torch(args.seed)

    csv_file_dir = Path(args.csv_dir)
    feature_dir = Path(args.feature_dir)
    h5file_dir = Path(args.h5_dir)

    for fold_id in range(4):

        csv_file_path = Path(os.path.join(csv_file_dir, f"dataset_fold_{fold_id}.csv"))
        if not csv_file_path.exists():
            build_fold_csv(
                feature_dir=feature_dir,
                clinical_pkl_path=args.clinical_pkl_path,
                out_dir=csv_file_dir,
                random_state=args.seed,
            )
        else:
            print("exist !!!")

        # patch dataloader
        train_dataset = h5file_Dataset(csv_file_path,h5file_dir,feature_dir,'train')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        validation_dataset = h5file_Dataset(csv_file_path,h5file_dir,feature_dir,'val')
        val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
        test_dataset = h5file_Dataset(csv_file_path,h5file_dir,feature_dir,'test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        Runtime = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_baseline_base_dir = Path(os.path.join(save_dir, "baseline"))
        save_baseline_dir = Path(os.path.join(save_baseline_base_dir, f"dataset_fold_{fold_id}"))
        os.makedirs(save_baseline_dir , exist_ok=True)
        
                
        classifier = MLP(in_channel=args.in_channel)
        if args.use_wandb: 
            wandb.login(key=wandb_token)
            wandb.init(
                project=save_dir.parents[2].name+"_"+save_dir.parents[1].name+"_baseline_"+save_dir.name,            
                name=f"dataset_fold_{fold_id}"+ "_" + Runtime + args.log_name,     
                config=vars(args)                    
            )
        else:
            print("No wandb")

        train_baseline(args=args, save_dir=save_baseline_dir, model=classifier, train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, wandb=wandb)

        baseline_weight = torch.load(os.path.join(save_baseline_dir, f'basemodel.pth'), map_location=device)
        classifier.load_state_dict(baseline_weight)
        
        eval_baseline_test(save_dir=save_baseline_dir, model=classifier, test_loader=test_loader)
        if args.use_wandb: 
            wandb.finish()

        
if __name__ == "__main__":

    cli_args = make_parse()              # argparse.Namespace
    args = load_yaml_config(cli_args.config)   # addict.Dict (YAML only)

    # ⭐ 關鍵：記住 YAML 原始定義的 key
    yaml_keys = set(args.keys())

    for k, v in vars(cli_args).items():
        if k == "config":
            continue
        if v is None:
            continue

        # YAML already defines it → do NOT override
        if k in yaml_keys:
            continue

        setattr(args, k, v)

    main(args)
    