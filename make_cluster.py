import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import seed_torch
from utilmodule.clustering_method import clustering 
from torch.utils.data import DataLoader
from utils.utils import load_yaml_config
from dataset.load_datasets import h5file_Dataset
from dataset.dataset_split import build_fold_csv 
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
    
        
def main(args):

    seed_torch(args.seed)

    csv_file_dir = Path(os.path.join(args.csv_dir), "data") if args.csv_dir is not None and args.csv_dir != "" else Path(os.path.join(args.save_dir, "data"))
    feature_dir = Path(args.feature_dir)
    h5file_dir = Path(args.h5_dir)
    cluster_pkl_path = Path(os.path.join(args.cluster_pkl_dir, "data", "cluster.pkl")) if args.cluster_pkl_dir is not None and args.cluster_pkl_path != "" else Path(os.path.join(args.save_dir, "data", "cluster.pkl"))

    csv_file_path = Path(os.path.join(csv_file_dir, "dataset_fold_0.csv"))
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

    clustering(save_path=cluster_pkl_path, train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader)

        
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
    