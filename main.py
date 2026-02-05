import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import train ,seed_torch, eval_baseline_test, eval_test, train_baseline
from utilmodule.clustering_method import clustering 
from utilmodule.significance_test import significance_test
from utils.utils import load_yaml_config
from torch.utils.data import DataLoader
from dataset.load_datasets import h5file_Dataset, GroupDataset 
from dataset.dataset_split import build_fold_csv 
import torch
import torch.nn as nn
import wandb
from models.ABMIL import ABMILPooling
from models.Classifier import MLP
import pickle
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


def Setup_group_loader(cluster_record, loader, batch_size=1, shuffle=True):
    
    bags_info = []
    for ide, (coords, data, label, name) in enumerate(tqdm(loader)):

        groups = cluster_record[name[0]]['groups']
        groups_feature = torch.cat([group.mean(dim=0, keepdim=True) for group in groups], dim=0) 
        groups_fack_coords = torch.zeros_like(groups_feature)

        bags_info.append((groups_fack_coords, groups_feature, label.item(), name[0]))

    group_dataset = GroupDataset(bags_info)
    group_loader = DataLoader(group_dataset, batch_size=batch_size, shuffle=shuffle)

    return group_loader
    
        
def main(args):

    wandb_token = "6c2e984aee5341ab06b1d26cefdb654ffea09bc7"
    save_dir = Path(args.save_dir) if args.save_dir != "" else None
    seed_torch(args.seed)

    feature_dir = Path(args.feature_dir)
    h5file_dir = Path(args.h5_dir)
    csv_file_dir = Path(os.path.join(args.csv_dir), "data") if args.csv_dir is not None and args.csv_dir != "" else Path(os.path.join(args.save_dir, "data"))
    cluster_pkl_path = Path(os.path.join(args.cluster_pkl_dir), "data", "cluster.pkl") if args.cluster_pkl_dir is not None and args.cluster_pkl_dir != "" else Path(os.path.join(args.save_dir, "data", "cluster.pkl"))

    # print(cluster_record)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.train:

        for fold_id in range(4):

            abmil = ABMILPooling(in_dim=args.in_channel, hidden_dim=args.hidden_dim).to(device)
            abmil_group = ABMILPooling(in_dim=args.in_channel, hidden_dim=args.hidden_dim).to(device)

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

            if not cluster_pkl_path.exists():
                clustering(save_path=cluster_pkl_path, train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader)


            with open(cluster_pkl_path, "rb") as f:
                cluster_record = pickle.load(f)

            # group dataloader
            group_train_loader = Setup_group_loader(cluster_record, train_loader)
            group_val_loader = Setup_group_loader(cluster_record, val_loader)
            group_test_loader = Setup_group_loader(cluster_record, test_loader)

            Runtime = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_groupConstraint_base_dir = Path(os.path.join(save_dir, "groupConstraint"))
            save_groupConstraint_dir = Path(os.path.join(save_groupConstraint_base_dir, f"dataset_fold_{fold_id}"))
            os.makedirs(save_groupConstraint_dir, exist_ok=True)
            if args.use_wandb: 
                wandb.login(key=wandb_token)
                project_name = Path(feature_dir).resolve()
                parents = project_name.parents

                def safe_parent_name(parents, idx, fallback="NA"):
                    return parents[idx].name if len(parents) > idx else fallback

                project = (
                    safe_parent_name(parents, 2)
                    + "_"
                    + safe_parent_name(parents, 1)
                    + "_GroupConstraintMIL_"
                    + save_dir.name
                )

                wandb.init(
                    project=project,
                    name=f"dataset_fold_{fold_id}_{Runtime}{args.log_name}",
                    config=dict(args),
                )
            else:
                print("No wandb")
                # wandb = None
            train(args, save_dir=save_groupConstraint_dir, abmil=abmil, group_abmil=abmil_group, train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, Group_train_loader=group_train_loader, Group_validation_loader=group_val_loader, Group_test_loader=group_test_loader, cluster_record=cluster_record, wandb=wandb)
            
            if args.use_wandb: 
                wandb.finish()


    save_groupConstraint_base_dir = Path(os.path.join(save_dir, "groupConstraint"))

    if args.eval_static:
        
        new_method_base_root = save_groupConstraint_base_dir if args.train else Path(os.path.join(args.test_dir), "groupConstraint") if args.test_dir is not None and args.test_dir != "" else Path(os.path.join(args.save_dir, "groupConstraint"))
        baseline_base_root = Path(os.path.join(args.save_dir, "baseline")) if args.train or (args.baseline_dir is not None and args.baseline_dir != "") else Path(os.path.join(args.baseline_dir), "baseline")

        # check probability csv exist
        for fold_id in range(4):

            csv_file_path = Path(os.path.join(csv_file_dir, f"dataset_fold_{fold_id}.csv"))

            # patch dataloader
            train_dataset = h5file_Dataset(csv_file_path,h5file_dir,feature_dir,'train')
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            validation_dataset = h5file_Dataset(csv_file_path,h5file_dir,feature_dir,'val')
            val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
            test_dataset = h5file_Dataset(csv_file_path,h5file_dir,feature_dir,'test')
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            new_method_root = Path(os.path.join(new_method_base_root, f"dataset_fold_{fold_id}"))
            path_new_probability = os.path.join(
                new_method_root, "probability.csv"
            )
            # eval model
            if not os.path.isfile(path_new_probability):
                # model
                abmil = ABMILPooling(in_dim=args.in_channel, hidden_dim=args.hidden_dim).to(device)
                # model
                abmil_weight = torch.load(os.path.join(new_method_root, f'abmil.pth'), map_location=device)
                abmil.load_state_dict(abmil_weight)
                
                eval_test(save_dir=new_method_root, model=abmil, test_loader=test_loader)
            
            baseline_root = Path(os.path.join(baseline_base_root, f"dataset_fold_{fold_id}"))
            path_baseline_probability = os.path.join(
                baseline_root, "probability.csv"
            )
            # eval model
            if not os.path.isfile(path_baseline_probability):
                

                classifier = MLP(in_channel=args.in_channel)
                args.save_dir = baseline_root
                if not os.path.isfile(os.path.join(baseline_root, "baseline.pth")):
                    Runtime = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_name = f"dataset_fold_{fold_id}"
                    os.makedirs(baseline_root, exist_ok=True)
                    if args.use_wandb: 
                        wandb.login(key=wandb_token)
                        project_name = Path(feature_dir).resolve()
                        parents = project_name.parents
                        def safe_parent_name(parents, idx, fallback="NA"):
                            return parents[idx].name if len(parents) > idx else fallback

                        project = (
                            safe_parent_name(parents, 2)
                            + "_"
                            + safe_parent_name(parents, 1)
                            + "_baseline_"
                            + save_dir.name
                        )
                        wandb.init(
                            project=project,
                            name=f"dataset_fold_{fold_id}_{Runtime}{args.log_name}",
                            config=dict(args),
                        )

                    else:
                        print("No wandb")
                    train_baseline(args=args, save_dir=baseline_root, model=classifier, train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, wandb=wandb)
                    if args.use_wandb: 
                        wandb.finish()
                baseline_weight = torch.load(os.path.join(baseline_root, f'basemodel.pth'), map_location=device)
                classifier.load_state_dict(baseline_weight)
                
                eval_baseline_test(save_dir=baseline_root, model=classifier, test_loader=test_loader)

        significance_test(new_method_root=new_method_base_root, baseline_root=baseline_base_root, out_dir=save_dir)

        
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
    