import torch.nn as nn
import torch.nn.functional as F
import torch
from utilmodule.utils import calculate_metrics, attention_entropy_norm, attention_entropy, chief_wsi_embedding
from utilmodule.loss import MILLoss
from models.CHIEF import CHIEF
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import os
from tqdm import tqdm
import pandas as pd
import copy

def test(args, cluster_record, abmil, loss_fn, test_loader, run_type="test", group_model=None, epoch=0, wandb=None, run_time_test=True, see_inside=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        group_list = []
        
        loss_record = {
            "attn_Hnorm": 0,
            "attn_H": 0,
        }

        for idx, (_, data, label, name) in enumerate (tqdm(test_loader)):
            
            # if name[0] != "TCGA-78-7152-01A-01-BS1.17987afb-41a5-44ff-94e2-a282163a10be.pt":
            #     continue
            # print(name[0])
            group = cluster_record[name[0]]["groups"]
            group_means = [
                g.mean(dim=0, keepdim=True)  # (1, D)
                for g in group
            ]
            group_feature = torch.cat(group_means, dim=0).unsqueeze(0).to(device)
            data = data.to(device)
            label = label.to(device).long()
            output = abmil(data)
            if isinstance(group_model, torch.nn.Module):
                groups = cluster_record[name[0]]["groups"]
                group_ids = cluster_record[name[0]]["id"]
                groups_feature = torch.cat([group.mean(dim=0, keepdim=True) for group in groups], dim=0).unsqueeze(0).to(device)
                with torch.no_grad():
                    group_out = group_model(groups_feature)
                loss = loss_fn(output["logits"], 
                    label,
                    attn=output["A"],
                    group_attn=group_out["A"].squeeze(0),
                    group_ids=group_ids
                )
            else:
                loss = loss_fn(output["logits"], label)
            ###
            # loss["loss"] -= 1e-4 * attention_entropy(output["A"])
            probs = F.softmax(output["logits"] , dim=1)
            loss_record["attn_Hnorm"] += attention_entropy_norm(output["A"]).item()
            loss_record["attn_H"] += attention_entropy(output["A"]).item()
            # record
            Y_prob_list.append(probs.detach().cpu())
            label_list.append(label.detach().cpu())

            for loss_type, loss in loss.items():
                if loss_type.startswith("log"):
                    continue
                elif loss_type not in loss_record:
                    loss_record[loss_type] = loss.item()
                else:
                    loss_record[loss_type] += loss.item()

        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        # 預測 label
        preds = np.argmax(probs, axis=1)

        if run_time_test and args.use_wandb:
            log_metrics(
                epoch+1,
                label_list,
                Y_prob_list,
                loss_record,
                test_loader,
                calculate_metrics,
                wandb,
                prefix=run_type,
            )


            
    return auc


def eval_test(save_dir, model, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        

        for idx, (_, data, label, name) in enumerate (tqdm(test_loader)):
            
            data = data.to(device)
            label = label.to(device).long()
            output = model(data)
            probs = F.softmax(output["logits"] , dim=1)
            # record
            Y_prob_list.append(probs.detach().cpu())
            label_list.append(label.detach().cpu())


        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        # 預測 label
        preds = np.argmax(probs, axis=1)
        correct = (preds == targets).astype(int)

        df = pd.DataFrame({
            'label': targets,
            'pred': preds,
            'correct': correct,
            'prob': probs[:, 1],
        })
        df.to_csv(os.path.join(save_dir, "probability.csv"), index=False)

        df_static = pd.DataFrame({
            'auc': [auc],
            'acc': [accuracy],
            'f1': [f1],
            'recall': [recall],
            'precision': [precision],
        })
        df_static.to_csv(os.path.join(save_dir, "static.csv"), index=False)



def train_loop(model, optimizer, loss_fn, loader, group_model=None, cluster_record=None):
    
    device = next(model.parameters()).device

    Y_prob_list = []
    label_list = []
    loss_record = {
        "attn_Hnorm": 0,
        "attn_H": 0,
    }
    
    model.train()
    attn_list = []
    thetaG_list = []
    for ide, (_, data, label, name) in enumerate(tqdm(loader)):

        data = data.to(device)
        label = label.to(device).long()
        output = model(data)
        if isinstance(group_model, torch.nn.Module):
            groups = cluster_record[name[0]]["groups"]
            group_ids = cluster_record[name[0]]["id"]
            groups_feature = torch.cat([group.mean(dim=0, keepdim=True) for group in groups], dim=0).unsqueeze(0).to(device)
            # print(groups_feature.shape)
            with torch.no_grad():
                group_out = group_model(groups_feature)
                attn_list.append(group_out["A"])
            loss = loss_fn(output["logits"], 
                label,
                attn=output["A"],
                group_attn=group_out["A"].squeeze(0),
                group_ids=group_ids
            )
            thetaG_list.append(loss["log_theta_g"])
            
        else:
            loss = loss_fn(output["logits"], label)
        loss_record["attn_Hnorm"] += attention_entropy_norm(output["A"]).item()
        loss_record["attn_H"] += attention_entropy(output["A"]).item()
        # loss["loss"] -= 1e-4 * attention_entropy(output["A"])
        loss["loss"].backward()
        optimizer.step()
        probs = F.softmax(output["logits"] , dim=1)

        # record
        Y_prob_list.append(probs.detach().cpu())
        label_list.append(label.detach().cpu())
        for loss_type, loss in loss.items():
            if loss_type.startswith("log"):
                continue
            elif loss_type not in loss_record:
                loss_record[loss_type] = loss.item()
            else:
                loss_record[loss_type] += loss.item()
        
    return Y_prob_list, label_list, loss_record
        

def train(args, save_dir, abmil, group_abmil, train_loader, validation_loader, Group_train_loader, Group_validation_loader, Group_test_loader, cluster_record, test_loader=None, wandb=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    abmil.train()
    group_abmil.train()

    optimizer = torch.optim.Adam(
        list(abmil.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )
    optimizer_group = torch.optim.Adam(
        list(group_abmil.parameters()),
        lr=args.group_lr,
        weight_decay=1e-3
    )

    patch_loss_fn = MILLoss(criterion=torch.nn.CrossEntropyLoss(),
        use_dirichlet=True,
        dirichlet_weight=args.dirichlet_weight,
        dirichlet_kwargs={
            "theta_min": args.theta_start,
            "theta_max": args.theta_end,
            "k": args.k,
        })
    group_loss_fn = MILLoss(criterion=torch.nn.CrossEntropyLoss())

    none_epoch = 0
    best_chief_auc = 0

    print("Training Group!!!")
    for epoch in range(args.num_epochs):
        
        Y_prob_list, label_list, loss_record = train_loop(model=group_abmil, optimizer=optimizer_group, loss_fn=group_loss_fn, loader=Group_train_loader)
        if args.use_wandb:
            log_metrics(
                epoch+1,
                label_list,
                Y_prob_list,
                loss_record,
                train_loader,
                calculate_metrics,
                wandb,
                prefix="train_group",
            )
        chief_auc = test(args, cluster_record, group_abmil, group_loss_fn, Group_validation_loader, run_type="val_group", group_model=None, epoch=epoch, wandb=wandb)


        if chief_auc >= best_chief_auc:
            best_chief_auc = chief_auc
            none_epoch = 0
            # save model
            best_group_abmil = copy.deepcopy(group_abmil)
            torch.save(group_abmil.state_dict(), os.path.join(save_dir, f"group_abmil.pth"))
        elif none_epoch > args.patience:
            break

        none_epoch += 1
    # Final group test
    chief_auc = test(args, cluster_record, group_abmil, group_loss_fn, Group_test_loader, run_type="test_group", group_model=None, epoch=epoch, wandb=wandb)
    group_abmil = best_group_abmil

    none_epoch = 0
    best_chief_auc = 0
    print("Training !!!")
    for epoch in range(args.num_epochs):

        Y_prob_list, label_list, loss_record = train_loop(model=abmil, optimizer=optimizer, loss_fn=patch_loss_fn, loader=train_loader, group_model=group_abmil, cluster_record=cluster_record)
        
        # train record
        if args.use_wandb:
            log_metrics(
                epoch+1,
                label_list,
                Y_prob_list,
                loss_record,
                train_loader,
                calculate_metrics,
                wandb,
                prefix="train",
            )
        

        # val
        chief_auc = test(args, cluster_record, abmil, patch_loss_fn, validation_loader, run_type="val", group_model=group_abmil, epoch=epoch, wandb=wandb)
        
        if chief_auc > best_chief_auc:
            best_chief_auc = chief_auc
            none_epoch = 0
            # save model
            best_abmil = copy.deepcopy(abmil)
            best_group_abmil = copy.deepcopy(group_abmil)
            torch.save(abmil.state_dict(), os.path.join(save_dir, f"abmil.pth"))
            torch.save(group_abmil.state_dict(), os.path.join(save_dir, f"group_abmil.pth"))
        elif none_epoch >= args.patience:
            print(f"Break at epoch {epoch}.")
            break

        none_epoch += 1

    # Final test
    test(args, cluster_record, best_abmil, patch_loss_fn, test_loader, run_type="test", group_model=best_group_abmil, epoch=epoch, wandb=wandb)
    eval_test(save_dir=save_dir, model=best_abmil, test_loader=test_loader)


def log_metrics(
    epoch,
    label_list,
    Y_prob_list,
    loss_record,
    train_loader,
    calculate_metrics,
    wandb,
    prefix="train",
):
    """
    Log training metrics and losses to wandb.

    Args:
        epoch (int)
        label_list (List[Tensor])
        Y_prob_list (List[Tensor])
        loss_record (dict): {loss_name: accumulated_loss}
        train_loader (DataLoader)
        calculate_metrics (callable)
        wandb (wandb module or None)
        prefix (str): e.g. 'train' or 'val'
    """

    # ---- concat targets & probs ----
    targets = (
        torch.cat(label_list, dim=0)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    probs = (
        torch.cat(Y_prob_list, dim=0)
        .detach()
        .cpu()
        .numpy()
    )

    precision, recall, f1, auc, acc = calculate_metrics(targets, probs)

    record = {
        f"{prefix}/epoch": epoch,
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/f1": f1,
        f"{prefix}/auc": auc,
        f"{prefix}/acc": acc,
    }

    # ---- losses ----
    if loss_record:
        for loss_type, loss in loss_record.items():
            record[f"{prefix}/{loss_type}"] = loss / len(train_loader)
    
    wandb.log(record)


def train_baseline(args, save_dir, model, train_loader, validation_loader, test_loader, wandb):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=5e-5, weight_decay=1e-5
    )

    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./CHIEF_model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()
    
    none_epoch = 0
    best_auc = 0


    for idx, epoch in enumerate(range(args.num_epochs)):
        model.train()

        loss_record = {
            "loss": 0,
        }

        label_list = []
        Y_prob_list = []

        for _, (_, data, label, name) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            data = data.squeeze(0).to(device)
            label = label.to(device).long()
            data = chief_wsi_embedding(chief_model, data)
            W_logits = model(data)
            W_Y_prob = F.softmax(W_logits, dim=1)

            loss = F.cross_entropy(W_logits, label)
            loss.backward()
            optimizer.step()

            # 計算正確率
            loss_record["loss"] += loss.item()
            label_list.append(label)
            Y_prob_list.append(W_Y_prob)

        if args.use_wandb:
            log_metrics(
                epoch,
                label_list,
                Y_prob_list,
                loss_record,
                train_loader,
                calculate_metrics,
                wandb,
                prefix="train",
            )

        val_auc = test_baseline(args=args, model=model, chief_model=chief_model, test_loader=validation_loader, run_type="val", epoch=epoch, wandb=wandb)
        

        if val_auc > best_auc:
            best_auc = val_auc
            # save model
            print(f'Save model at epoch {epoch}.')
            print(f'val auc: {val_auc}')
            torch.save(model.state_dict(), os.path.join(save_dir, "basemodel.pth"))
            none_epoch = 0
        elif none_epoch >= 10:
            print(f'Break at epoch {epoch}.')
            break
        
        none_epoch += 1

def test_baseline(args, model, chief_model, test_loader, run_type="test", epoch=0, wandb=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        group_list = []
        
        loss_record = {
            "loss": 0,
        }

        for _, (_, data, label, name) in enumerate(tqdm(test_loader)):
            
            data = data.squeeze(0).to(device)
            label = label.to(device).long()
            data = chief_wsi_embedding(chief_model, data)
            logits = model(data)
            probs = F.softmax(logits , dim=1)
            loss_record["loss"] += F.cross_entropy(logits, label).item()

            # record
            Y_prob_list.append(probs.detach().cpu())
            label_list.append(label.detach().cpu())


        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        # 預測 label
        preds = np.argmax(probs, axis=1)
        correct = (preds == targets).astype(int)

        if args.use_wandb:
            log_metrics(
                epoch+1,
                label_list,
                Y_prob_list,
                loss_record,
                test_loader,
                calculate_metrics,
                wandb,
                prefix=run_type,
            )
            
    return auc


def eval_baseline_test(save_dir, model, test_loader):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./CHIEF_model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()

    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        

        for idx, (_, data, label, name) in enumerate (tqdm(test_loader)):
            
            data = data.squeeze(0).to(device)
            label = label.to(device).long()
            data = chief_wsi_embedding(chief_model, data)
            logits = model(data)
            probs = F.softmax(logits , dim=1)
            # record
            Y_prob_list.append(probs.detach().cpu())
            label_list.append(label.detach().cpu())


        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        # 預測 label
        preds = np.argmax(probs, axis=1)
        correct = (preds == targets).astype(int)

        df = pd.DataFrame({
            'label': targets,
            'pred': preds,
            'correct': correct,
            'prob': probs[:, 1],
        })
        df.to_csv(os.path.join(save_dir, "probability.csv"), index=False)

        df_static = pd.DataFrame({
            'auc': [auc],
            'acc': [accuracy],
            'f1': [f1],
            'recall': [recall],
            'precision': [precision],
        })
        df_static.to_csv(os.path.join(save_dir, "static.csv"), index=False)


def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
