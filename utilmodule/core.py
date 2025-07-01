import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from utilmodule.utils import calculate_metrics

import numpy as np
from sklearn.cluster import KMeans
import time
import random
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import wandb
import datetime
import os
from tqdm import tqdm
from models.CHIEF import CHIEF
from CHIEF_network import ClfNet
from gigapath.pipeline import run_inference_with_slide_encoder


def test(args,ppo,classifier_chief, classifier_giga,memory,test_loader, chief_model, gigapath_model, run_type="test", epoch=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        for idx, (coords, chief_data, gigapath_data, label) in enumerate (tqdm(test_loader)):
            correct = 0
            total = 0
            coords = coords.squeeze(dim=3)
            update_coords, update_chief_data, update_gigapath_data, label = coords.to(device), chief_data.to(device), gigapath_data.to(device), label.to(device).long()
            # 預處理
            if args.type == 'camelyon16':
                update_data = basedmodel.fc1(update_data)
            else:
                chief_data = chief_data.float()
                gigapath_data = gigapath_data.float()

            #if args.ape:
            #    chief_data += basedmodel.absolute_pos_embed.expand(1, chief_data.shape[1], basedmodel.args.embed_dim)
            #    gigapath_data += basedmodel.absolute_pos_embed.expand(1, gigapath_data.shape[1], basedmodel.args.embed_dim)

            grouping_instance = grouping(action_size=args.action_size)
            whole_chief = update_chief_data.squeeze(0) 
            # CHIEF model replace the basemodel                                       
            memory.merge_msg_states.append(cheif_wsi_embedding(chief_model, whole_chief))    

            _ = ppo.select_action(
                None, memory, restart_batch=True, training=False
            )

            action_index_pro, memory = grouping_instance.rlselectindex_grouping(
                ppo, memory, update_coords, sigma=0.02, restart=False
            )

            chief_features_group, gigapath_features_group, update_coords, update_chief_data, update_gigapath_data, memory = grouping_instance.action_make_subbags(
                ppo, memory, action_index_pro, update_coords, update_chief_data, update_gigapath_data,
                action_size=args.action_size, restart=False, delete_begin=True
            )

            chief_features_group = chief_features_group[0].squeeze(0) 
            memory.select_chief_feature_pool.append(chief_features_group)                                             

            wsi_embedding_chief = cheif_wsi_embedding(chief_model, torch.cat(memory.select_chief_feature_pool, dim=0))

            output = classifier_chief(wsi_embedding_chief)

            # Soft Voting
            pred = torch.argmax(output, dim=1)
            probs = F.softmax(output, dim=1)

            # 計算正確率
            correct += (pred == label).sum().item()
            total += label.size(0)
            memory.clear_memory()
            label_list.append(label)
            Y_prob_list.append(probs)

        targets = np.asarray(torch.cat(label_list, dim=0).cpu().numpy()).reshape(-1)  
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).cpu().numpy())  
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        #print(args.csv.split("/")[-1])
        #print(f'[Epoch {epoch+1}/{args.num_epochs}] {run_type} Accuracy: {accuracy:.4f} " {run_type} Precision: {precision:.4f}, {run_type} Recall: {recall:.4f}, {run_type} F1 Score: {f1:.4f}, {run_type} AUC: {auc:.4f}')
    return precision, recall, f1, auc, accuracy

def test_baseline(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,test_loader, run_type="test", epoch=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        for idx, (coords, data, label) in enumerate (tqdm(test_loader)):
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()


            #if args.type == 'camelyon16':
            #    update_data = basedmodel.fc1(update_data)
            #else:
            #    update_data = update_data.float()
            #if args.ape:
            #    update_data = update_data + basedmodel.absolute_pos_embed.expand(1,update_data.shape[1],basedmodel.args.embed_dim)

            #update_coords, update_data ,total_T = expand_data(update_coords, update_data, action_size = args.action_size, total_steps=args.test_total_T)
            #grouping_instance = grouping(action_size=args.action_size)

            W_logits = classifymodel (update_data).squeeze(1)
            W_Y_prob = F.softmax(W_logits, dim=1)
            W_Y_hat = torch.argmax(W_Y_prob, dim=1)

            label_list.append(label)
            Y_prob_list.append(W_Y_prob)

        targets = np.asarray(torch.cat(label_list, dim=0).cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        #print(f'[Epoch {epoch+1}/{args.num_epochs}] {run_type} Accuracy: {accuracy:.4f} " {run_type} Precision: {precision:.4f}, {run_type} Recall: {recall:.4f}, {run_type} F1 Score: {f1:.4f}, {run_type} AUC: {auc:.4f}')
    return precision, recall, f1, auc, accuracy

def cheif_wsi_embedding(chief_model, feature):
    anatomical=13
    with torch.no_grad():
        x,tmp_z = feature,anatomical
        result = chief_model(x, torch.tensor([tmp_z]))
        wsi_feature_emb = result['WSI_feature']  ###[1,768]
        # print(wsi_feature_emb.size())

    return wsi_feature_emb

def gigapath_wsi_embedding(gigapath_model, feature, coords):
    with torch.no_grad():
        #wsi_feature_emb = gigapath_model(feature, coords).squeeze()
        wsi_feature_emb = run_inference_with_slide_encoder(slide_encoder_model=gigapath_model, tile_embeds=feature, coords=coords)

    return wsi_feature_emb['last_layer_embed']

def train_stage1(args,ppo,classifier_chief, classifier_giga,gigapath_model, memory,train_loader, validation_loader, test_loader, wandb):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list = []
    Y_prob_list = []

    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()

    none_epoch = 0
    best_auc = 0
    
    for idx, epoch in enumerate(range(20)):
        classifier_chief.train()
        classifier_giga.train()

        chief_loss = 0
        giga_loss = 0
        correct = 0
        total = 0

        for ide, (coords, chief_data, gigapath_data, label) in enumerate(tqdm(train_loader)):
            coords = coords.squeeze(dim=3)
            update_coords, update_chief_data, update_gigapath_data, label = coords.to(device), chief_data.to(device), gigapath_data.to(device), label.to(device).long()
            # 預處理
            if args.type == 'camelyon16':
                update_data = basedmodel.fc1(update_data)
            else:
                chief_data = chief_data.float()
                gigapath_data = gigapath_data.float()

            grouping_instance = grouping(action_size=args.action_size)
            whole_chief = update_chief_data.squeeze(0) 
            # CHIEF model replace the basemodel                                       
            memory.merge_msg_states.append(cheif_wsi_embedding(chief_model, whole_chief))    

            _ = ppo.select_action(
                None, memory, restart_batch=True, training=True
            )

            action_index_pro, memory = grouping_instance.rlselectindex_grouping(
                ppo, memory, update_coords, sigma=0.02, restart=False
            )

            chief_features_group, gigapath_features_group, update_coords, update_chief_data, update_gigapath_data, memory = grouping_instance.action_make_subbags(
                ppo, memory, action_index_pro, update_coords, update_chief_data, update_gigapath_data,
                action_size=args.action_size, restart=False, delete_begin=True
            )

            chief_features_group = chief_features_group[0].squeeze(0) 
            memory.select_chief_feature_pool.append(chief_features_group)                                             

            
            gigapath_features_group = gigapath_features_group[0].unsqueeze(0)
            memory.select_gigapath_feature_pool.append(gigapath_features_group)

            # 最終分類（WSI level）
            wsi_embedding_chief = cheif_wsi_embedding(chief_model, torch.cat(memory.select_chief_feature_pool, dim=0)).detach().requires_grad_()
            wsi_embedding_gigapath = gigapath_wsi_embedding(gigapath_model, torch.cat(memory.select_gigapath_feature_pool, dim=1), torch.cat(memory.coords_actions, dim=1)).to(device).detach().requires_grad_()
            
            # chief
            output = classifier_chief(wsi_embedding_chief)
            loss = F.cross_entropy(output, label)
            loss.backward()
            chief_grad_norms = wsi_embedding_chief.grad.norm(p=2, dim=1)  # shape = [B]
            # print(chief_grad_norms)
            
            # gigapath
            output_giga = classifier_giga(wsi_embedding_gigapath)
            loss_giga = F.cross_entropy(output_giga, label)
            loss_giga.backward()
            giga_grad_norms = wsi_embedding_gigapath.grad.norm(p=2, dim=1)  # shape = [B]
            # print(giga_grad_norms)


            values = torch.stack([-1*chief_grad_norms, -1*giga_grad_norms])
            # print(values.mean())
            memory.rewards.append(values.mean().unsqueeze(0))
            record_reward = memory.rewards[-1]

            wandb.log({
                "reward": record_reward,
            })
            classifier_chief.zero_grad()
            classifier_giga.zero_grad()

        memory.actions.insert(0, -1)
        memory.logprobs.insert(0, -1)
        ppo.update(memory)
        memory.clear_memory()
    

    return ppo


def train(args,basedmodel,ppo,classifier_chief, classifier_giga,FusionHisF,gigapath_model, memory,train_loader, validation_loader, test_loader=None, wandb=None):
    
    run_name = f"{args.csv.split('/')[-1].split('.')[0]}"
    save_dir = os.path.join(args.save_dir, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list = []
    Y_prob_list = []
    giga_label_list = []
    giga_Y_prob_list = []

    optimizer_chief = torch.optim.Adam(list(classifier_chief.parameters()),lr=1e-4, weight_decay=1e-5)
    optimizer_giga = torch.optim.Adam(list(classifier_giga.parameters()),lr=1e-4, weight_decay=1e-5)

    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()

    none_epoch = 0
    best_auc = 0
    
    for idx, epoch in enumerate(range(args.num_epochs)):
        classifier_chief.train()
        classifier_giga.train()

        chief_loss = 0
        giga_loss = 0
        correct = 0
        total = 0
        giga_correct = 0
        giga_total = 0

        for ide, (coords, chief_data, gigapath_data, label) in enumerate(tqdm(train_loader)):
            optimizer_chief.zero_grad()
            optimizer_giga.zero_grad()
            coords = coords.squeeze(dim=3)
            update_coords, update_chief_data, update_gigapath_data, label = coords.to(device), chief_data.to(device), gigapath_data.to(device), label.to(device).long()
            # 預處理
            if args.type == 'camelyon16':
                update_data = basedmodel.fc1(update_data)
            else:
                chief_data = chief_data.float()
                gigapath_data = gigapath_data.float()

            #if args.ape:
            #    chief_data += basedmodel.absolute_pos_embed.expand(1, chief_data.shape[1], basedmodel.args.embed_dim)
            #    gigapath_data += basedmodel.absolute_pos_embed.expand(1, gigapath_data.shape[1], basedmodel.args.embed_dim)

            grouping_instance = grouping(action_size=args.action_size)
            # CHIEF model replace the basemodel    
            # whole_chief = update_chief_data.detach().squeeze(0)                                   
            # memory.merge_msg_states.append(cheif_wsi_embedding(chief_model, whole_chief)) 
            whole_gigapath = update_gigapath_data.detach()
            memory.merge_msg_states.append(gigapath_wsi_embedding(gigapath_model, whole_gigapath, update_coords).to(device))  

            _ = ppo.select_action(
                None, memory, restart_batch=True, training=True
            )

            action_index_pro, memory = grouping_instance.rlselectindex_grouping(
                ppo, memory, update_coords, sigma=0.02, restart=False
            )

            chief_features_group, gigapath_features_group, update_coords, update_chief_data, update_gigapath_data, memory = grouping_instance.action_make_subbags(
                ppo, memory, action_index_pro, update_coords, update_chief_data, update_gigapath_data,
                action_size=args.action_size, restart=False, delete_begin=True
            )

            chief_features_group = chief_features_group[0].squeeze(0) 
            memory.select_chief_feature_pool.append(chief_features_group)                                             

            gigapath_features_group = gigapath_features_group[0].unsqueeze(0)
            memory.select_gigapath_feature_pool.append(gigapath_features_group)

            # 最終分類（WSI level）
            wsi_embedding_chief = cheif_wsi_embedding(chief_model, torch.cat(memory.select_chief_feature_pool, dim=0))
            wsi_embedding_gigapath = gigapath_wsi_embedding(gigapath_model, torch.cat(memory.select_gigapath_feature_pool, dim=1), torch.cat(memory.coords_actions, dim=1))
            
            # chief
            optimizer_chief.zero_grad()
            output = classifier_chief(wsi_embedding_chief)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer_chief.step()
            
            # record loss
            chief_loss += loss.item()
            
            pred_chief = torch.argmax(output, dim=1)
            probs_chief = F.softmax(output, dim=1)
            
            # gigapath
            optimizer_giga.zero_grad()
            output_giga = classifier_giga(wsi_embedding_gigapath.to(device))
            loss_giga = F.cross_entropy(output_giga, label)
            loss_giga.backward()
            optimizer_giga.step()

            # record loss
            giga_loss += loss_giga.item()

            pred_giga = torch.argmax(output_giga, dim=1)
            probs_giga = F.softmax(output_giga, dim=1)

            values = torch.stack([probs_chief[0][label.item()].detach(), probs_giga[0][label.item()].detach()])
            memory.rewards.append((values.mean() - values.var(unbiased=False)).unsqueeze(0))
            record_reward = memory.rewards[-1]
            if (ide+1) % 64 == 0 or ide == len(train_loader)-1:
                memory.actions.insert(0, -1)
                memory.logprobs.insert(0, -1)
                ppo.update(memory)
                memory.clear_memory()

            wandb.log({
                "reward": record_reward,
                "probability/chief_true_label": probs_chief[0][label.item()].detach(), 
                "probability/chief_false_label": probs_chief[0][1 - label.item()].detach(), 
                "probability/gigapath_true_label": probs_giga[0][label.item()].detach(), 
                "probability/gigapath_false_label": probs_giga[0][1 - label.item()].detach(),
            })

            # 計算正確率 chief
            correct += (pred_chief == label).sum().item()
            total += label.size(0)
            label_list.append(label)
            Y_prob_list.append(probs_chief)

            # gigapath
            giga_correct += (pred_giga == label).sum().item()
            giga_total += label.size(0)
            giga_label_list.append(label)
            giga_Y_prob_list.append(probs_giga)

            
            memory.clear_memory_training()

        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)

        memory.clear_memory()
        #print(f'[Epoch {epoch+1}/{args.num_epochs}] train Loss: {epoch_loss:.4f}, train Accuracy: {accuracy:.4f}, train Precision: {precision:.4f},train Recall: {recall:.4f},train F1 Score: {f1:.4f},train AUC: {auc:.4f}')
        
        wandb.log({
            f"classifier_chief_loss": chief_loss/len(train_loader),
        })

        wandb.log({
            f"classifier_giga_loss": giga_loss/len(train_loader),
        })


        wandb.log({
            "epoch": epoch,
            "train/precision": precision,
            "train/recall": recall,
            "train/f1": f1,
            "train/auc": auc,
            "train/acc": accuracy
        })

        # gigapath record
        targets = np.asarray(torch.cat(giga_label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(giga_Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        wandb.log({
            "epoch": epoch,
            "train_giga/precision": precision,
            "train_giga/recall": recall,
            "train_giga/f1": f1,
            "train_giga/auc": auc,
            "train_giga/acc": accuracy
        })

        # val
        precision, recall, f1, val_auc, val_accuracy = test(args,ppo,classifier_chief, classifier_giga,memory,test_loader, chief_model, gigapath_model, run_type="val", epoch=epoch)
        wandb.log({
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "val/auc": val_auc,
            "val/acc": val_accuracy
        })

        # test
        precision, recall, f1, auc, accuracy = test(args,ppo,classifier_chief, classifier_giga,memory,test_loader, chief_model, gigapath_model, run_type="test", epoch=epoch)
        wandb.log({
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
            "test/auc": auc,
            "test/acc": accuracy
        })
        
        if val_auc >= best_auc:
            best_auc = val_auc
            # save model
            print(f'Save model at epoch {epoch}.')
            print(f'val auc: {val_auc}')
            torch.save(classifier_chief.state_dict(), os.path.join(save_dir, "classifier_chief.pth"))
            torch.save(classifier_giga.state_dict(), os.path.join(save_dir, "classifier_gigapath.pth"))
            ppo.save(save_dir)
            none_epoch = 0
        elif none_epoch >= args.patience:
            print(f"Break at epoch {epoch}.")
            break

        none_epoch += 1
        
def train_baseline(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,train_loader, validation_loader, test_loader=None):
    run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.login(key="")
    wandb.init(
        project="WSI_baseline",      # 可以在網站上看到
        name=run_name,      # optional，可用於區分實驗
        config=vars(args)                    # optional，紀錄一些超參數
    )

    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifymodel.to(device)
    optimizer = torch.optim.Adam(
        list(classifymodel.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    
    none_epoch = 0
    best_auc = 0

    for idx, epoch in enumerate(range(args.num_epochs)):
        classifymodel.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        label_list = []
        Y_prob_list = []

        for _, (coords, data, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()

            # 預處理
            #if args.type == 'camelyon16':
            #    update_data = basedmodel.fc1(update_data)
            #else:
            #    update_data = update_data.float()
            
            W_logits = classifymodel(update_data).squeeze(1)
            W_Y_prob = F.softmax(W_logits, dim=1)
            W_Y_hat = torch.argmax(W_Y_prob, dim=1)

            # 計算 loss（以交叉熵為例）
            #print(W_logits.shape)
            #print(label.shape)
            loss_WSI = F.cross_entropy(W_logits, label)
            #print(loss_SIA.item(), loss_WSI.item())
            loss = loss_WSI
            loss.backward()
            optimizer.step()

            #print(f'Idx: {idx},loss: {loss.item()}')

            # reward

            # 計算正確率
            epoch_loss += loss.item()
            correct += (W_Y_hat == label).sum().item()
            total += label.size(0)
            label_list.append(label)
            Y_prob_list.append(W_Y_prob)

        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)


        wandb.log({
            "loss": epoch_loss,
            "epoch": epoch,
            "train/precision": precision,
            "train/recall": recall,
            "train/f1": f1,
            "train/auc": auc,
            "train/acc": accuracy
        })

        #acc = correct / total
        #print(f"[Epoch {epoch+1}/{args.num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")
        precision, recall, f1, val_auc, val_accuracy = test_baseline(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,validation_loader, "val", epoch)
        wandb.log({
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "val/auc": val_auc,
            "val/acc": val_accuracy
        })
        precision, recall, f1, auc, accuracy = test_baseline(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,test_loader, "test", epoch)
        wandb.log({
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
            "test/auc": auc,
            "test/acc": accuracy
        })
        if val_auc >= best_auc:
            best_auc = val_auc
            # save model
            print(f'Save model at epoch {epoch}.')
            print(f'val auc: {val_auc}')
            torch.save(classifymodel.state_dict(), os.path.join(save_dir, "classifymodel.pth"))
            none_epoch = 0
        elif none_epoch >= args.patience:
            print(f"Break at epoch {epoch}.")
            break

        none_epoch += 1


def interpolate_probs(probs, new_length,action_size):
    b,c = probs.shape
    x = np.linspace(0, 1, c)
    x_new = np.linspace(0, 1, new_length)
    new_probs = np.interp(x_new, x, probs.view(-1).cpu().numpy())
    interpolate_action_probs = new_probs / new_probs.sum()   
    index = np.random.choice(np.arange(new_length), size=action_size, p=interpolate_action_probs)
    return index


def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


    
class grouping:

    def __init__(self,action_size = 128):
        self.action_size = action_size 
        self.action_std = 0.1
           
     
    def rlselectindex_grouping(self,ppo,memory,coords,sigma=0.02,restart=False): 
        B, N, C = coords.shape
        # print(coords.shape)
        if restart  : 
            if self.action_size < N: 
                action = torch.distributions.dirichlet.Dirichlet(torch.ones(self.action_size)).sample((1,)).to(coords.device)
                # torch.manual_seed(int(time.time() * 1000) % (2**32 - 1))
                # action = torch.distributions.dirichlet.Dirichlet(torch.ones(self.action_size)).sample((1,)).to(coords.device)
                # torch.manual_seed(2021)
                # print(action)
            else: 
                random_values = torch.rand(1, N).to(coords.device)
                indices = torch.randint(0, N, (self.action_size,)).to(coords.device)
                action = random_values[0, indices].unsqueeze(0)
            memory.actions.append(action) 
            memory.logprobs.append(action) 
            return action.detach(), memory
        else:  
            action = memory.actions[-1] 
            
            return action.detach(), memory
          
    
    def action_make_subbags(self,ppo,memory, action_index_pro, update_coords,chief_features,gigapath_features,action_size= None,restart= False,delete_begin=False): 
        
        B, N, C = update_coords.shape
        idx = interpolate_probs(action_index_pro, new_length = N ,action_size = action_size)
        idx = torch.tensor(idx)
        chief_features_group = chief_features[:, idx[:], :]
        gigapath_features_group = gigapath_features[:, idx[:], :]
        action_group = update_coords[:, idx[:], :] 

        if restart and delete_begin:
            memory.coords_actions.append(action_group)
            return chief_features_group, gigapath_features_group, update_coords, chief_features, gigapath_features ,memory
        else:
            idx = torch.unique(idx)
            chief_mask = torch.ones(chief_features.size(1), dtype=torch.bool) 
            chief_mask[idx] = False  
            updated_chief_features = chief_features[:, chief_mask, :] 
            gigapath_mask = torch.ones(gigapath_features.size(1), dtype=torch.bool)
            gigapath_mask[idx] = False
            updated_gigapath_features = gigapath_features[:, gigapath_mask, :]
            updated_coords = update_coords[:, chief_mask, :]
            memory.coords_actions.append(action_group)
            return chief_features_group, gigapath_features_group, updated_coords, updated_chief_features, updated_gigapath_features ,memory

    
