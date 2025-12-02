import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch.nn as nn
import torch.nn.functional as F
import torch
from utilmodule.utils import calculate_metrics
from NoisetwoModel_group_rewardCalibration_MCDCP.models.DPSF import PPO,Memory
from NoisetwoModel_group_rewardCalibration_MCDCP.models.classifier import TwoLayerClassifier
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch.nn.functional as F
import os
from tqdm import tqdm
from models.CHIEF import CHIEF
from CHIEF_network import ClfNet
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch.optim as optim

def test(args,ppo_chief,classifier_chief,chief_memory, cluster_list, test_loader, chief_model, run_type="test", epoch=0, wandb=None, run_time_test=True, record_csv=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classifiers = len(classifier_chief)
    with torch.no_grad():
        label_list = []
        sample_names = []
        chief_Y_prob_list = []
        loss_classifier = [0 for _ in range(num_classifiers)]

        for idx, (coords, chief_data, gigapath_data, label, name) in enumerate (tqdm(test_loader)):
            sample_names.append(name[0])
            coords = coords.squeeze(dim=3)
            
            update_chief_coords, update_chief_data, label = coords.to(device), chief_data.squeeze(0).to(device), label.to(device).long()
            patch_num = update_chief_data.size(1)
            
            centers, groups = cluster_list[name[0]]["centers"], cluster_list[name[0]]["groups"]

            for center, feature_group in zip(centers, groups):
                select_and_embed_chief_patches(
                    ppo_chief=ppo_chief,
                    chief_memory=chief_memory,
                    center=center.unsqueeze(0),
                    Now_picked_num=len(chief_memory.select_chief_feature_pool),
                    Total_picked_num=len(centers),
                    feature_group=feature_group,
                )


            # 最終分類（WSI level）
            if len(chief_memory.select_chief_feature_pool) == 0:
                # 若沒有任何被選到的 feature，就建立一個 [1, 768] 的 0 tensor
                wsi_embedding_chief = torch.zeros((1, 768), device=device)
            else:
                wsi_embedding_chief = chief_wsi_embedding(
                    chief_model,
                    torch.cat(chief_memory.select_chief_feature_pool, dim=0).to(device)
                ).detach()
            # wsi_embedding_gigapath = gigapath_wsi_embedding(gigapath_model, torch.cat(gigapath_memory.select_gigapath_feature_pool, dim=1), torch.cat(gigapath_memory.coords_actions, dim=1))


            ensemble_pros = torch.zeros((1, 2), dtype=torch.float, device=device)
            for i, chief_classifier in enumerate(classifier_chief):

                # forward
                chief_output = chief_classifier(wsi_embedding_chief)  # [1,2] 

                # loss
                loss = F.cross_entropy(chief_output, label.expand(chief_output.size(0)))
                loss_classifier[i] += loss.item()
                
                # prob
                probs_chief = F.softmax(chief_output, dim=1)
                ensemble_pros += probs_chief


            # ensemble
            label_list.append(label)

            # chief
            ensemble_pros /= num_classifiers
            chief_Y_prob_list.append(ensemble_pros)

            # # gigapath
            # giga_Y_prob_list.append(probs_giga)

            chief_memory.clear_memory()

        if run_time_test:
            # chief record
            targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
            probs = np.asarray(torch.cat(chief_Y_prob_list, dim=0).detach().cpu().numpy())
            precision, recall, f1, chief_auc, accuracy = calculate_metrics(targets, probs)
            wandb.log({
                f"{run_type}_chief/precision": precision,
                f"{run_type}_chief/recall": recall,
                f"{run_type}_chief/f1": f1,
                f"{run_type}_chief/auc": chief_auc,
                f"{run_type}_chief/acc": accuracy,
            })

            for i in range(num_classifiers):
                wandb.log({
                    f"{run_type}_chief_classifier_{i}/loss": loss_classifier[i]/len(test_loader),
                })


        targets = np.asarray(torch.cat(label_list, dim=0).cpu().numpy()).reshape(-1)  
        probs = np.asarray(torch.cat(chief_Y_prob_list, dim=0).detach().cpu().numpy()) # change now -> CHIEF
        if record_csv:
            
            # 預測 label
            preds = np.argmax(probs, axis=1)
            correct = (preds == targets).astype(int)

            # 假設 idx_patches 只在 if/else 裡面生成，可以先存到一個 list
            # 在 loop 外先建一個 list
            # selected_patch_ids = []
            # 在每個 iter 裡，把 idx_patches 加進去
            # selected_patch_ids.append(idx_patches.cpu().numpy())

            df = pd.DataFrame({
                'name': sample_names,
                'label': targets,
                'pred': preds,
                'correct': correct,
                'prob': probs[:, 1],
            })
            df.to_csv(os.path.join(args.test_dir, args.csv_saveName), index=False)
            

    return chief_auc, torch.cat(label_list, dim=0).cpu(), torch.cat(chief_Y_prob_list, dim=0).detach().cpu()

def test_baseline(args,chief_model,classifymodel,FusionHisF,memory_space,test_loader, run_type="test", chose="top"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        for idx, (chief_data, label) in enumerate(tqdm(test_loader)):
            label = label.to(device).long()
            chief_data = chief_data.to(device)
            W_logits = classifymodel(chief_data).squeeze(1)
            W_Y_prob = F.softmax(W_logits, dim=1)
            W_Y_hat = torch.argmax(W_Y_prob, dim=1)

            label_list.append(label)
            Y_prob_list.append(W_Y_prob)

        targets = np.asarray(torch.cat(label_list, dim=0).cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        # df = pd.DataFrame({
        #         'label': targets,
        #         'prob': probs[:, 1]
        #     })
        # df.to_csv(os.path.join(args.test_dir, args.csv_saveName), index=False)
        #print(f'[Epoch {epoch+1}/{args.num_epochs}] {run_type} Accuracy: {accuracy:.4f} " {run_type} Precision: {precision:.4f}, {run_type} Recall: {recall:.4f}, {run_type} F1 Score: {f1:.4f}, {run_type} AUC: {auc:.4f}')
    return precision, recall, f1, auc, accuracy

def chief_wsi_embedding(chief_model, feature, type="model"):

    if type == "model":
        anatomical=13
        with torch.no_grad():
            x,tmp_z = feature,anatomical
            result = chief_model(x, torch.tensor([tmp_z]))
            wsi_feature_emb = result['WSI_feature']  ###[1,768]
            # print(wsi_feature_emb.size())
    elif type == "mean-pooling":
        wsi_feature_emb = feature.mean(dim=0, keepdim=True)
    elif type == "max-pooling":
        wsi_feature_emb, _ = feature.max(dim=0, keepdim=True)

    return wsi_feature_emb


class GCELoss(nn.Module):
    def __init__(self, q=0.7):  # q ∈ (0,1], 控制曲線形狀
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B] (LongTensor)
        probs = F.softmax(logits, dim=1)
        probs_true = probs[torch.arange(len(targets)), targets]
        if self.q == 1.0:
            return -torch.log(probs_true).mean()  # 退化成CE
        else:
            return ((1 - probs_true ** self.q) / self.q).mean()


# ===== Mapping function h(·) =====
class LabelConfidence(nn.Module):
    def __init__(self, m=0.0, momentum=0.7):
        super().__init__()
        self.m = m  # 控制樣本數比例
        self.momentum = momentum
        self.register_buffer("mu", torch.tensor(0.0))  # moving average

    def forward(self, robust_loss_values):
        # robust_loss_values: [B]
        # 更新 batch 平均 loss 的 moving average
        batch_mean = robust_loss_values.mean().detach()
        self.mu = self.momentum * self.mu + (1 - self.momentum) * batch_mean
        # 計算 confidence
        scores = -robust_loss_values + self.mu + self.m
        return torch.sigmoid(0.5 * scores)  # ∈ (0,1)


# ===== Noise-robust model Loss =====
def noise_robust_loss(logits, targets, robust_loss_fn):
    return robust_loss_fn(logits, targets)


# ===== Noise-free model Loss =====
def noise_free_loss(logits, targets, robust_logits, conf_func, robust_loss_fn,
                    label_correction=True, type="noise", beta_coe=0.5):
    """
    logits: noise-free model 輸出 [B, C]
    targets: ground truth labels [B]
    robust_logits: noise-robust model 輸出 [B, C]
    conf_func: LabelConfidence 實例
    """
    # 1. robust loss value (per-sample)
    robust_losses = F.cross_entropy(robust_logits, targets, reduction='none')
    conf = conf_func(robust_losses).detach()  # label confidence

    # 2. Label correction (after warm-up才開啟)
    if label_correction:
        with torch.no_grad():
            robust_probs = F.softmax(robust_logits, dim=1)
        y_onehot = F.one_hot(targets, num_classes=logits.size(1)).float()
        soft_targets = (1 - beta_coe) * y_onehot + beta_coe * robust_probs
    else:
        soft_targets = F.one_hot(targets, num_classes=logits.size(1)).float()

    # 3. Cross-entropy with soft labels (confidence 加權)
    log_probs = F.log_softmax(logits, dim=1)
    ce_loss = -(soft_targets * log_probs).sum(dim=1)  # [B]
    if type == "noise":
        weighted_ce = (conf * ce_loss).mean()
    elif type == "CE":
        weighted_ce = F.cross_entropy(logits, targets)


    return weighted_ce


def train_noise_robust_and_free_models(
    wsi_embedding_chief,
    label,
    chief_noise_robust_model,
    classifier_chief,
    optimizer_noise_robust_chief,
    optimizer_chief,
    noise_robust_loss,
    noise_free_loss,
    robust_loss_fn,
    conf_func,
    label_correction,
    lambda_reg=1e-3,
    beta_coe=0.5,
    Q=None,
):
    """
    同步訓練 Noise-Robust 與 Classifier 模型。

    Args:
        wsi_embedding_chief (Tensor): [B, D] slide-level embedding。
        label (Tensor): [B] labels。
        chief_noise_robust_model (nn.Module)
        classifier_chief (nn.Module)
        optimizer_noise_robust_chief (torch.optim.Optimizer)
        optimizer_chief (torch.optim.Optimizer)
        noise_robust_loss (callable): 計算 robust model loss。
        noise_free_loss (callable): 計算 free model loss。
        robust_loss_fn (callable): 基礎 robust loss 函數，如 GCE, NCE。
        conf_func (callable): confidence function。
        label_correction (bool): 是否啟用 label 修正。
        lambda_reg (float): 正則化權重。
        beta (float): noise-free loss 中的 beta 係數。

    Returns:
        dict: {
            "loss_total": total loss tensor,
            "loss_robust": float,
            "loss_free": float,
            "reg_term": float,
            "probs_robust": Tensor,
            "probs_free": Tensor
        }
    """

    # === Noise-Robust Model ===
    optimizer_noise_robust_chief.zero_grad()
    noise_robust_output = chief_noise_robust_model(wsi_embedding_chief)
    probs_noise_robust = F.softmax(noise_robust_output, dim=1)

    loss_robust = noise_robust_loss(
        noise_robust_output,
        label.expand(noise_robust_output.size(0)),
        robust_loss_fn,
    )

    # === Classifier (Free Model) ===
    optimizer_chief.zero_grad()
    chief_output = classifier_chief(wsi_embedding_chief)
    probs_chief = F.softmax(chief_output, dim=1)

    loss_free = noise_free_loss(
        logits=chief_output,
        targets=label.expand(chief_output.size(0)),
        robust_logits=noise_robust_output,
        conf_func=conf_func,
        robust_loss_fn=robust_loss_fn,
        label_correction=label_correction,
        beta_coe=beta_coe,
    )

    # === Regularization (L2 distance between params) ===
    params_robust = torch.cat([p.view(-1) for p in chief_noise_robust_model.parameters()])
    params_free = torch.cat([p.view(-1) for p in classifier_chief.parameters()])
    reg_term = lambda_reg * torch.norm(params_robust - params_free, p=2)


    # === Total Loss ===
    loss_total = loss_robust + loss_free + reg_term

    # === Backward ===
    optimizer_noise_robust_chief.zero_grad()
    optimizer_chief.zero_grad()
    loss_total.backward()
    optimizer_noise_robust_chief.step()
    optimizer_chief.step()

    if isinstance(Q, torch.Tensor):
        probs_whole_mean, _, _, probs_whole_std = mcd_cp_inference(classifier_chief, wsi_embedding_chief, Q, T=25, dropout_rate=0.1)
        probs_whole = (probs_whole_mean-probs_whole_std)
    else:
        probs_whole = probs_chief


    return {
        "loss_total": loss_total,
        "loss_robust": loss_robust.item(),
        "loss_free": loss_free.item(),
        "reg_term": reg_term.item(),
        "probs_robust": probs_noise_robust.detach(),
        "probs_free": probs_chief.detach(),
        "reward": probs_whole.detach(),
    }


def intermidiate_score(
    whole_embedding,   # [1, D]
    test_embedding,    # [1, D]
    label,
    classifier,        # slide-level classifier
    Q=None,
):
    """
    Compute intermediate reward as a 1D tensor [1]:
        reward = score(test_embedding) - score(whole_embedding)
    """

    # 推進 device 和 dtype
    device = next(classifier.parameters()).device
    dtype  = next(classifier.parameters()).dtype

    whole_embedding = whole_embedding.to(device=device, dtype=dtype)
    test_embedding  = test_embedding.to(device=device, dtype=dtype)


    if isinstance(Q, torch.Tensor):
        probs_whole_mean, _, _, probs_whole_std = mcd_cp_inference(classifier, whole_embedding, Q, T=25, dropout_rate=0.1)
        probs_test_mean, _, _, probs_test_std  = mcd_cp_inference(classifier, test_embedding, Q, T=25, dropout_rate=0.1)
        probs_whole = (probs_whole_mean-probs_whole_std)[0]
        probs_test = (probs_test_mean-probs_test_std)[0]
    else:
        # forward
        logits_whole = classifier(whole_embedding)  # [1, C]
        logits_test  = classifier(test_embedding)   # [1, C]
        probs_whole = torch.softmax(logits_whole, dim=1)[0]  # [C]
        probs_test  = torch.softmax(logits_test, dim=1)[0]   # [C]

    # 取得正確 label 的 score
    if label is None:
        label = 1  # default: positive class

    score_whole = probs_whole[label.item()]
    score_test  = probs_test[label.item()]

    # reward = test - whole，並回傳 shape = [1]
    reward = (score_test - score_whole).unsqueeze(0)  # [1]

    return reward
    


def select_and_embed_chief_patches(
    ppo_chief,
    chief_memory,
    center,
    Now_picked_num,
    Total_picked_num,
    feature_group,
    label=None,
    classifier=None,
    chief_model=None,
    wandb=None,
    Q=None,
    device="cuda"
):
    """
    使用 PPO agent 選擇 patch，建立 sub-bag，並生成 WSI-level embedding。

    Args:
        ppo_chief (nn.Module): Chief PPO agent。
        chief_memory (object): 儲存 agent 狀態的記憶體物件。
        grouping_instance (object): 負責 patch grouping 與 sub-bag 生成的物件。
        chief_model (nn.Module): Chief feature embedding 模型。
        update_chief_coords (Tensor): [N, 2] patch 座標。
        update_chief_data (Tensor): [1, N, D] patch 特徵。
        num_patch (int): patch 數量。
        sigma (float): Gaussian noise 參數，用於 action sampling。
        action_ratio (float): 每次選取的比例（預設 0.4, 即 2/5）。

    Returns:
        dict: {
            "chief_memory": 更新後的記憶體物件,
            "chief_features_group": Tensor [K, D],
            "wsi_embedding_chief": Tensor [1, D],
            "action_index_pro": Tensor (選取的索引機率),
            "idx_patches": Tensor (被選到的 patch 索引)
        }
    """

    # === Step 1. 輸入 WSI-level embedding 到記憶體 ===
    # State prepare
    center = center.float().to(device)                           # 保證 float tensor
    extra = torch.tensor([[Now_picked_num, Total_picked_num]], 
                        dtype=center.dtype, 
                        device=center.device)          # [1, 2]

    state = torch.cat([center, extra], dim=1).to(device)             # [1, 770]
    chief_memory.merge_msg_states.append(state)

    # === Step 2. 讓 PPO 選擇動作 ===
    action = ppo_chief.select_action(
        None,
        chief_memory,
        restart_batch=True,
        training=True
    )

    # === Step 5. 將選到的特徵存入記憶體池 ===
    if action.item() == 1:
        if label != None:
            if len(chief_memory.select_chief_feature_pool) > 0:
                join_embeding = chief_memory.select_chief_feature_pool[:]
                join_embeding.append(feature_group)
                reward = intermidiate_score(
                    whole_embedding=chief_wsi_embedding(chief_model, torch.cat(chief_memory.select_chief_feature_pool, dim=0).to(center.device)).detach(),
                    test_embedding=chief_wsi_embedding(chief_model, torch.cat(join_embeding, dim=0).to(center.device)).detach(),
                    label=label,
                    classifier=classifier,
                    Q=Q,
                ).detach()
                # print(f"reward: {reward}")
            else:
                reward = torch.tensor([0.0], dtype=torch.float, device=center.device)
                # print("No")

        chief_memory.select_chief_feature_pool.append(feature_group.detach())
    else:
        reward = torch.tensor([0.0], dtype=torch.float, device=center.device)


    if wandb:
        chief_memory.rewards.append(reward)
        wandb.log({
            "chief_ppo/reward": reward.item(),
        })

    return {
        "chief_memory": chief_memory,
    }


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mcd_forward(model, x, T=25, dropout_rate=0.1):
    model.eval()
    model.set_dropout_rate(dropout_rate)
    enable_dropout(model)

    preds = []
    for _ in range(T):
        logits = model(x)                      # [B, 2]
        probs = torch.softmax(logits, dim=-1)  # ⭐ logits → prob
        preds.append(probs.unsqueeze(0))       # [1, B, 2]

    preds = torch.cat(preds, dim=0)            # [T, B, 2]
    mean = preds.mean(0)
    std  = preds.std(0)
    return mean, std


# ----------------------------------------------------------
# ⭐ 專門用來算 CP quantile（只在 VAL 用）
# ----------------------------------------------------------
@torch.no_grad()
def compute_cp_Q(
    model,
    calib_loader,
    device="cuda",
    alpha=0.1,
    T=25,
    dropout_rate=0.1
):
    model.to(device)
    all_scores = []

    for idx, (_, xb, _, yb, _) in enumerate (tqdm(calib_loader)):
        xb = xb.squeeze(0).to(device)
        yb = yb.to(device)

        mean_cal, std_cal = mcd_forward(model, xb, T=T, dropout_rate=dropout_rate)

        eps = 1e-8
        scores = torch.abs(mean_cal - yb) / (std_cal + eps)
        all_scores.append(scores.cpu())

    S = torch.cat(all_scores, dim=0)  # [N, D]
    N, D = S.shape

    Q_list = []
    for d in range(D):
        s = S[:, d].numpy()
        s_sorted = np.sort(s)
        k = int(np.ceil((N + 1) * (1 - alpha))) - 1
        k = np.clip(k, 0, N - 1)
        Q_list.append(s_sorted[k])

    return torch.tensor(Q_list).float().to(device)   # [D]

@torch.no_grad()
def mcd_cp_inference(model, x, Q, T=25, dropout_rate=0.1):
    mean, std = mcd_forward(model, x, T=T, dropout_rate=dropout_rate)

    Q_ = Q.view(1, -1)      # [1, D]
    interval = Q_ * std     # [B, D]
    lower = mean - interval
    upper = mean + interval

    return mean, lower, upper, std


def train_chief_epoch(
    wandb,
    epoch,
    train_loader,
    device,
    agent_train_step,
    ppo_chief,
    classifier_chief,
    chief_noise_robust_model,
    chief_model,
    chief_memory,
    grouping_instance,
    chief_wsi_embedding,
    optimizer_chief,
    optimizer_noise_robust_chief,
    noise_robust_loss,
    noise_free_loss,
    conf_func,
    robust_loss_fn,
    calculate_metrics,
    ppo_feature_list,
    ppo_label_list,
    ppo_reward_list, 
    cluster_list,
    lambda_reg=1e-4,
    Q=None,
):
    """
    執行一次 Chief 模型與 Noise-Robust 模型的訓練 Epoch。

    Args:
        epoch: int, 當前 epoch
        train_loader: DataLoader
        device: torch.device
        args: argparse.Namespace，包含 num_epochs 等參數
        ppo_chief: PPO 代理
        classifier_chief: 主分類器
        chief_noise_robust_model: Noise-robust 分類器
        chief_model: 特徵提取模型
        chief_memory: RL 記憶體
        grouping_instance: 負責 patch grouping 的模組
        chief_wsi_embedding: 特徵聚合函式
        optimizer_chief, optimizer_noise_roust_chief: 優化器
        noise_robust_loss, noise_free_loss: loss 函式
        conf_func, robust_loss_fn: 噪聲控制函式
        calculate_metrics: 指標計算函式
        lambda_reg: float, 兩模型正則化係數
    """

    Y_prob_list = []
    Y_noise_robust_prob_list = []
    label_list = []

    chief_loss = 0
    noise_loss = 0
    reg_loss = 0

    for ide, (coords, chief_data, _, label, name) in enumerate(tqdm(train_loader)):

        num_patch = chief_data.size(1)
        label_correction = epoch >= 1

        coords = coords.squeeze(dim=3)
        update_chief_coords, update_chief_data, label = (
            coords.to(device),
            chief_data.squeeze(0).to(device),
            label.to(device).long(),
        )

        classifier_chief.train()
        chief_noise_robust_model.train()

        # print(cluster_list)
        centers, groups = cluster_list[name[0]]["centers"], cluster_list[name[0]]["groups"]

        for center, feature_group in zip(centers, groups):
            select_and_embed_chief_patches(
                ppo_chief=ppo_chief,
                chief_memory=chief_memory,
                center=center.unsqueeze(0),
                Now_picked_num=len(chief_memory.select_chief_feature_pool),
                Total_picked_num=len(centers),
                feature_group=feature_group,
                label=label,
                classifier=classifier_chief,
                chief_model=chief_model,
                Q=Q,
                wandb=wandb,
            )

        if len(chief_memory.select_chief_feature_pool) == 0:
            # 若沒有任何被選到的 feature，就建立一個 [1, 768] 的 0 tensor
            wsi_embedding_chief = torch.zeros((1, 768), device=device)
        else:
            wsi_embedding_chief = chief_wsi_embedding(
                chief_model,
                torch.cat(chief_memory.select_chief_feature_pool, dim=0).to(device)
            ).detach()

        result_main_classifier = train_noise_robust_and_free_models(
            wsi_embedding_chief=wsi_embedding_chief,
            label=label,
            chief_noise_robust_model=chief_noise_robust_model,
            classifier_chief=classifier_chief,
            optimizer_noise_robust_chief=optimizer_noise_robust_chief,
            optimizer_chief=optimizer_chief,
            noise_robust_loss=noise_robust_loss,
            noise_free_loss=noise_free_loss,
            robust_loss_fn=robust_loss_fn,
            conf_func=conf_func,
            label_correction=label_correction,
            lambda_reg=lambda_reg,
            beta_coe=0.5,
            Q=Q,
        )

        chief_loss += result_main_classifier["loss_free"]
        noise_loss += result_main_classifier["loss_robust"]
        reg_loss += result_main_classifier["reg_term"]

        Y_prob_list.append(result_main_classifier["probs_free"].cpu())
        Y_noise_robust_prob_list.append(result_main_classifier["probs_robust"].cpu())
        label_list.append(label.cpu())

        select_patch = 0
        for i in chief_memory.select_chief_feature_pool:
            select_patch += i.size(0)

        # === RL Reward ===
        chief_classifier_reward = result_main_classifier["reward"][0][label.item()].unsqueeze(0).detach()
        if select_patch == 0:
            chief_memory.rewards[-1] += torch.tensor([-5], device=device, dtype=torch.float)
        elif select_patch == num_patch:
            chief_memory.rewards[-1] += torch.tensor([-1], device=device, dtype=torch.float)
        else:
            chief_memory.rewards[-1] += chief_classifier_reward.float()
        

        # === Logging ===
        wandb.log({
            "chief_ppo/select_patch": select_patch/num_patch,
            "chief_ppo/pro_reward": chief_classifier_reward.item(),
            "probability/chief_true_label": result_main_classifier["probs_free"][0][label.item()].detach(),
            "probability/chief_false_label": result_main_classifier["probs_free"][0][1 - label.item()].detach(),
        })

        # === PPO Update ===

        chief_memory.actions.insert(0, -1)
        chief_memory.logprobs.insert(0, -1)
        policy_loss, value_loss, total_loss = ppo_chief.update(chief_memory)
        wandb.log({
            "chief_ppo/policy_loss": policy_loss,
            "chief_ppo/value_loss": value_loss,
            "chief_ppo/total_loss": total_loss,
        })
        agent_train_step += 1
        chief_memory.clear_memory()


    # === Epoch Metrics ===
    targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
    probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
    precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
    wandb.log({
        "train_chief/precision": precision,
        "train_chief/recall": recall,
        "train_chief/f1": f1,
        "train_chief/auc": auc,
        "train_chief/acc": accuracy,
        "train_chief/loss": chief_loss / len(train_loader),
        "train_chief/reg_loss": reg_loss / len(train_loader),
    })

    probs = np.asarray(torch.cat(Y_noise_robust_prob_list, dim=0).detach().cpu().numpy())
    precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
    wandb.log({
        f"noise_Model/precision": precision,
        f"noise_Model/recall": recall,
        f"noise_Model/f1": f1,
        f"noise_Model/auc": auc,
        f"noise_Model/acc": accuracy,
        f"noise_Model/loss": noise_loss / len(train_loader),
    })

    return agent_train_step, ppo_feature_list, ppo_label_list, ppo_reward_list


def group_and_get_centers(update_chief_data, k=100):

    # 準備資料
    if isinstance(update_chief_data, torch.Tensor):
        device = update_chief_data.device
        update_chief_data_np = update_chief_data.detach().cpu().numpy()
    else:
        device = "cpu"
        update_chief_data_np = update_chief_data

    N = len(update_chief_data_np)
    n_clusters = max(2, N // k)

    # 執行 KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels_np = kmeans.fit_predict(update_chief_data_np)
    centers_np = kmeans.cluster_centers_

    # 轉回 tensor
    centers = torch.as_tensor(centers_np, device=device, dtype=torch.float32)

    # 依群標籤取出對應的 feature
    groups = []
    for i in range(n_clusters):
        cluster_feats = update_chief_data[torch.as_tensor(labels_np == i, device=device)]
        # print(cluster_feats.shape)
        groups.append(cluster_feats)

    return centers, groups


def cluster_chief_data(train_loader, device="cpu", save_path=None):
    """
    使用 group_and_get_centers() 自動分群，
    回傳 dict，以 name 為 key：
        cluster_record[name] = {
            "centers": torch.Tensor,
            "groups": list[Tensor],
            "label": torch.LongTensor([label])
        }
    """
    cluster_record = {}

    for ide, (coords, chief_data, _, label, name) in enumerate(tqdm(train_loader)):
        # 移到 device
        if isinstance(chief_data, torch.Tensor):
            chief_data = chief_data.to(device)

        # 分群
        centers, groups = group_and_get_centers(chief_data.squeeze(0))

        label_long = torch.tensor(label.item(), dtype=torch.long, device=device)

        # name 可能是 list
        if isinstance(name, (list, tuple)):
            for n in name:
                cluster_record[n] = {
                    "centers": centers,
                    "groups": groups,
                    "label": label_long,    # ← long tensor
                }
        else:
            cluster_record[name] = {
                "centers": centers,
                "groups": groups,
                "label": label_long,        # ← long tensor
            }

    # optional save
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(cluster_record, f)

    return cluster_record



def train(args, ppo_chief, ppo_gigapath, classifier_chief, classifier_giga, chief_noise_robust_model, gigapath_rewardModels, gigapath_model, chief_memory, train_classifier_memory, train_loader, validation_loader, test_loader=None, wandb=None):
    
    run_name = f"{args.csv.split('/')[-1].split('.')[0]}"
    save_dir = os.path.join(args.save_dir, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer_noise_robust_chief = torch.optim.Adam(chief_noise_robust_model.parameters(),lr=3e-5,weight_decay=1e-5)
    optimizer_chief = torch.optim.Adam(classifier_chief.parameters(),lr=3e-5,weight_decay=1e-5)

    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()

    none_epoch = 0
    best_chief_auc = 0
    agent_train_step = 0

    ppo_feature_list = []
    ppo_label_list = []
    ppo_reward_list = []

    grouping_instance = grouping(action_size=args.action_size)
    robust_loss_fn = GCELoss(q=0.7)
    conf_func = LabelConfidence(m=0.0, momentum=0.7)

    # cluster_list = cluster_chief_data(train_loader, device="cpu")
    # cluster_list = cluster_chief_data(train_loader, device="cpu", save_path=os.path.join(save_dir, "cluster_train.pth"))
    # cluster_list_val = cluster_chief_data(validation_loader, device="cpu")
    # cluster_list_val = cluster_chief_data(validation_loader, device="cpu", save_path=os.path.join(save_dir, "cluster_val.pth"))
    pkl_path = os.path.join(args.chief_feature_dir, "cluster_record_AgglomerativeComplete.pkl")
    with open(pkl_path, "rb") as f:
        cluster_record = pickle.load(f)

    Q = None
    

    print("Training !!!") 
    for idx, epoch in enumerate(range(args.num_epochs)):
        
        agent_train_step, ppo_feature_list, ppo_label_list, ppo_reward_list = train_chief_epoch(
            wandb=wandb,
            epoch=epoch,
            train_loader=train_loader,
            device=device,
            agent_train_step=agent_train_step,
            ppo_chief=ppo_chief,
            classifier_chief=classifier_chief,
            chief_noise_robust_model=chief_noise_robust_model,
            chief_model=chief_model,
            chief_memory=chief_memory,
            grouping_instance=grouping_instance,
            chief_wsi_embedding=chief_wsi_embedding,
            optimizer_chief=optimizer_chief,
            optimizer_noise_robust_chief=optimizer_noise_robust_chief,
            noise_robust_loss=noise_robust_loss,
            noise_free_loss=noise_free_loss,
            conf_func=conf_func,
            robust_loss_fn=robust_loss_fn,
            calculate_metrics=calculate_metrics,
            ppo_feature_list = ppo_feature_list,
            ppo_label_list = ppo_label_list,
            ppo_reward_list = ppo_reward_list,
            cluster_list=cluster_record,
            Q=Q,
        )
        
        wandb.log({
            "epoch": epoch,
        })

        # val
        chief_auc, targets, probs = test(args,ppo_chief,[classifier_chief],train_classifier_memory, cluster_record, validation_loader, chief_model, run_type="val", epoch=epoch, wandb=wandb)
        # _ = test(args,ppo_chief,[classifier_chief],train_classifier_memory,test_loader, chief_model, run_type="test", epoch=epoch, wandb=wandb)

        Q = compute_cp_Q(classifier_chief, validation_loader)
        ece_before = compute_binary_ece(probs[:,1], targets)
        print("[Calibration] ECE (before):", ece_before)

        
        if chief_auc >= best_chief_auc:
            best_chief_auc = chief_auc
            none_epoch = 0
            # save model
            torch.save(classifier_chief.state_dict(), os.path.join(save_dir, f"classifier_chief.pth"))
            torch.save(chief_noise_robust_model.state_dict(), os.path.join(save_dir, f"classifier_noise_model.pth"))
            ppo_chief.save(save_dir, "ppo_chief")
        elif none_epoch >= args.patience and epoch >= 70:
            print(f"Break at epoch {epoch}.")
            break

        none_epoch += 1

def sample_all_clusters(cluster_record, chief_model, ratio=0.4):

    data_list = []   # list of (embedding, label)

    for slide_name, entry in cluster_record.items():
        centers = entry["centers"]
        groups = entry["groups"]
        label = entry["label"].long()   # torch.long
        # print(label.shape)

        if centers is None or len(groups) == 0:
            continue

        k = centers.shape[0]
        num_select = max(1, int(k * ratio))
        selected_idx = random.sample(range(k), num_select)

        selected_groups = [groups[i] for i in selected_idx]

        # flatten
        all_feats = torch.cat(selected_groups, dim=0).to(next(chief_model.parameters()).device)

        # compute wsi embedding
        wsi_embedding_chief = chief_wsi_embedding(
            chief_model,
            all_feats
        ).detach()

        # ⭐ 直接存成 tuple
        data_list.append((wsi_embedding_chief, label))

    return data_list

def build_dataloader_from_list(tensor_list, batch_size=8, shuffle=True):
    return DataLoader(tensor_list, batch_size=batch_size, shuffle=shuffle)


class BetaCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        # parameters
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, p):
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        logit = self.a * torch.log(p) + self.b * torch.log(1 - p) + self.c
        return torch.sigmoid(logit)

    def fit(self, x, labels, lr=0.01, max_iter=2000):

        def is_prob_tensor(t):
            if t.dim() == 2:
                row_sum = t.sum(dim=1)
                return torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3)
            else:
                return (t.min() >= 0) and (t.max() <= 1)

        # ------------------------------
        # 判斷 x 是 prob 還是 logits
        # ------------------------------
        if x.dim() == 2:
            if is_prob_tensor(x):      # prob [N,2]
                probs = x[:, 1]
            else:                      # logits [N,2]
                probs = torch.softmax(x, dim=1)[:, 1]
        else:
            if is_prob_tensor(x):      # prob [N]
                probs = x
            else:                      # logits [N]
                probs = torch.sigmoid(x)

        probs = probs.detach().clamp(1e-8, 1-1e-8)
        labels = labels.detach().float()

        # ------------------------------
        # Optimize (LBFGS)
        # ------------------------------
        optimizer = optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        bce = nn.BCELoss()

        def closure():
            optimizer.zero_grad()
            pred = self.forward(probs)
            loss = bce(pred, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return (self.a.item(), self.b.item(), self.c.item())
    

def collect_logits_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)   # assume output is [N,2]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels


def compute_binary_ece(probs, labels, n_bins=15):
    """
    probs: predicted prob of class=1, shape [N]
    labels: {0,1}, shape [N]
    """
    ece = 0.0
    bins = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        left, right = bins[i], bins[i+1]
        mask = (probs >= left) & (probs < right)
        if mask.sum() == 0:
            continue

        bin_probs = probs[mask]
        bin_labels = labels[mask].float()

        avg_prob = bin_probs.mean()
        avg_true = bin_labels.mean()

        ece += (mask.float().mean() * torch.abs(avg_prob - avg_true))

    return ece.item()


def run_beta_calibration(probs, labels, device, n_bins=15):
    """
    執行 Beta Calibration + ECE 計算，並回傳:
        (beta calibrator, 原本 ECE, 校準後 ECE)
    """
    # === 原始機率 ===
    ece_before = compute_binary_ece(probs[:, 1], labels, n_bins=n_bins)
    print("[Calibration] ECE (before):", ece_before)

    # === Beta Calibration ===
    beta = BetaCalibrator()
    params = beta.fit(probs, labels)
    # print("[Calibration] Beta params (a, b, c):", params)

    # === 校準後機率 ===
    calibrated_probs = beta(probs[:, 1])
    ece_after = compute_binary_ece(calibrated_probs, labels, n_bins=n_bins)
    print("[Calibration] ECE (after):", ece_after)

    beta = beta.to(device)

    return beta, ece_before, ece_after


def train_baseline(args,basedmodel,classifymodel_top,classifymodel_bottom,FusionHisF,memory_space,train_loader, validation_loader, test_loader, wandb):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifymodel_top.to(device)
    classifymodel_bottom.to(device)
    optimizer_top = torch.optim.Adam(
        list(classifymodel_top.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    optimizer_bottom = torch.optim.Adam(
        list(classifymodel_bottom.parameters()),
        lr=1e-4, weight_decay=1e-5
    )

    run_name = f"{args.csv.split('/')[-1].split('.')[0]}"
    save_dir = os.path.join(args.save_dir, run_name)

    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'./model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()
    # print(next(chief_model.parameters()).device)
    
    none_epoch = 0
    best_auc_top = 0

    cluster_list = cluster_chief_data(train_loader, device="cpu", save_path=os.path.join(save_dir, "train_cluster.pkl"))
    embedding_list = sample_all_clusters(cluster_list, chief_model, ratio=0.4)
    train_loader = build_dataloader_from_list(embedding_list, batch_size=32)


    cluster_list_val = cluster_chief_data(validation_loader, device="cpu", save_path=os.path.join(save_dir, "val_cluster.pkl"))
    embedding_list_val = sample_all_clusters(cluster_list_val, chief_model, ratio=0.4)
    validation_loader = build_dataloader_from_list(embedding_list_val, batch_size=32, shuffle=False)

    cluster_list_test = cluster_chief_data(test_loader, device="cpu", save_path=os.path.join(save_dir, "test_cluster.pkl"))
    embedding_list_test = sample_all_clusters(cluster_list_test, chief_model, ratio=0.4)
    test_loader = build_dataloader_from_list(embedding_list_test, batch_size=32, shuffle=False)


    for idx, epoch in enumerate(range(args.num_epochs)):
        classifymodel_bottom.train()
        classifymodel_top.train()

        epoch_loss = 0.0
        epoch_loss_bottom = 0.0

        label_list = []
        Y_prob_list = []
        Y_prob_list_bottom = []

        for _, (chief_data, label) in enumerate(tqdm(train_loader)):
            # print(label.shape)
            optimizer_top.zero_grad()
            optimizer_bottom.zero_grad()
            label = label.to(device).long()
            chief_data = chief_data.to(device)

            W_logits = classifymodel_top(chief_data).squeeze(1)
            W_Y_prob = F.softmax(W_logits, dim=1)

            # 計算 loss（以交叉熵為例）
            #print(W_logits.shape)
            #print(label.shape)
            loss_WSI = F.cross_entropy(W_logits, label)
            loss = loss_WSI
            loss.backward()
            optimizer_top.step()

            #print(f'Idx: {idx},loss: {loss.item()}')

            # reward

            # 計算正確率
            epoch_loss += loss.item()
            label_list.append(label)
            Y_prob_list.append(W_Y_prob)

            # W_logits = classifymodel_bottom(update_data_bottom).squeeze(1)
            # W_Y_prob = F.softmax(W_logits, dim=1)

            # # 計算 loss（以交叉熵為例）
            # #print(W_logits.shape)
            # #print(label.shape)
            # loss_WSI = F.cross_entropy(W_logits, label)
            # loss = loss_WSI
            # loss.backward()
            # optimizer_bottom.step()

            # epoch_loss_bottom += loss.item()
            # Y_prob_list_bottom.append(W_Y_prob)


        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)

        wandb.log({
            "loss_top": epoch_loss/len(train_loader),
            "epoch": epoch,
            "train_top/precision": precision,
            "train_top/recall": recall,
            "train_top/f1": f1,
            "train_top/auc": auc,
            "train_top/acc": accuracy
        })

        # probs = np.asarray(torch.cat(Y_prob_list_bottom, dim=0).detach().cpu().numpy())
        # precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)

        # wandb.log({
        #     "loss_bottom": epoch_loss_bottom/len(train_loader),
        #     "epoch": epoch,
        #     "train_bottom/precision": precision,
        #     "train_bottom/recall": recall,
        #     "train_bottom/f1": f1,
        #     "train_bottom/auc": auc,
        #     "train_bottom/acc": accuracy
        # })

        #acc = correct / total
        #print(f"[Epoch {epoch+1}/{args.num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")
        precision, recall, f1, val_auc_top, val_accuracy = test_baseline(args,chief_model,classifymodel_top,FusionHisF,memory_space,validation_loader, "val", chose="top")
        wandb.log({
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "val/auc": val_auc_top,
            "val/acc": val_accuracy
        })
        precision, recall, f1, val_auc_bottom, accuracy = test_baseline(args,chief_model,classifymodel_top,FusionHisF,memory_space,test_loader, "test", chose="top")
        wandb.log({
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
            "test/auc": val_auc_bottom,
            "test/acc": val_accuracy
        })


        if val_auc_top >= best_auc_top:
            best_auc_top = val_auc_top
            # save model
            print(f'Save model at epoch {epoch}.')
            print(f'val auc: {val_auc_top}')
            torch.save(classifymodel_top.state_dict(), os.path.join(save_dir, "classifymodel.pth"))
            none_epoch = 0
        elif none_epoch >= 20:
            print(f'Break at epoch {epoch}.')
            break
            
            
        # if val_auc_bottom >= best_auc_bottom:
        #     best_auc_bottom = val_auc_bottom
        #     # save model
        #     print(f'Save model at epoch {epoch}.')
        #     print(f'val auc: {val_auc_bottom}')
        #     torch.save(classifymodel_bottom.state_dict(), os.path.join(save_dir, "classifymodel_bottom.pth"))
        #     none_epoch = 0

        none_epoch += 1


def interpolate_probs(probs, new_length,action_size):
    b,c = probs.shape
    x = np.linspace(0, 1, c)
    x_new = np.linspace(0, 1, new_length)
    new_probs = np.interp(x_new, x, probs.view(-1).cpu().numpy())
    interpolate_action_probs = new_probs / new_probs.sum()   
    index = np.random.choice(np.arange(new_length), size=action_size, p=interpolate_action_probs, replace=False)
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
          
    
    def action_make_subbags(self ,memory, action_index_pro, update_coords,features,action_size= None,restart= False,delete_begin=False): 
        
        B, N, C = update_coords.shape
        idx = interpolate_probs(action_index_pro, new_length = N ,action_size = action_size)
        idx_recored = idx
        idx = torch.tensor(idx)
        features_group = features[:, idx[:], :]
        action_group = update_coords[:, idx[:], :] 

        idx = torch.unique(idx)
        mask = torch.ones(features.size(1), dtype=torch.bool) 
        mask[idx] = False  
        updated_features = features[:, mask, :] 
        updated_coords = update_coords[:, mask, :]
        memory.coords_actions.append(action_group)
        return idx_recored, features_group, updated_coords, updated_features , memory