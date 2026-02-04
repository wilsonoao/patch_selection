import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch.nn as nn
import torch.nn.functional as F
import torch
from utilmodule.utils import calculate_metrics
import pickle
import random
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    AffinityPropagation,
    BisectingKMeans
)
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from tqdm import tqdm
import torch.nn.functional as F
import os
from tqdm import tqdm
from models.CHIEF import CHIEF
from models.CHIEF_network import ClfNet
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

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


def group_and_get_centers(update_chief_data, 
                          k=100, 
                          method="kmeans", 
                          min_cluster_size=10, 
                          n_clusters=None,
                          enforce_min_clusters=True,
                          coords=None,
                          spatial_k=None,
                          embed_k=None,
                          spatial_weight=None,
                          resolution=None,
                          ):
    """
    強化版 clustering：
    - 保證群數 >= 2（含 Spectral / Affinity / Ward / Bisect）
    - 保證不會有空群
    - HDBSCAN 若群太少自動 fallback → KMeans
    """

    # ==== prepare data ====
    if isinstance(update_chief_data, torch.Tensor):
        device = update_chief_data.device
        data_np = update_chief_data.detach().cpu().numpy()
    else:
        device = "cpu"
        data_np = update_chief_data

    N, D = update_chief_data.shape
    auto_clusters = max(10, N // k)     # default number of clusters

    method = method.lower()


    # ===============================================================
    # ⭐ METHOD 1: KMeans
    # ===============================================================
    if method == "kmeans":
        # n_clusters = n_clusters or auto_clusters
        n_clusters = 10

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto"
        )
        labels_np = kmeans.fit_predict(data_np)
        centers_np = kmeans.cluster_centers_

        labels = torch.tensor(labels_np, device=device)
        centers = torch.tensor(centers_np, device=device, dtype=torch.float32)
        groups = [update_chief_data[labels == i].detach().clone() for i in range(n_clusters)]

        return centers, groups, labels


    # ===============================================================
    # ⭐ METHOD 2: Bisecting KMeans（保證群數固定且 >=2）
    # ===============================================================
    if method == "bisect" or method == "bisecting":
        n_clusters = n_clusters or auto_clusters

        model = BisectingKMeans(
            n_clusters=n_clusters,
            random_state=42
        )
        labels_np = model.fit_predict(data_np)
        centers_np = model.cluster_centers_

        labels = torch.tensor(labels_np, device=device)
        centers = torch.tensor(centers_np, device=device, dtype=torch.float32)
        groups = [update_chief_data[labels == i].detach().clone() for i in range(n_clusters)]

        return centers, groups, labels


    # ===============================================================
    # ⭐ METHOD 3: HDBSCAN（可能沒群 → fallback）
    # ===============================================================
    if method == "hdbscan":
    
        def run_hdbscan(data_np, min_size):
            clusterer = HDBSCAN(
                min_cluster_size=min_size,
                min_samples=min_size,
                metric='cosine'
            )
            labels_np = clusterer.fit_predict(data_np)
            return labels_np

        # 初始 min_cluster_size
        cur_min_size = min_cluster_size
        labels_np = run_hdbscan(data_np, cur_min_size)
        labels = torch.tensor(labels_np, device=device)

        valid_clusters = [c for c in set(labels_np) if c != -1]
        print(f"valid_clusters: {len(valid_clusters)}")

        # ====== 如果群太少 → 不斷縮小 min_cluster_size ======
        while enforce_min_clusters and len(valid_clusters) < 2 and cur_min_size > 1:
            cur_min_size = max(2, cur_min_size // 2)  # ⭐ 你也可以改成 -1
            print(cur_min_size)

            labels_np = run_hdbscan(data_np, cur_min_size)
            labels = torch.tensor(labels_np, device=device)
            valid_clusters = [c for c in set(labels_np) if c != -1]

        # ====== 如果縮到不能再縮 → fallback KMeans ======
        
        if enforce_min_clusters and len(valid_clusters) < 2:
            print("KMEANS !!!")
            return group_and_get_centers(
                update_chief_data, k=k, method="kmeans"
            )

        # ====== HDBSCAN 正常輸出 ======
        groups = []
        centers_list = []
        for c in valid_clusters:
            feats = update_chief_data[labels == c].detach().clone()
            groups.append(feats)
            centers_list.append(feats.mean(0, keepdim=True))

        centers = torch.cat(centers_list, 0)
        return centers, groups, labels


    # ===============================================================
    # ⭐ METHOD 4: Agglomerative（complete / average / ward）
    # ===============================================================
    if method in ["complete", "average", "ward"]:
        n_clusters = n_clusters or auto_clusters
        

        if method == "ward":
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="ward"
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=method,
                metric="euclidean"
            )

        labels_np = model.fit_predict(data_np)
        labels = torch.tensor(labels_np, device=device)

        groups = []
        centers_list = []
        for c in range(n_clusters):
            feats = update_chief_data[labels == c].detach().clone()
            groups.append(feats)
            centers_list.append(feats.mean(0, keepdim=True))

        centers = torch.cat(centers_list, 0)
        return centers, groups, labels


    # ===============================================================
    # ⭐ METHOD 5: Spectral Clustering（可能 fail → fallback）
    # ===============================================================
    if method == "spectral":
        n_clusters = n_clusters or auto_clusters

        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            assign_labels='kmeans',
            random_state=42
        )

        try:
            labels_np = model.fit_predict(data_np)
        except Exception:
            # fallback
            return group_and_get_centers(update_chief_data, method="kmeans", k=k)

        labels = torch.tensor(labels_np, device=device)

        # Spectral clustering 不會空群
        groups = []
        centers_list = []

        for c in range(n_clusters):
            feats = update_chief_data[labels == c].detach().clone()
            groups.append(feats)
            centers_list.append(feats.mean(0, keepdim=True))

        centers = torch.cat(centers_list, 0)
        return centers, groups, labels


    # ===============================================================
    # ⭐ METHOD 6: Affinity Propagation（需要 fallback）
    # ===============================================================
    if method == "affinity":

        model = AffinityPropagation(random_state=42, damping=0.9)

        try:
            labels_np = model.fit_predict(data_np)
        except Exception:
            return group_and_get_centers(update_chief_data, method="kmeans", k=k)

        # AP 可能只出 1 群 → fallback
        if enforce_min_clusters and len(set(labels_np)) < 2:
            return group_and_get_centers(update_chief_data, method="kmeans", k=k)

        labels = torch.tensor(labels_np, device=device)
        unique_clusters = list(sorted(set(labels_np)))

        groups = []
        centers_list = []
        for c in unique_clusters:
            feats = update_chief_data[labels == c].detach().clone()
            groups.append(feats)
            centers_list.append(feats.mean(0, keepdim=True))

        centers = torch.cat(centers_list, 0)
        return centers, groups, labels
    

    # ===============================================================
    # ⭐ METHOD 7: SpatialLeiden
    # ===============================================================
    if method == "spatialleiden":

        import anndata
        import scanpy as sc
        import squidpy as sq
        import spatialleiden as sl
        import numpy as np

        # ==== prepare data ====
        if isinstance(coords, torch.Tensor):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = coords

        # -------------------------
        # 1. 建 AnnData
        # -------------------------
        adata = anndata.AnnData(data_np)
        adata.obsm["spatial"] = coords_np

        # -------------------------
        # 2. spatial graph
        # -------------------------
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",   # WSI / patch
            n_neighs=spatial_k or 8
        )

        # -------------------------
        # 3. embedding graph
        # -------------------------
        sc.pp.neighbors(
            adata,
            use_rep="X",
            n_neighbors=embed_k or 15
        )

        # -------------------------
        # 4. SpatialLeiden
        # -------------------------
        sl.spatialleiden(
            adata, layer_ratio=1.8, directed=(False, True), random_state=42
        )

        # -------------------------
        # 5. labels
        # -------------------------
        labels_np = adata.obs["spatialleiden"].to_numpy().astype(int)
        labels = torch.tensor(labels_np, device=device)

        # -------------------------
        # 6. groups（完全對齊你 KMeans）
        # -------------------------
        unique_labels = np.unique(labels_np)
        groups = [update_chief_data[labels == i].detach().clone() for i in unique_labels]

        # -------------------------
        # 7. centers（用 mean embedding）
        # -------------------------
        centers = torch.stack([
            update_chief_data[labels == i].mean(dim=0)
            for i in unique_labels
        ]).to(device)

        orig_ids = np.arange(len(labels_np))

        unique_labels = np.unique(labels_np)

        id_list = [
            orig_ids[labels_np == cid].tolist()
            for cid in unique_labels
        ]

        return centers, groups, id_list

    # fallback
    raise ValueError(f"Unknown clustering method: {method}")

def standardize_embeddings(chief_data, eps=1e-6):
    """
    chief_data: Tensor [1, N, 768]
    return: Tensor [1, N, 768] (z-score normalized)
    """

    # remove batch dim since B=1
    x = chief_data.squeeze(0)   # [N, 768]

    mean = x.mean(dim=0, keepdim=True)      # [1, 768]
    std  = x.std(dim=0, keepdim=True)       # [1, 768]

    x_norm = (x - mean) / (std + eps)       # avoid divide-by-zero

    return x_norm.unsqueeze(0)              # [1, N, 768]


def cluster_chief_data(train_loader, device="cpu", method="kmeans"):
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

    for _, (coords, data, label, name) in enumerate(tqdm(train_loader)):
        # 移到 device
        # if name[0] == "TCGA-44-6147-11A-05-TS5.41088AE2-F335-4BA2-B096-F9135208100D.pt":
        #     print(chief_data)
        if isinstance(data, torch.Tensor):
            data = data.to(device)

        # if name[0] != "TCGA-78-7146-01A-01-BS1.d85d10d7-e31b-45ba-a089-dfe554ceb6c2.pt":
        #     continue
            
        
        # 分群
        # chief_data = standardize_embeddings(chief_data)
        centers, groups, group_id = group_and_get_centers(data.squeeze(0), k=50, method=method, min_cluster_size=2, coords=coords.squeeze(0).squeeze(-1))
        

        # name 可能是 list
        if isinstance(name, (list, tuple)):
            for n in name:
                cluster_record[n] = {
                    "centers": centers,
                    "groups": groups,
                    "id": group_id,    # ← long tensor
                }
        else:
            cluster_record[name] = {
                "centers": centers,
                "groups": groups,
                "id": group_id,        # ← long tensor
            }

    return cluster_record


def clustering(save_path, train_loader, validation_loader, test_loader=None):
    

    print(save_path)

    cluster_list = cluster_chief_data(train_loader, device="cpu", method="spatialleiden")
    cluster_list_val = cluster_chief_data(validation_loader, device="cpu", method="spatialleiden")
    cluster_list_test = cluster_chief_data(test_loader, device="cpu", method="spatialleiden")

    all_clusters = {}
    all_clusters.update(cluster_list)
    all_clusters.update(cluster_list_val)
    all_clusters.update(cluster_list_test)


    with open(save_path, "wb") as f:
        pickle.dump(all_clusters, f)

    print(f"Saved merged dict to {save_path}, total keys = {len(all_clusters)}")
 
