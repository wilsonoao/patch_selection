import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

# ---------- 1) Q(s,i) 模型 ----------
class QNet(nn.Module):
    """輸入 (state,item) 的聯合特徵，輸出 item-level Q(s,i)。"""
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: [B, in_dim]
        return self.mlp(x).squeeze(-1)  # [B]

# ---------- 2) Choice model：P(i | s, A) ----------
def choice_probs(v_scores: torch.Tensor, slate_idx: List[int], null_score: float = 0.0):
    """
    v_scores: [N_items]，對應當前 state 下所有候選的吸引度 v(s,i)（非負）。
    slate_idx: A 的 item 索引（在 v_scores 的索引）。
    回傳: p: dict{i -> P(i|s,A)}；此處忽略顯式 ⊥（等價於分母加 v(s,⊥)）
    """
    vs = v_scores[slate_idx]  # [k]
    denom = vs.sum() + torch.tensor(null_score, dtype=vs.dtype, device=vs.device)
    p = vs / (denom + 1e-12)
    return {i: p[t].item() for t, i in enumerate(slate_idx)}

# ---------- 3) SlateQ Agent ----------
class SlateQAgent:
    def __init__(self, qnet: nn.Module, lr=1e-3, gamma=0.99, alpha=None):
        self.qnet = qnet
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.gamma = gamma

    def q_values(self, phi_batch: torch.Tensor) -> torch.Tensor:
        return self.qnet(phi_batch)

    # ---- SARSA 更新（式 14）----
    def update_sarsa(self,
                     s_clicked_i_feat: torch.Tensor,  # phi(s, i_clicked) : [d]
                     r: float,
                     next_A_feats: List[torch.Tensor],  # [phi(s', j) for j in A']
                     next_A_v: torch.Tensor             # v(s', j) for j in A'，shape [k']
                    ):
        self.opt.zero_grad()

        # Q(s, i_clicked)
        q_si = self.qnet(s_clicked_i_feat.unsqueeze(0)).squeeze(0)  # scalar

        if len(next_A_feats) == 0:
            target = torch.tensor(r, dtype=q_si.dtype, device=q_si.device)
        else:
            phi_next = torch.stack(next_A_feats, dim=0)            # [k', d]
            q_next = self.qnet(phi_next)                           # [k']
            probs = next_A_v / (next_A_v.sum() + 1e-12)            # P(j|s',A') （忽略 ⊥）
            exp_q = (probs * q_next).sum()
            target = torch.tensor(r, dtype=q_si.dtype, device=q_si.device) + self.gamma * exp_q

        loss = F.mse_loss(q_si, target.detach())
        loss.backward()
        self.opt.step()
        return loss.item()

    # ---- Q-learning 更新（式 15）----
    def update_qlearning(self,
                         s_clicked_i_feat: torch.Tensor,     # phi(s, i_clicked)
                         r: float,
                         next_candidate_feats: List[torch.Tensor],  # 候選池 C(s') 的特徵
                         next_candidate_v: torch.Tensor,     # v(s', ·) over C(s')，shape [M]
                         k: int,
                         slate_opt: str = "topk"             # "topk" 或 "greedy"
                        ):
        self.opt.zero_grad()

        q_si = self.qnet(s_clicked_i_feat.unsqueeze(0)).squeeze(0)

        if len(next_candidate_feats) == 0:
            target = torch.tensor(r, dtype=q_si.dtype, device=q_si.device)
        else:
            # 先算所有候選的 Q(s', ·)
            phi_next_all = torch.stack(next_candidate_feats, dim=0)   # [M, d]
            q_next_all = self.qnet(phi_next_all)                      # [M]

            # 由 Q 與 v 構建 A'* 最優 slate（approx: top-k / greedy）
            if slate_opt == "topk":
                # 以 v * Q 排序取前 k
                scores = next_candidate_v * q_next_all.clamp_min(0)   # 常見穩定作法
                idx = torch.topk(scores, k=min(k, scores.numel())).indices.tolist()
            elif slate_opt == "greedy":
                idx = greedy_indices(v=next_candidate_v, q=q_next_all, k=k)
            else:
                raise ValueError("slate_opt must be 'topk' or 'greedy'.")

            v_sel = next_candidate_v[idx]                             # [k]
            q_sel = q_next_all[idx]                                   # [k]
            probs = v_sel / (v_sel.sum() + 1e-12)                     # 近似 P(j|s',A'*)
            max_exp_q = (probs * q_sel).sum()

            target = torch.tensor(r, dtype=q_si.dtype, device=q_si.device) + self.gamma * max_exp_q

        loss = F.mse_loss(q_si, target.detach())
        loss.backward()
        self.opt.step()
        return loss.item()

# ---------- 4) Slate 選擇（Serving / 訓練內 max A'） ----------
def select_slate_topk(v_scores: torch.Tensor, q_scores: torch.Tensor, k: int) -> List[int]:
    """Top-k：依 v * Q 取前 k（論文中的常用近似）。"""
    s = v_scores * q_scores.clamp_min(0)
    idx = torch.topk(s, k=min(k, s.numel())).indices.tolist()
    return idx

def greedy_indices(v: torch.Tensor, q: torch.Tensor, k: int) -> List[int]:
    """
    Greedy：逐個加入，考慮邊際 vQ / (v_⊥ + sum 已選 v) 的效益（近似論文描述）。
    """
    chosen = []
    remaining = set(range(v.numel()))
    denom = 0.0
    for _ in range(min(k, v.numel())):
        best_i, best_val = None, -1e9
        for j in list(remaining):
            # 以當前部份 slate 的近似 marginal gain 打分
            gain = (v[j] * q[j]) / (1e-12 + denom + v[j])
            if gain > best_val:
                best_val, best_i = gain, j
        chosen.append(best_i)
        remaining.remove(best_i)
        denom += v[best_i].item()
    return chosen
