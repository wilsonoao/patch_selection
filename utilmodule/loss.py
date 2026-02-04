import torch
import torch.nn as nn
import torch.nn.functional as F


def dirichlet_group_prior_loss(
    alpha,
    group_attn,
    group_ids,
    theta_min=0.2,
    theta_max=2.2,
    k=4.0,
    eps=1e-12,
):

    """
    alpha:      (N,) or (1,N) patch attention (sum=1)
    group_attn: (G,) group-level importance (softmax or sigmoid)
    group_ids:  (G,) long tensor in [0, G-1]
    """

    log = {}

    if alpha.dim() == 2:
        alpha = alpha.squeeze(0)

    alpha = alpha.clamp(min=eps)
    G = len(group_ids)

    # map group importance -> Dirichlet concentration
    # ranks = torch.argsort(torch.argsort(group_attn, descending=True)).float()
    # r = ranks / (G - 1 + 1e-8)
    # theta_g = theta_min + (theta_max - theta_min) * r
    theta_g = theta_min + (theta_max - theta_min) * torch.sigmoid(k * (group_attn - group_attn.mean()))
    # theta_g = theta_min + (theta_max - theta_min) * torch.sigmoid(k * (group_attn - 1/G))
    theta_g = theta_g.clamp(min=eps)
    # print(f"group atten: {group_attn}, mean attn: {group_attn.mean()}")
    # print(f"theta_g: {theta_g}")

    log["theta_g"] = theta_g

    losses = []

    for g in range(G):
        idx = torch.as_tensor(
            group_ids[g],
            dtype=torch.long,
            device=alpha.device,
        )
        if idx.numel() <= 1:
            continue  # skip tiny groups

        p = alpha[idx]
        # print(f"theta_g: {theta_g[g]}")
        # print("p 原本分布:")
        # print(p)
        p = p / (p.sum() + eps)
        # p = p.clamp(min=eps)
        # print("p 分布: ")
        # print(p)

        n_g = p.numel()
        th = theta_g[g]

        # entropy = -torch.sum(p * torch.log(p))
        # entropy_norm = entropy / torch.log(
        #     torch.tensor(n_g, device=p.device, dtype=p.dtype)
        # )

        # print(entropy_norm.item())

        # target entropy h(theta) in [0, 1]
        # h_theta = th / (th + 2)

        # squared error loss
        # loss = (entropy_norm - h_theta) ** 2
        # print(loss)

        log_prob = (
            torch.lgamma(n_g * th)
            - n_g * torch.lgamma(th)
            + (th - 1.0) * torch.sum(torch.log(p))
        )
        # log_prob = (th - 1.0) * torch.sum(torch.log(p))
        # print(log_prob)

        # losses.append(loss)
        losses.append(-log_prob)

    if len(losses) == 0:
        return alpha.new_tensor(0.0)

    return torch.stack(losses).mean(), log


class MILLoss(nn.Module):
    def __init__(
        self,
        criterion,
        use_dirichlet=False,
        dirichlet_weight=0.0,
        dirichlet_kwargs=None,
    ):
        """
        criterion: classification loss (e.g. nn.CrossEntropyLoss)

        use_dirichlet:      whether to use Dirichlet attention prior
        dirichlet_weight:  lambda for Dirichlet loss
        dirichlet_kwargs:  dict for Dirichlet hyperparameters
        """
        super().__init__()
        self.criterion = criterion

        self.use_dirichlet = use_dirichlet
        self.dirichlet_weight = dirichlet_weight
        self.dirichlet_kwargs = dirichlet_kwargs or {}

    def forward(
        self,
        logits,
        label,
        attn=None,
        group_attn=None,
        group_ids=None,
    ):
        """
        logits:     [1, num_classes]
        label:      [1]

        attn:       [N] or [1, N] patch attention
        group_attn: [G] group importance
        group_ids:  [N] patch -> group index

        return:
            dict with individual losses and total loss
        """
        losses = {}

        # ---- classification loss ----
        ce_loss = self.criterion(logits, label)
        losses["ce_loss"] = ce_loss

        total_loss = ce_loss

        # ---- optional Dirichlet prior ----
        if self.use_dirichlet:
            assert attn is not None, "attn required when use_dirichlet=True"
            assert group_attn is not None, "group_attn required when use_dirichlet=True"
            assert group_ids is not None, "group_ids required when use_dirichlet=True"

            dir_loss, log = dirichlet_group_prior_loss(
                attn,
                group_attn,
                group_ids,
                **self.dirichlet_kwargs,
            )
            losses["log_theta_g"] = log["theta_g"].detach().cpu()


            # losses["dirichlet_loss"] = self.dirichlet_weight * torch.mean(attn ** 2)
            losses["dirichlet_loss"] = self.dirichlet_weight * dir_loss
            total_loss = total_loss + losses["dirichlet_loss"]

        losses["loss"] = total_loss
        return losses