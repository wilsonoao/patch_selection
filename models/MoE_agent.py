import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.nn as nn
import os
from torch.distributions import Normal, TransformedDistribution, SigmoidTransform, AffineTransform, Categorical

def load_balancing_loss(probs, selected_idx, num_experts=2, alpha=1):
    B = probs.shape[0]
    f = torch.zeros(num_experts, device=probs.device)
    f.scatter_add_(0, selected_idx.squeeze(-1), torch.ones_like(selected_idx.squeeze(-1), dtype=torch.float))
    f = f / B
    P = probs.sum(dim=0) / B
    aux_loss = alpha * num_experts * torch.sum(f * P)
    return aux_loss

def importance_loss_from_memory(memory, weight=0.1, eps=1e-8):
    """
    根據 memory.expert_select_logprobs 計算 Importance Loss。
    假設你儲存的是 [B, num_experts] 的 gate weights（非 log-prob）。
    """
    if len(memory.expert_all_probs) == 0:
        return torch.tensor(0.0, device='cuda')

    gate_tensor = torch.cat(memory.expert_all_probs, dim=0)  # [B_total, num_experts]
    importance = gate_tensor.sum(dim=0)  # [num_experts]

    mean = importance.mean()
    std = importance.std(unbiased=False)

    cv = std / (mean + eps)
    return weight * (cv ** 2)


def get_aux_coeff(epoch, warmup_epochs=20, max_coeff=1.0, min_coeff=0.0):
    # 線性遞減探索強度（可改成指數、餘弦等）
    if epoch < warmup_epochs:
        return max_coeff * (1 - epoch / warmup_epochs)
    else:
        return min_coeff

class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, device, hidden_state_dim=1024, policy_conv=False, action_std=0.1, action_size=2):
        super(ActorCritic, self).__init__()
        
        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim / feature_dim))

        # self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, action_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, 1)
        )

        self.action_var = torch.full((action_size,), action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
        
    def act(self, current_state, memory, restart_batch=False, training=False):
        # state_ini = memory.merge_msg_states[-1].detach()
        # if restart_batch:
        #     del memory.hidden[:]
        #     memory.hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda())
        # msg_state, hidden_output = self.gru(state_ini.view(1, state_ini.size(0), state_ini.size(-1)), memory.hidden[-1])  
        # memory.hidden.append(hidden_output) 
        # action_mean = self.actor(msg_state[0]) 
        state_ini = memory.expert_states[-1].detach()  # shape: [B, state_dim]
        state_ini = state_ini.squeeze(1)
        action_mean = self.actor(state_ini)  # shape: [B, action_size]
        

        expert_select_logits = self.actor(state_ini)  # shape: [B, max_select]
        expert_select_dist = Categorical(logits=expert_select_logits)
        expert_select_sampled = expert_select_dist.sample()        # shape: [B], 值 ∈ [0, max_select-1]
        expert_select_logprob = expert_select_dist.log_prob(expert_select_sampled)

        memory.expert_select_actions.append(expert_select_sampled.detach())        
        memory.expert_select_logprobs.append(expert_select_logprob.detach())    
        memory.expert_all_probs.append(expert_select_dist.probs.detach())

        return expert_select_sampled
    

    def evaluate(self, state, action):
        seq_l, batch_size, state_dim = state.shape
        state = state.squeeze(1)
        action = action.view(seq_l * batch_size)

        # ---- Evaluate for k_select head ----
        action_logits = self.actor(state)
        action_dist = Categorical(logits=action_logits)
        action_logprobs = action_dist.log_prob(action)  
        dist_entropy = action_dist.entropy().mean()
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
            state_value.view(seq_l, batch_size), \
            dist_entropy
            

class MoE_agent:
    def __init__(self, feature_dim, state_dim, hidden_state_dim, policy_conv, device,
                 action_std=0.1, lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2, action_size=2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.bagsize = state_dim 

        self.policy = ActorCritic(feature_dim, state_dim, device, hidden_state_dim, policy_conv, action_std, action_size).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(feature_dim, state_dim, device, hidden_state_dim, policy_conv, action_std, action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, data, memory, restart_batch=False, training=True):
        return self.policy_old.act(data, memory, restart_batch, training)

    
    def select_features(self, idx, features,len_now_coords):
        index = idx
        features_group = []
        for i in range(len(index)):
            member_size = (index[i].size)
            if member_size > self.max_size: 
                index[i] = np.random.choice(index[i],size=self.max_size,replace=False)
            temp = features[index[i]]
            temp = temp.unsqueeze(dim=0) 
            features_group.append(temp)
        return features_group

    def update(self, memory, epoch=0):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.moe_rewards):
            discounted_reward = reward.detach()
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards, 0).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_msg_states = torch.stack(memory.expert_states, 0).cuda().detach() 

        old_actions = torch.stack(memory.expert_select_actions[1:], 0).cuda().detach() 
        old_logprobs = torch.stack(memory.expert_select_logprobs[1:], 0).cuda().detach() 

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_msg_states, old_actions)
            rewards = rewards.view(-1,1)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # aux_loss
            selected_probs = logprobs.exp()
            aux_loss = load_balancing_loss(selected_probs, old_actions)
            aux_loss_alpha = get_aux_coeff(epoch, warmup_epochs=20, max_coeff=0.1, min_coeff=0.01)
            # entropy_alpha = get_aux_coeff(epoch, warmup_epochs=10, max_coeff=2, min_coeff=0.01)

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy + aux_loss_alpha * aux_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
    
        return -torch.min(surr1, surr2).mean().item(), self.MseLoss(state_values, rewards).mean().item(), dist_entropy.mean().item(), aux_loss.mean().item(), loss.mean().item()

    def save(self, save_dir):
        torch.save(self.policy_old.state_dict(), os.path.join(save_dir, "MoE.pth"))
