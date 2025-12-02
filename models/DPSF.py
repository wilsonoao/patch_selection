import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.nn as nn
import os
from collections import defaultdict
from torch.distributions import Bernoulli, Independent, MultivariateNormal, TransformedDistribution, SigmoidTransform
        
class Memory:
    def __init__(self):
        # action 
        self.actions = [] 
        self.coords_actions = [] 
        self.logprobs = [] 
        
        self.rewards = []
        self.moe_rewards = []
        self.is_terminals = []
        self.hidden = []
        self.select_chief_feature_pool = []
        self.select_gigapath_feature_pool = []
        self.expert_all_probs = []
        device = torch.device("cuda")
        self.last_performance = defaultdict(lambda: torch.zeros(1, device=device))
        
        #state
        # self.origin_states = []  
        self.msg_states = [] 
        self.cls_states = []  
        # self.action_states = []  
        self.merge_msg_states = [] 
        self.expert_select_logprobs = []
        self.expert_states = []
        self.expert_select_actions = []
        self.hiddens = []
        
        self.results_dict = []


    def clear_memory(self):
       
        del self.actions[:]
        del self.coords_actions[:]
        del self.logprobs[:]
        del self.select_chief_feature_pool[:]
        del self.select_gigapath_feature_pool[:]
        
        
        del self.moe_rewards[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]
        del self.hiddens[:]
        
        # del self.action_states[:]
        # del self.origin_states[:]
        del self.msg_states[:]
        del self.cls_states[:]
        del self.merge_msg_states[:]
        del self.expert_select_logprobs[:]
        del self.expert_select_actions[:]
        del self.expert_states[:]
        del self.expert_all_probs[:]
        
        del self.results_dict[:]
    
    def clear_memory_training(self):
       
        # del self.actions[:]
        del self.coords_actions[:]
        # del self.logprobs[:]
        del self.select_chief_feature_pool[:]
        del self.select_gigapath_feature_pool[:]
        
        # del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]
        
        # del self.action_states[:]
        # del self.origin_states[:]
        del self.msg_states[:]
        del self.cls_states[:]
        # del self.merge_msg_states[:]
        
        
        del self.results_dict[:]
        

class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, device, hidden_state_dim=1024, policy_conv=False, action_std=0.1, action_size=2):
        super(ActorCritic, self).__init__()

        self.k = 30
        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim / feature_dim))
        self.gru_hidden_size = action_size

        # self.policy_rnn = nn.GRU(
        #     input_size=feature_dim,
        #     hidden_size=action_size,
        #     batch_first=True
        # )

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, action_size)
        )

        # === Policy head (Dirichlet) ===
        self.policy_head = nn.Linear(hidden_state_dim, action_size)

        # === Critic head ===
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, 1)
        )

        self.alpha_min = 1e-3
        self.alpha_max = 50

        self.action_var = torch.full((action_size,), action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
        
    def act(self, current_state, memory, restart_batch=False, training=False):
        
        
        state_ini = memory.merge_msg_states[-1].detach()  # [B, state_dim]
        # state_ini = state_ini.squeeze(1)


        # # === Shared encoder ===
        # if len(memory.hiddens) == 0:
        #     hidden_state = torch.zeros(state_ini.size(0), self.gru_hidden_size).to(state_ini.device)
        # else:
        #     hidden_state = memory.hiddens[-1]
        
        # policy_out, hidden_state = self.policy_rnn(state_ini, hidden_state)
        # memory.hiddens.append(hidden_state.detach())
        # policy_embed = policy_out[-1, :]  # [T, hidden_dim]

        policy_out = self.policy(state_ini)  # [B, hidden_dim]
        policy_embed = policy_out

        prob = torch.sigmoid(policy_embed)             # [T, 1]，範圍 0~1

        # === Bernoulli 分佈 (二元選擇) ===
        dist = torch.distributions.Bernoulli(prob)
        action = dist.sample()            # 0 或 1
        action_logprob = dist.log_prob(action)

        # === 儲存 PPO memory ===
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action
    

    def evaluate(self, state, action):
        batch_size, seq_l , state_dim = state.shape
        # state = state.squeeze(1)

        # policy_out, _ = self.policy_rnn(state)
        policy_out = self.policy(state)  # [B, hidden_dim]
        prob = torch.sigmoid(policy_out)           # [T, 1]，範圍 0~1

        # === Bernoulli 分佈 (二元選擇) ===
        dist = torch.distributions.Bernoulli(prob)
        action = action.unsqueeze(-1) if action.dim() == 2 else action
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # === Critic branch ===
        state_value = self.critic(state)

        return {
            "action_logprobs": action_logprobs.view(batch_size, seq_l),
            "state_value": state_value.view(batch_size, seq_l),
            "dist_entropy": dist_entropy.view(batch_size, seq_l),
            # "class_logits": class_logits,
        }


class PPO:
    def __init__(self, feature_dim, state_dim, hidden_state_dim, policy_conv, device,
                 action_std=0.1, lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.4, action_size=2):
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

    def update(self, memory, lambda_cls=0.25):
        """
        PPO 更新，固定包含分類 loss。
        Args:
            memory: 儲存 PPO 過程的記憶物件 (states, actions, logprobs, rewards)
            class_labels: 每個 state 對應的真實分類標籤 (Tensor: [T, B])
            lambda_cls: 分類 loss 權重 (建議 0.05~0.1)
        """
        rewards = []
        discounted_reward = 0

        # === 累積 reward ===
        for reward in reversed(memory.rewards):
            discounted_reward = reward.detach()
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards, 0)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)



        old_msg_states = torch.stack(memory.merge_msg_states, 1).detach()
        old_actions = torch.stack(memory.actions[1:], 1).detach()
        old_logprobs = torch.stack(memory.logprobs[1:], 1).detach()

        for _ in range(self.K_epochs):
            # === 評估 PPO ===
            output = self.policy.evaluate(old_msg_states, old_actions)
            logprobs = output["action_logprobs"]
            state_values = output["state_value"]
            dist_entropy = output["dist_entropy"]

            rewards = rewards.view(1, -1)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # === PPO 主損失 ===
            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.125 * self.MseLoss(state_values, rewards)
            # entropy_bonus = -0.0001 * dist_entropy


            # === 分類 loss ===
            # class_logits = output["class_logits"].view(-1, output["class_logits"].shape[-1])
            # cls_loss = F.cross_entropy(class_logits, class_labels)

            # print("class_logits:", output["class_logits"].shape)
            # print("class_labels:", class_labels.shape)

            # === 總損失 ===
            # total_loss = policy_loss + value_loss + entropy_bonus
            total_loss = policy_loss + value_loss 

            # === 反向傳遞 ===
            self.optimizer.zero_grad()

            # cls_loss.backward(retain_graph=True)
            total_loss.mean().backward()
            
            # for name, param in self.policy.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name:30s} | grad mean: {param.grad.mean():.6f} | grad std: {param.grad.std():.6f}")

            # print("cls_loss:", cls_loss.item())
            # print("classifier_head grad mean:", self.policy.classifier_head.weight.grad.abs().mean().item())
            self.optimizer.step()

        # === 更新舊 policy ===
        self.policy_old.load_state_dict(self.policy.state_dict())
    
        return -torch.min(surr1, surr2).mean().item(), self.MseLoss(state_values, rewards).mean().item(), total_loss.mean().item()

    def save(self, save_dir, name="ppo"):
        torch.save(self.policy_old.state_dict(), os.path.join(save_dir, name+".pth"))
