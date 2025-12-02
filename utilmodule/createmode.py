 


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from NoisetwoModel_group_rewardCalibration_MCDCP.models.DPSF import PPO,Memory
from NoisetwoModel_group_rewardCalibration_MCDCP.utilmodule.utils import make_parse
import torch


def create_model(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ppo = PPO(args.feature_dim,args.state_dim, args.policy_hidden_dim, args.policy_conv,
                        device=device,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.action_size
                        )
    memory = Memory()
    
    return ppo, memory

if __name__ == "__mian__":
    
    args = make_parse()
    create_model(args)
