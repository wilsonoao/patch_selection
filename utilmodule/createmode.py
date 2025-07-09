 


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from PAMIL_two_round.models.DPSF import PPO,Memory
from PAMIL_two_round.models.MoE_agent import MoE_agent
from PAMIL_two_round.utilmodule.utils import make_parse
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
    MoE = MoE_agent(args.feature_dim,args.expert_state_dim, args.policy_hidden_dim, args.policy_conv,
                        device=device,
                        action_size=args.expert_action_size)

    memory = Memory()
    
    return None,ppo,None,memory ,None, MoE

if __name__ == "__mian__":
    
    args = make_parse()
    create_model(args)
