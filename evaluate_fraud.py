import os
import sys
import torch
import numpy as np

# Add engine and envs to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engine'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
sys.path.append(os.path.dirname(__file__))

from configs import cfg
from config_files.configs_amlsim import cfg as amlsim_cfg
from envs.amlsim_env import AMLSimEnv
from model.transdreamer import TransDreamer

def unnormalize(obs_tensor):
    """Reverses the Atari-based normalization: (x / 255.0) - 0.5"""
    return (obs_tensor + 0.5) * 255.0

def evaluate(checkpoint_path, history_steps=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    cfg.merge_from_other_cfg(amlsim_cfg)
    cfg.resume = True
    cfg.resume_ckpt = checkpoint_path
    
    # Force amlsim configuration overrides
    cfg.env.name = 'amlsim'
    cfg.env.action_size = 2
    cfg.arch.world_model.input_type = 'image'
    
    # Force evaluation mode settings
    cfg.train.batch_size = 1
    cfg.env.max_steps = 1000

    csv_path = cfg.env.amlsim_csv_path if hasattr(cfg.env, 'amlsim_csv_path') else '../aml_sim/outputs/10K/tx_log.csv'
    env = AMLSimEnv(csv_path, max_steps=cfg.env.max_steps, seed=42)
    
    model = TransDreamer(cfg).to(device)
    
    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    import random
    
    for seq_num in range(5):
        # Reset environment with a random seed to get different accounts
        seed = 42 + seq_num * 10
        env = AMLSimEnv(csv_path, max_steps=cfg.env.max_steps, seed=seed)
        obs = env.reset()
        
        # Initialize latent states
        action_list = torch.zeros(1, 1, cfg.env.action_size).float().to(device)
        action_list[:, 0, 0] = 1. # Default action
        step = 0
        temp = 1.0
        state = None
        input_type = cfg.arch.world_model.input_type
        
        for step in range(history_steps):
            next_obs, reward, done, _ = env.step(0)
            prev_image = torch.tensor(obs[input_type]).unsqueeze(0).to(device)
            next_image = torch.tensor(next_obs[input_type]).unsqueeze(0).to(device)
            action_list, state = model.policy(prev_image, next_image, action_list, step, temp, state, training=False, context_len=cfg.train.batch_length)
            obs = next_obs
            if done:
                break

        # Predict the NEXT state using the world model's inferred prior
        post_stoch = state['stoch'][:, -1:]
        last_action = action_list[:, -1:]
        
        with torch.no_grad():
            pred_prior = model.world_model.dynamic.infer_prior_stoch(post_stoch, temp, last_action)
            rnn_feature = model.world_model.dynamic.get_feature(pred_prior, layer=model.world_model.reward_layer)
            predicted_obs_dist = model.world_model.img_dec(rnn_feature)
            predicted_obs = predicted_obs_dist.mean.squeeze().cpu().numpy()
            predicted_reward_dist = model.world_model.reward(rnn_feature)
            predicted_reward = predicted_reward_dist.mean.item()

        pred_amt = predicted_obs[2]
        pred_old_bal = predicted_obs[3]
        pred_new_bal = predicted_obs[4]
        
        print(f"\n--- Sequence {seq_num+1} Prediction ---")
        print(f"  > Amount:           ${pred_amt:,.2f}")
        print(f"  > Old Balance:      ${pred_old_bal:,.2f}")
        print(f"  > New Balance:      ${pred_new_bal:,.2f}")
        print(f"  > Difference (Old+Amt - New): ${((pred_old_bal + pred_amt) - pred_new_bal):,.2f}")

if __name__ == "__main__":
    # Expect the checkpoint in the same directory as this script by default
    default_ckpt = os.path.join(os.path.dirname(__file__), "model_000030001.pth")
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else default_ckpt
    
    evaluate(checkpoint_path, history_steps=10)
