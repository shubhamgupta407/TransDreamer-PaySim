import gym
import numpy as np
import pandas as pd
from gym import spaces

class AMLSimEnv(gym.Env):
    def __init__(self, csv_path, max_steps=1000, seed=0):
        super(AMLSimEnv, self).__init__()
        self.csv_path = csv_path
        self.max_steps = max_steps
        self.seed(seed)
        
        # Load dataset
        print(f"Loading {csv_path} into AMLSimEnv...")
        df = pd.read_csv(csv_path)
        
        # Map type to int
        type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'CASH_IN': 3, 'DEBIT': 4}
        # In case there are missing types not in mapping, we provide a default or handle it gracefully
        df['type'] = df['type'].map(type_mapping).fillna(5).astype(int)
        
        # We need trajectories grouped by nameOrig
        print("Grouping AMLSim dataset by nameOrig (account trajectories)...")
        self.grouped = df.groupby('nameOrig')
        self.account_ids = list(self.grouped.groups.keys())
        
        # Observation space: 7 features (state, named image for TransDreamer pipeline)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })
        
        # Action space: 0 (Allow) or 1 (Block)
        self.action_space = spaces.Discrete(2)
        
        self.current_account = None
        self.current_trajectory = None
        self.current_step_idx = 0
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        # Pick a random account
        self.current_account = self.np_random.choice(self.account_ids)
        self.current_trajectory = self.grouped.get_group(self.current_account).reset_index(drop=True)
        self.current_step_idx = 0
        
        obs = self._get_obs()
        return obs
        
    def _get_obs(self):
        row = self.current_trajectory.iloc[self.current_step_idx]
        features = np.array([
            row['step'],
            row['type'],
            row['amount'],
            row['oldbalanceOrig'],
            row['newbalanceOrig'],
            row['oldbalanceDest'],
            row['newbalanceDest']
        ], dtype=np.float32)
        return {'image': features}
        
    def step(self, action):
        row = self.current_trajectory.iloc[self.current_step_idx]
        is_sar = int(row['isSAR'])
        
        # Reward logic: +1 if action == isSAR else -1
        reward = 1.0 if action == is_sar else -1.0
        
        self.current_step_idx += 1
        done = self.current_step_idx >= len(self.current_trajectory) or self.current_step_idx >= self.max_steps
        
        if not done:
            obs = self._get_obs()
        else:
            self.current_step_idx -= 1
            obs = self._get_obs()
            self.current_step_idx += 1
            
        info = {}
        return obs, float(reward), done, info
