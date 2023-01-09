import gym
import torch as th
from stable_baselines3 import PPO, DQN
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from wrapper import ReacherRewardWrapper
import highway_env

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ALGORITHM = "DQN"

env = ReacherRewardWrapper(gym.make("intersection-v0"),render=True)

env.configure({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [0, 3, 9]
            },
            "duration": 40,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "initial_vehicle_count": 3,
            "spawn_probability": 0.8,
            "screen_width": 500,
            "screen_height": 500,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": -6,
            "high_speed_reward": 0,
            "arrived_reward": 2,
            "reward_speed_range": [0.0, 1.0],
            "normalize_reward": True,
            "offroad_terminal": False
})

model = DQN('MlpPolicy', env, verbose=0, tensorboard_log="../logs/intersection_DQN/DQN_1")

TIMESTEPS = 50000

obs = env
for i in range(1, 1000000):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORITHM)

    model.save(f"../models/intersection_{ALGORITHM}/{TIMESTEPS*i}")

env.close()
