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
from wrapper import *
import highway_env

import warnings
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.simplefilter(action='ignore', category=FutureWarning)

ALGORITHM = "DQN_merge_2L"

env = ReacherRewardWrapper(gym.make("merge-v0"),render=True)

env.configure({
    
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
})

#model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="logs/roundabout_ppo/PPO_first_exec")

model = DQN.load("../models/merge-v0-DQN_3L/165000.zip", env =env, verbose=1, tensorboard_log="logs/DQN_merge_2L/")

TIMESTEPS = 50000


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)