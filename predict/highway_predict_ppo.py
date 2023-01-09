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
from wrapper import *
import highway_env

import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

ALGORITHM = "PPO"

env = ReacherRewardWrapper(gym.make("highway-fast-v0"),render=True)


model = PPO.load("../models/highway-fast-v0-PPO/240000", env =env, verbose=1, tensorboard_log="../logs/highway-fast-v0_ppo")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print('Mean reward:' + mean_reward)
print('Standard deviation of rewards: ' + std_reward)