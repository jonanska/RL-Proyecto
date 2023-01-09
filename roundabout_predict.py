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

env = ReacherRewardWrapper(gym.make("roundabout-v0"),render=True)

env.configure({
"screen_width": 640,
"screen_height": 480
})

#model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="logs/roundabout_ppo/PPO_first_exec")

model = PPO.load("models/roundabout_ppo/180000.zip", env =env, verbose=0, tensorboard_log="logs/roundabout_ppo/PPO_first_exec")

TIMESTEPS = 50000


# obs = env
# for i in range(20):
#     #model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORITHM)

#     #model.save(f"roundabout_{ALGORITHM}/mode/{TIMESTEPS*i}")

#     action, _states = model.predict(obs, deterministic = True)
#     obs, rewards, dones, info = env.step(action)


# env.close()

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)