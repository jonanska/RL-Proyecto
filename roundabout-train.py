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

os.environ['KMP_DUPLICATE_LIB_OK']='True'


warnings.simplefilter(action='ignore', category=FutureWarning)

ALGORITHM = "roundabout"

N_ENV = "roundabout-v0"

# env = ReacherRewardWrapper(gym.make(N_ENV),render=True)

env = gym.make(N_ENV)

env.reset()

env.configure({
    "duration": 9
})

# model = DQN('MlpPolicy', 
#             env,
#             policy_kwargs=dict(net_arch=[256, 256, 128, 64]),
#             learning_rate=5e-4,
#             buffer_size=20000,
#             learning_starts=200,
#             batch_size=32,
#             gamma=0.8,
#             train_freq=1,
#             gradient_steps=1,
#             target_update_interval=50,
#             verbose=1,
#             tensorboard_log="logs-mod/highway-fast_dqn-mod/")


# model.learn(int(2e4))
# model.save("highway_ppo/model")

# Load and test saved model
# model = DQN.load("highway_dqn/model")

model = PPO.load(f"models\\roundabout_ppo\\180000.zip", 
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            batch_size=32,
            gamma=0.8,
            verbose=1,
            tensorboard_log="logs-round-mod/roundabout_final_ppo/")

TIMESTEPS = 2000


for i in range(1, 10000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

    model.save(f"models_round_final-mod/{N_ENV}-{ALGORITHM}/{TIMESTEPS*i}")
    
     # action, _states = model.predict(obs, deterministic = True)
     # obs, rewards, dones, info = env.step(action)

    print(f"Save from Epoch {i} !!")


env.close()

# # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# # print(mean_reward)
# # print(std_reward)