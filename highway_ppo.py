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

ALGORITHM = "PPO"

N_ENV = "roundabout-v0"

# env = ReacherRewardWrapper(gym.make(N_ENV),render=True)
env = gym.make(N_ENV)

env.reset()

env.configure({
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 5,
    "vehicles_count": 200,
    "duration": 60,  # [s]
    "initial_spacing": 2,
    "collision_reward": -4,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 40],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD]. subido de 30 a 40
    "simulation_frequency": 34,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 300,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
})

# model = PPO('MlpPolicy', 
#             env,
#             policy_kwargs=dict(net_arch=[256, 256]),
#             learning_rate=5e-4,
#             batch_size=32,
#             gamma=0.8,
#             verbose=0,
#             tensorboard_log="logs/highway-fast_ppo/")


# model.learn(int(2e4))
# model.save("highway_ppo/model")

# Load and test saved model
# model = DQN.load("highway_dqn/model")

model = PPO.load(f"models\\highway-fast-v0-PPO\\240000.zip", 
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            batch_size=32,
            gamma=0.8,
            verbose=0,
            tensorboard_log="logs/highway-fast_ppo/")

TIMESTEPS = 1000

obs = env.reset()

for i in range(1, 100000):
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

    # model.save(f"models/{N_ENV}-{ALGORITHM}/{TIMESTEPS*i}")
    
    action, _states = model.predict(obs, deterministic = True)
    obs, rewards, dones, info = env.step(action)
    # print(f"Save from Epoch {i} !!")
    env.render()


env.close()

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)