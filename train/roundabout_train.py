import gym
from stable_baselines3 import PPO
from wrapper import ReacherRewardWrapper
import highway_env

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ALGORITHM = "ppo"

env = ReacherRewardWrapper(gym.make("roundabout-v0"),render=True)

env.configure({
"screen_width": 640,
"screen_height": 480
})

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="../logs/roundabout_ppo/PPO_first_exec")

model = PPO.load("models/roundabout_ppo/180000.zip", env =env, verbose=0, tensorboard_log="../logs/roundabout_ppo/PPO_first_exec")

TIMESTEPS = 50000

obs = env
for i in range(20):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORITHM)

    model.save(f"../models/roundabout_{ALGORITHM}/{TIMESTEPS*i}")

env.close()