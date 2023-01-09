import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from wrapper import ReacherRewardWrapper
import highway_env

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

ALGORITHM = "PPO"

env = ReacherRewardWrapper(gym.make("roundabout-v0"),render=True)

env.configure({
"screen_width": 640,
"screen_height": 480
})

model = PPO.load("../models/roundabout_ppo/180000.zip", env =env, verbose=0, tensorboard_log="../logs/roundabout_ppo/PPO_first_exec")

TIMESTEPS = 50000

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)