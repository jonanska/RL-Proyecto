import gym

class ReacherRewardWrapper(gym.Wrapper):
    def __init__(self, env, render=False):
        super().__init__(env)
        self.env=env
        self.render=render
    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        if self.render==True:
            self.env.render()
        return obs, reward, terminated,  info