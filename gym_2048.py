import gym
from gym import spaces
from main import getSuccessor, getInitState, gameSimul
import numpy as np

class Gym2048Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, MAX_LOG2NUM=16, obstype="tensor"):
        super(Gym2048Env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        HEIGHT, WIDTH = 4, 4
        self.MAX_LOG2NUM = MAX_LOG2NUM
        self.obstype = obstype
        if obstype == "logmap":
            self.observation_space = spaces.Box(low=0, high=MAX_LOG2NUM, shape=(HEIGHT, WIDTH), dtype=np.uint8)
        elif obstype == "tensor":
            self.observation_space = spaces.Box(low=0, high=1, shape=(MAX_LOG2NUM + 1, HEIGHT, WIDTH), dtype=np.uint8)
        else:
            raise ValueError("obstype must be map or tensor")
        self.board = getInitState()

    def step(self, action):
        board, reward, done = getSuccessor(self.board, action=action, show=False, clone=False)
        info = {}
        observation = self.board2logscale(self.board)  # np.floor(np.log2(self.board + 1)).astype("uint8")
        if self.obstype == "tensor":
            observation = np.eye(self.MAX_LOG2NUM + 1, dtype=np.uint8)[observation]
            observation = observation.transpose((2, 0, 1))
        return observation, reward, done, info

    def reset(self):
        self.board = getInitState()
        observation = self.board2logscale(self.board)
        if self.obstype == "tensor":
            observation = np.eye(self.MAX_LOG2NUM + 1, dtype=np.uint8)[observation]
            observation = observation.transpose((2, 0, 1))
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        pass

    def board2logscale(self, board):
        return np.floor(np.log2(board + 1)).astype("uint8")


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    for obstype in ["tensor", "logmap"]:
        env = Gym2048Env(obstype=obstype)
        # env = Gym2048Env(obstype="logmap")
        check_env(env)