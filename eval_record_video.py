from gym_2048 import Gym2048Env, logscale2board
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def eval_record_trajectory(env, model, reward_threshold=0):
    while True:
        obs_list = []
        action_list = []
        reward_list = []
        obs = env.reset()
        done = False
        while not done:
            # qval = model(torch.from_numpy(obs).cuda())
            action, state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)

        print(f"Total reward: {sum(reward_list)} total steps: {len(reward_list)}")
        if sum(reward_list) >= reward_threshold:
            break
    return obs_list, action_list, reward_list

#%% Cherry pick a trajectory
from stable_baselines3 import DQN
env = Gym2048Env(obstype="tensor")
model = DQN.load(r"E:\Github_Projects\2048\results\DQN_3dCNN_rew_norm_5mBuffer_batch256\DQN_0\DQN_26M_rew_norm_clip2000")
#%%
obs_list, action_list, reward_list = eval_record_trajectory(env, model, reward_threshold=35000)
#%%
from tqdm import tqdm
from main_GUI import drawBoard, pg, screen
score = 0
for i in tqdm(range(len(obs_list))):
    board = logscale2board(obs_list[i])
    score += reward_list[i]
    drawBoard(board, score)
    pg.display.update()
    pg.image.save(screen, "record\\screen%04d.png" % i)

#%%
import cv2
import numpy as np
import glob
img = cv2.imread("record\\screen%04d.png" % 0)
size = img.shape[:2]
out = cv2.VideoWriter(f'game_record_DQN3_{sum(reward_list)}.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in tqdm(range(len(obs_list))):
    img = cv2.imread("record\\screen%04d.png" % i)
    out.write(img)

out.release()

