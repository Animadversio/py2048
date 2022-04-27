import gym
import torch as th
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from gym_2048 import Gym2048Env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from approximator import MapCNN, Map3DCNN

#%% DQN training with 3D CNN
venv = make_vec_env(Gym2048Env, n_envs=4, env_kwargs=dict(obstype="tensor"))
venv_norm = VecNormalize(venv, training=True, norm_obs=False, norm_reward=True,
                         clip_obs=10.0, clip_reward=1500, gamma=0.99, epsilon=1e-08)
policy_kwargs = dict(
    features_extractor_class=Map3DCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = DQN("CnnPolicy", venv_norm, policy_kwargs=policy_kwargs, verbose=1, batch_size=64,
            tensorboard_log=r"E:\Github_Projects\2048\results\DQN_3dCNN_rew_norm_clip1500")
#%%
model.learn(10000000, tb_log_name="DQN", log_interval=1, reset_num_timesteps=False)
#%%
venv_norm.clip_reward = 2500
model.learn(10000000, tb_log_name="DQN", log_interval=1, reset_num_timesteps=False)
#%%
model.save(r"E:\Github_Projects\2048\results\DQN_3dCNN_rew_norm_clip1500\DQN_0\DQN_18M_rew_norm_clip2500")
venv_norm.save(r"E:\Github_Projects\2048\results\DQN_3dCNN_rew_norm_clip1500\DQN_0\DQN_18M_rew_norm_clip2500_vecnorm.pkl")
#%%
model.save(r"E:\Github_Projects\2048\results\DQN_3dCNN_rew_norm_clip1500\DQN_0\DQN_08M_rew_norm_clip1500")
#%% Evaluation
eps_rew, eps_len = evaluate_policy(model, venv, n_eval_episodes=1000, render=False, return_episode_rewards=True)
print(f"Episode reward {np.mean(eps_rew)}+-{np.std(eps_rew)}")
print(f"Episode length {np.mean(eps_len)}+-{np.std(eps_len)}")

#%% Visualization
plt.hist(eps_rew, bins=65)
plt.title(f"DQN 3D CNN 18M step reward {np.mean(eps_rew):.2f}+-{np.std(eps_rew):.2f}")
plt.xlabel("Episode Reward")
plt.savefig("DQN_eps_reward_hist.png")
plt.show()
#%
plt.hist(eps_len, bins=65)
plt.title(f"DQN 3D CNN 18M step episode len {np.mean(eps_len):.2f}+-{np.std(eps_len):.2f}")
plt.xlabel("Episode Length")
plt.savefig("DQN_eps_len_hist.png")
plt.show()

