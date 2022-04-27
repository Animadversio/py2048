
import gym
import torch as th
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from gym_2048 import Gym2048Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from approximator import MapCNN, Map3DCNN
#%% Train PPO environment
#%% First run, lower reward clip
venv = make_vec_env(Gym2048Env, n_envs=4, env_kwargs=dict(obstype="tensor"))
venv_norm = VecNormalize(venv, training=True, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=1000, gamma=0.99, epsilon=1e-08)
policy_kwargs = dict(
    features_extractor_class=Map3DCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO("CnnPolicy", venv_norm, policy_kwargs=policy_kwargs, verbose=1,
            tensorboard_log=r"E:\Github_Projects\2048\results\3dCNN_rew_norm")
#%%
model.learn(10000000, tb_log_name="PPO", log_interval=1, reset_num_timesteps=False)
#%%
model.save(r"E:\Github_Projects\2048\results\3dCNN_rew_norm\PPO_1\ppo_10M_rew_norm_clip1000")
venv_norm.save(r"E:\Github_Projects\2048\results\3dCNN_rew_norm\PPO_1\ppo_10M_rew_norm_clip1000_envnorm.pkl")




#%% Second run, higher reward clip
venv = make_vec_env(Gym2048Env, n_envs=4, env_kwargs=dict(obstype="tensor"))
venv_norm = VecNormalize(venv, training=True, norm_obs=False, norm_reward=True,
                         clip_obs=10.0, clip_reward=2500, gamma=0.99, epsilon=1e-08)
policy_kwargs = dict(
    features_extractor_class=Map3DCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO("CnnPolicy", venv_norm, policy_kwargs=policy_kwargs, verbose=1,
            tensorboard_log=r"E:\Github_Projects\2048\results\3dCNN_rew_norm_clip2500")
#%%
model.learn(10000000, tb_log_name="PPO", log_interval=1, reset_num_timesteps=False)
#%%
model.learn(15000000, tb_log_name="PPO", log_interval=1, reset_num_timesteps=False)
#%%
model.save(r"E:\Github_Projects\2048\results\3dCNN_rew_norm_clip2500\PPO_2\ppo_28M_rew_norm_clip2500")
venv_norm.save(r"E:\Github_Projects\2048\results\3dCNN_rew_norm_clip2500\PPO_2\ppo_28M_rew_norm_clip2500_envnorm.pkl")

#%% Evaluation
eps_rew, eps_len = evaluate_policy(model, venv, n_eval_episodes=1000, render=False, return_episode_rewards=True)
print(f"Episode reward {np.mean(eps_rew)}+-{np.std(eps_rew)}")
print(f"Episode length {np.mean(eps_len)}+-{np.std(eps_len)}")
#%% Visualization
plt.hist(eps_rew, bins=65)
plt.title(f"PPO 3D CNN 28M step reward {np.mean(eps_rew):.2f}+-{np.std(eps_rew):.2f}")
plt.xlabel("Episode Reward")
plt.savefig("PPO_eps_reward_hist.png")
plt.show()
plt.hist(eps_len, bins=65)
plt.title(f"PPO 3D CNN 28M step episode len {np.mean(eps_len):.2f}+-{np.std(eps_len):.2f}")
plt.xlabel("Episode Length")
plt.savefig("PPO_eps_len_hist.png")
plt.show()
#%%
np.savez(r"E:\Github_Projects\2048\exp_data\PPO_3dCNN_28M_scores.npz", eps_rew=eps_rew, eps_len=eps_len)
#%%


# #%% Dev zone
# model.policy(th.as_tensor(env.reset()).float().unsqueeze(0).cuda())
# #%%
# layer = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=0)
# layer2 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(2, 2, 2), stride=1, padding=0)
# # layer = nn.Conv3d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
# #%%
# obs = env.reset()
# obsth = th.tensor(obs).float().unsqueeze(0).unsqueeze(0)
# print(obsth.shape)
# tsr_out = layer(obsth)
# print(tsr_out.shape)
# tsr_out = layer2(tsr_out)
# print(tsr_out.shape)


