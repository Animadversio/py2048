#%%
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from expCollector import traj_sampler
from expCollector import episodeLoader, episodeSaver
from approximator import policy_CNN, Value_CNN, Adam, Pnet_policy
from approximator import Map3DCNN, MapCNN
def behav_cloning(Pnet, Poptim, target_buffer,
                       update_step_freq=40000, K_epochs=100, beta=0.05,
                       writer=None, global_step=0,
                       reward_weighted=False, gamma=0.99):
    Pnet.train()
    actseq = []
    rewardseq = []
    stateseq = []
    is_doneseq = []
    perm_idx = np.random.permutation(len(target_buffer))
    for runi in range(len(target_buffer)):
        actseq_ep, rewardseq_ep, stateseq_ep, _ = episodeLoader(perm_idx[runi], episode_buffer=target_buffer)

        L = len(actseq_ep)  # min(len(actseq), T)
        is_done = np.zeros(L + 1, dtype=bool)
        is_done[-1] = True

        actseq.extend(actseq_ep)
        rewardseq.extend(rewardseq_ep)
        stateseq.extend(stateseq_ep)
        is_doneseq.extend(is_done)
        actseq.append(0)
        rewardseq.append(0)

        if len(actseq) > update_step_freq \
                or (runi == len(target_buffer) - 1 \
                    and len(actseq) > 10000):
            assert len(actseq) == len(rewardseq) == len(is_doneseq) == len(stateseq)

            T = min(update_step_freq, len(actseq))
            stateseq_tsr = th.tensor(stateseq)
            actseq_tsr = th.tensor(actseq)
            if reward_weighted:
                reward2go = 0  # torch.zeros(1).cuda()
                reward2go_vec = []
                for reward_cur, is_terminal in zip(reversed(rewardseq), reversed(is_doneseq)):
                    if is_terminal:
                        reward2go = 0
                    reward2go = reward_cur + gamma * reward2go
                    reward2go_vec.insert(0, reward2go)

            # reward2go_vec = torch.tensor(reward2go_vec).cuda()
            for iK in range(K_epochs):
                logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())

                logactprob_vec = logactprob_mat[th.arange(T), actseq_tsr[0:T].long()]

                # probratio_vec = (logactprob_vec - logactprob_vec_orig).exp()
                # cumprobratio_vec = torch.cumprod(probratio_vec, dim=0)
                # advantages = reward2go_vec[0: T] - value_vec.detach()

                entropy_bonus = - (logactprob_mat * logactprob_mat.exp()).sum(dim=1)  # .sum(dim=1)

                # value_err_vec = (value_vec - reward2go_vec[0: T]) ** 2
                # value_err_vec = (value_vec - reward2go_vec[0: T]) ** 2

                loss = - (logactprob_vec + beta * entropy_bonus)

                Poptim.zero_grad()
                loss.mean().backward()  # retain_graph=True
                Poptim.step()
                if iK % 10 == 0:
                    # valL2_mean = value_err_vec.mean().item()
                    cross_entropy_mean = logactprob_vec.mean().item()
                    entrp_bonus_mean = entropy_bonus.mean().item()
                    print(
                        f"Run{runi:d}-opt{iK:d} cross entropy {cross_entropy_mean:.1f} entropy bonus {entrp_bonus_mean:.1f}")
                    if writer is not None:
                        # writer.add_scalar("optim/value_L2err", valL2_mean, global_step)
                        writer.add_scalar("optim/cross_entropy", cross_entropy_mean, global_step)
                        writer.add_scalar("optim/act_entropy", entrp_bonus_mean, global_step)
                        global_step += 10

            actseq = []
            rewardseq = []
            stateseq = []
            is_doneseq = []

            print(
                f"Run{runi:d}-opt{iK:d} cross entropy {cross_entropy_mean:.1f} entropy bonus {entrp_bonus_mean:.1f}")
    return global_step


#%%
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gym_2048 import Gym2048Env

venv = make_vec_env(Gym2048Env, n_envs=4, env_kwargs=dict(obstype="tensor"))
venv_norm = VecNormalize(venv, training=True, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=1000, gamma=0.99, epsilon=1e-08)

policy_kwargs = dict(
    features_extractor_class=Map3DCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO("CnnPolicy", venv_norm, policy_kwargs=policy_kwargs, verbose=1,
            tensorboard_log=r"E:\Github_Projects\2048\results\3dCNN_rew_norm_BC_init")

#%%
def discount_return(reward, done, bootstrap_value, discount, return_dest=None):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc. Computes
    discounted sum of future rewards from each time-step to the end of the
    batch, including bootstrapping value.  Sum resets where `done` is 1.
    Optionally, writes to buffer `return_dest`, if provided.  Operations
    vectorized across all trailing dimensions after the first [T,]."""
    return_ = return_dest if return_dest is not None else np.zeros(
        reward.shape, dtype=reward.dtype)
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, th.Tensor) else nd
    return_[-1] = reward[-1] + discount * bootstrap_value * nd[-1]
    for t in reversed(range(len(reward) - 1)):
        return_[t] = reward[t] + return_[t + 1] * discount * nd[t]
    return return_
#%%
expmax_buffer = {}
for triali in range(1000):
    episodeLoader(triali, episode_buffer=expmax_buffer, savetensor=False, align=True)
#%% Buffer processing
def pool_buffer(target_buffer = expmax_buffer, gamma = 0.99, compact_state = True):
    actseq = []
    rewardseq = []
    returnseq = []
    stateseq = []
    is_doneseq = []
    # perm_idx = np.random.permutation(len(target_buffer))
    for runi in range(len(target_buffer)):
        actseq_ep, rewardseq_ep, stateseq_ep, _ = episodeLoader(runi, episode_buffer=target_buffer)
        if compact_state:
            stateseq_ep = np.floor(np.log2(stateseq_ep + 1)).astype("uint8")
        L = len(actseq_ep)  # min(len(actseq), T)
        is_done = np.zeros(L, dtype=bool)
        is_done[-1] = True
        return_ = discount_return(rewardseq_ep, is_done, 0, gamma)
        actseq.extend(actseq_ep)
        rewardseq.extend(rewardseq_ep)
        returnseq.extend(return_)
        stateseq.extend(stateseq_ep)
        is_doneseq.extend(is_done)
    #%
    actseq = np.array(actseq)
    rewardseq = np.array(rewardseq)
    returnseq = np.array(returnseq)
    stateseq = np.array(stateseq)
    is_doneseq = np.array(is_doneseq)
    return actseq, rewardseq, returnseq, stateseq, is_doneseq
#%%
actseq, rewardseq, returnseq, stateseq, is_doneseq = pool_buffer(expmax_buffer, gamma=0.99, compact_state=True)
#%%
# Pnet = policy_CNN().cuda()
optimizer = Adam([*model.policy.parameters()], lr=0.0005)
#%%

# def train_step(model, optimizer, batch_size, states, actions, returns, is_dones, clip_ratio=0.2):
beta = 0.01
batch_size = 256
for epoch in range(20):
    permidx = np.random.permutation(len(actseq))
    for iK in range(0, len(permidx), batch_size):
        batchid = permidx[iK:iK + batch_size]
        states = stateseq[batchid]
        states = th.tensor(stateseq[batchid])
        # states = (states + 1).log2().floor().int()
        statestsr = F.one_hot(states.long(), 17).permute(0, 3, 1, 2)
        actions = th.tensor(actseq[batchid])
        returns = th.tensor(returnseq[batchid])
        is_dones = th.tensor(is_doneseq[batchid])

        # value_err_vec = (value_vec - returns) ** 2

        act_distr = model.policy.get_distribution(statestsr.cuda())
        logactprob_vec = act_distr.log_prob(actions.cuda())
        entropy_bonus = act_distr.entropy()
        loss = - (logactprob_vec + beta * entropy_bonus).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if iK % 2560 == 0:
            # valL2_mean = value_err_vec.mean().item()
            cross_entropy_mean = logactprob_vec.mean().item()
            entrp_bonus_mean = entropy_bonus.mean().item()
            print(
                f"{epoch}-opt{iK:d} cross entropy {cross_entropy_mean:.3f} entropy bonus {entrp_bonus_mean:.3f}")
            # if writer is not None:
            #     # writer.add_scalar("optim/value_L2err", valL2_mean, global_step)
            #     writer.add_scalar("optim/cross_entropy", cross_entropy_mean, global_step)
            #     writer.add_scalar("optim/act_entropy", entrp_bonus_mean, global_step)
            #     global_step += 10
#%%
from stable_baselines3.common.evaluation import evaluate_policy
eps_rew, eps_len = evaluate_policy(model, venv, n_eval_episodes=50, render=False, return_episode_rewards=True)
print(f"Episode reward {np.mean(eps_rew)}+-{np.std(eps_rew)}")
print(f"Episode length {np.mean(eps_len)}+-{np.std(eps_len)}")
#%%
model.save(r"E:\Github_Projects\2048\results\BC_init_3DCNN_model")
#%%
model.env.clip_reward = 2500
model.learn(15000000, tb_log_name="PPO", log_interval=1, reset_num_timesteps=False)
#%%
eps_rew, eps_len = evaluate_policy(model, venv, n_eval_episodes=50, render=False, return_episode_rewards=True)
print(f"Episode reward {np.mean(eps_rew)}+-{np.std(eps_rew)}")
print(f"Episode length {np.mean(eps_len)}+-{np.std(eps_len)}")
#%%
model.save(r"E:\Github_Projects\2048\results\BC_init_3DCNN_model_PPO4M")
#%%

#%%
writer = SummaryWriter("logs\\behav_clone_pilot")
global_step = 0

