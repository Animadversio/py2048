#%% Behavior cloning
"""In fact behavior cloning is very effective in this task
the mean expctimax score is 3400; the cloning agent score is 2900
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD
from collections import OrderedDict
import copy
from main import getInitState, getSuccessor, getSuccessors, gameSimul, actions, sample
import matplotlib.pylab as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from expCollector import traj_sampler
from expCollector import episodeLoader, episodeSaver
from approximator import policy_CNN, Value_CNN, Adam, Pnet_policy


def update_behav_clone(Pnet, Poptim, target_buffer,
               update_step_freq=40000, K_epochs=100, beta=0.05,
               writer=None, global_step=0, 
               reward_weighted=False):
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
            stateseq_tsr = torch.tensor(stateseq)
            actseq_tsr = torch.tensor(actseq)
            # reward2go = 0  # torch.zeros(1).cuda()
            # reward2go_vec = []
            # for reward_cur, is_terminal in zip(reversed(rewardseq), reversed(is_doneseq)):
            #     if is_terminal:
            #         reward2go = 0
            #     reward2go = reward_cur + gamma * reward2go
            #     reward2go_vec.insert(0, reward2go)

            # reward2go_vec = torch.tensor(reward2go_vec).cuda()

            for iK in range(K_epochs):
                logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())

                logactprob_vec = logactprob_mat[torch.arange(T), actseq_tsr[0:T].long()]
                
                # probratio_vec = (logactprob_vec - logactprob_vec_orig).exp()
                # cumprobratio_vec = torch.cumprod(probratio_vec, dim=0)
                # advantages = reward2go_vec[0: T] - value_vec.detach()

                entropy_bonus = -(logactprob_mat * logactprob_mat.exp()).sum(dim=1)  # .sum(dim=1)

                # value_err_vec = (value_vec - reward2go_vec[0: T]) ** 2
                # value_err_vec = (value_vec - reward2go_vec[0: T]) ** 2

                loss = - (logactprob_vec + beta * entropy_bonus)

                Poptim.zero_grad()
                loss.mean().backward()  # retain_graph=True            
                Poptim.step()
                if iK % 10 ==0:
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


def Pnet_policy(board, Pnet, device="cpu"):
    with torch.no_grad():
        prob = Pnet(torch.tensor(board).unsqueeze(0).to(device))
        choices = torch.multinomial(prob.exp(), num_samples=1) # output is B-by-1
    return choices, 0
#%%
expmax_buffer = {}
for triali in range(1000):
    episodeLoader(triali, episode_buffer=expmax_buffer, savetensor=False)

#%%
B = 150
beta = 0.5
epsilon = 0.2
gamma = 0.99

Pnet = policy_CNN().cuda()
Poptim = Adam([*Pnet.parameters()], lr=0.0005)
#%%
writer = SummaryWriter("logs\\behav_clone_pilot")
global_step = 0
#%%
update_step_freq = 40000
for cycle in range(22, 40):
    # update model
    global_step = update_behav_clone(Pnet, Poptim, expmax_buffer,
                                     update_step_freq=update_step_freq, K_epochs=200,
                                     writer=writer, global_step=global_step, beta=0.025)
    if cycle % 1 == 0:
        torch.save(Pnet.state_dict(), f"ckpt\\behav_clone\\Pnet_iter{cycle:d}_gs{global_step:d}.pt")

    # collect data
    Pnet.eval()
    score_list = []
    epsL_list = []
    onpolicy_buffer = {}
    for runi in tqdm(range(B)):
        stateseq, actseq, rewardseq, score = traj_sampler(Pnet_policy,
                                          policyArgs={"Pnet": Pnet, "device": "cuda"}, printfreq=-1)
        episodeSaver(runi, actseq, rewardseq, stateseq, score,
                     episode_buffer=onpolicy_buffer, savetensor=False)
        score_list.append(score)
        epsL_list.append(len(actseq))
    print(f"iteration {cycle:d} summary {np.mean(score_list):.2f}+-{np.std(score_list):.2f}")

    if writer is not None:
        writer.add_histogram("eval/scores", np.array(score_list), global_step)
        writer.add_histogram("eval/episode_len", np.array(epsL_list), global_step)
        writer.add_scalar("eval/scores_mean", np.array(score_list).mean(), global_step)
        writer.add_scalar("eval/scores_std", np.array(score_list).std(), global_step)

# iteration 20 summary 2914.67+-1398.87