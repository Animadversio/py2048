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
#%
def update_value_iter(Vnet, Voptim, target_buffer,
               update_step_freq=40000, K_epochs=100,
               writer=None, global_step=0, gamma=0.99, value_normalize=1.0):
    Vnet.train()
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
            or (runi == len(target_buffer) - 1 and len(actseq) > 10000):
            assert len(actseq) == len(rewardseq) == len(is_doneseq) == len(stateseq)
            
            T = min(update_step_freq, len(actseq))
            stateseq_tsr = torch.tensor(stateseq)
            actseq_tsr = torch.tensor(actseq)
            reward2go = 0  # torch.zeros(1).cuda()
            reward2go_vec = []
            for reward_cur, is_terminal in zip(reversed(rewardseq), reversed(is_doneseq)):
                if is_terminal:
                    reward2go = 0
                reward2go = reward_cur + gamma * reward2go
                reward2go_vec.insert(0, reward2go)

            reward2go_vec = torch.tensor(reward2go_vec).float().cuda() / value_normalize
            # reward_vec = torch.tensor(rewardseq).float().cuda()
            for iK in range(K_epochs):
                # logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())
                # logactprob_vec = logactprob_mat[torch.arange(T), actseq_tsr[0:T].long()]
                # probratio_vec = (logactprob_vec - logactprob_vec_orig).exp()
                # cumprobratio_vec = torch.cumprod(probratio_vec, dim=0)
                # advantages = reward2go_vec[0: T] - value_vec.detach()
                # entropy_bonus = -(logactprob_mat * logactprob_mat.exp()).sum(dim=1)  # .sum(dim=1)
                value_vec = Vnet(stateseq_tsr[0: T].cuda()).squeeze()

                value_err_vec = (value_vec - reward2go_vec[0: T]) ** 2

                loss = value_err_vec

                Voptim.zero_grad()
                loss.mean().backward()  # retain_graph=True            
                Voptim.step()
                if iK % 10 ==0:
                    valL2_mean = value_err_vec.mean().item()
                    # cross_entropy_mean = logactprob_vec.mean().item()
                    # entrp_bonus_mean = entropy_bonus.mean().item()
                    print(
                    f"Run{runi:d}-opt{iK:d} value L2 error {valL2_mean:.1f}")
                    if writer is not None:
                        writer.add_scalar("optim/value_L2err", valL2_mean, global_step)
                        # writer.add_scalar("optim/cross_entropy", cross_entropy_mean, global_step)
                        # writer.add_scalar("optim/act_entropy", entrp_bonus_mean, global_step)
                        global_step += 10

            actseq = []
            rewardseq = []
            stateseq = []
            is_doneseq = []
            print(f"Run{runi:d}-opt{iK:d} value L2 error {valL2_mean:.1f}")
    return global_step


def evaluate_value_iter(Vnet, target_buffer,
               update_step_freq=40000, global_step=0, gamma=0.99, value_normalize=1.0,
               writer=None, ):
    Vnet.eval()
    rew2go_all = []
    value_all = []
    value_err_all = []
    value_L1_err_all = []
    actseq = []
    rewardseq = []
    stateseq = []
    is_doneseq = []
    for runi in range(len(target_buffer)):
        actseq_ep, rewardseq_ep, stateseq_ep, _ = episodeLoader(runi, episode_buffer=target_buffer)
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
            or (runi == len(target_buffer) - 1 and len(actseq) > 10000):
            assert len(actseq) == len(rewardseq) == len(is_doneseq) == len(stateseq)
            
            T = min(update_step_freq, len(actseq))
            stateseq_tsr = torch.tensor(stateseq)
            reward2go = 0  # torch.zeros(1).cuda()
            reward2go_vec = []
            for reward_cur, is_terminal in zip(reversed(rewardseq), reversed(is_doneseq)):
                if is_terminal:
                    reward2go = 0
                reward2go = reward_cur + gamma * reward2go
                reward2go_vec.insert(0, reward2go)

            reward2go_vec = torch.tensor(reward2go_vec).float().cuda() / value_normalize
            # reward_vec = torch.tensor(rewardseq).float().cuda()
            value_vec = Vnet(stateseq_tsr[0: T].cuda()).squeeze()
            value_err_vec = (value_vec - reward2go_vec[0: T]) ** 2
            value_L1_err_vec = (value_vec - reward2go_vec[0: T]).abs()

            rew2go_all.append(reward2go_vec.detach().cpu()[0: T])
            value_all.append(value_vec.detach().cpu())
            value_err_all.append(value_err_vec.detach().cpu())
            value_L1_err_all.append(value_L1_err_vec.detach().cpu())

            actseq = actseq[T:]
            rewardseq = rewardseq[T:]
            stateseq = stateseq[T:]
            is_doneseq = is_doneseq[T:]

    rew2go_all = torch.cat(rew2go_all, dim=0)
    value_all = torch.cat(value_all, dim=0)
    value_err_all = torch.cat(value_err_all, dim=0)
    value_L1_err_all = torch.cat(value_L1_err_all, dim=0)
    valL2_mean = value_err_all.mean()
    valL1_mean = value_L1_err_all.mean()
    value_corr = torch.dot(rew2go_all, value_all) / value_all.norm() / rew2go_all.norm()
    if writer is not None:
        writer.add_scalar("eval/value_L2err", valL2_mean, global_step)
        writer.add_scalar("eval/value_L1err", valL1_mean, global_step)
        writer.add_scalar("eval/value_corr", value_corr, global_step)
    return valL2_mean, valL1_mean, value_corr, rew2go_all, value_all
#%%
expmax_buffer = {}
for triali in tqdm(range(1000)):
    episodeLoader(triali, episode_buffer=expmax_buffer, savetensor=False)
#%%
Pnet = policy_CNN()
Pnet.load_state_dict(torch.load(f"ckpt\\behav_clone\\Pnet_iter37_best.pt"))
Pnet.cuda()
#%%
Vnet = Value_CNN().cuda()
Voptim = Adam([*Vnet.parameters()], lr=5E-4)
# Vnet.load_state_dict(torch.load("ckpt\\value_iter\\Vnet_iter32_gs8350.pt"))
#%%
writer = SummaryWriter("logs\\value_iter_norm_pilot")
global_step = 8350
global_step = 0
#%%
update_step_freq = 50000
for cycle in range(0, 50):
    # update model
    global_step = update_value_iter(Vnet, Voptim, expmax_buffer,
                         update_step_freq=update_step_freq, K_epochs=50,
                         writer=writer, global_step=global_step, gamma=0.99, value_normalize=500)
    if cycle % 2 == 0:
        torch.save(Vnet.state_dict(), f"ckpt\\value_iter_norm\\Vnet_iter{cycle:d}_gs{global_step:d}.pt")

    valL2_mean, valL1_mean, value_corr, rew2go_all, value_all = evaluate_value_iter(Vnet, expmax_buffer,
               update_step_freq=update_step_freq, writer=writer,
               global_step=global_step, gamma=0.99, value_normalize=500)
    print(f"iteration {cycle:d} Eval value L2 error {valL2_mean:.1f} value L1 error {valL1_mean:.1f} value corr {value_corr:.4f}")

#%% Train it on actively collected data
B = 100
csr = 0
bufferlen = 1000
onpolicy_buffer = copy.deepcopy(expmax_buffer)
update_step_freq = 50000
for cycle in range(0, 50):
    # collect data
    Pnet.eval()
    score_list = []
    epsL_list = []
    for runi in tqdm(range(B)):
        stateseq, actseq, rewardseq, score = traj_sampler(Pnet_policy,
                                                          policyArgs={"Pnet": Pnet, "device": "cuda"}, printfreq=-1)
        episodeSaver((csr+runi) % bufferlen, actseq, rewardseq, stateseq, score,
                     episode_buffer=onpolicy_buffer, savetensor=False)
        score_list.append(score)
        epsL_list.append(len(actseq))
    print(f"iteration {cycle:d} summary (fix policy) {np.mean(score_list):.2f}+-{np.std(score_list):.2f}")

    # update model
    global_step = update_value_iter(Vnet, Voptim, onpolicy_buffer,
                         update_step_freq=update_step_freq, K_epochs=50,
                         writer=writer, global_step=global_step, gamma=0.99, value_normalize=500)
    if cycle % 2 == 0:
        torch.save(Vnet.state_dict(), f"ckpt\\value_iter_norm\\Vnet_behcln_policy_iter{cycle:d}_gs{global_step:d}.pt")

    valL2_mean, valL1_mean, value_corr, rew2go_all, value_all = evaluate_value_iter(Vnet, onpolicy_buffer,
               update_step_freq=update_step_freq, writer=writer,
               global_step=global_step, gamma=0.99, value_normalize=500)
    print(f"iteration {cycle:d} Eval value L2 error {valL2_mean:.1f} value L1 error {valL1_mean:.1f} value corr {value_corr:.4f}")

    # if writer is not None:
    #     writer.add_histogram("eval/scores", np.array(score_list), cycle)
    #     writer.add_histogram("eval/episode_len", np.array(epsL_list), cycle)
    #     writer.add_scalar("eval/scores_mean", np.array(score_list).mean(), cycle)
    #     writer.add_scalar("eval/scores_std", np.array(score_list).std(), cycle)
#%%
plt.scatter(value_all, rew2go_all, alpha=0.1)
plt.axis("equal")
plt.show()
#%%
plt.figure(figsize=[10,5])
plt.plot(value_all[:2000], alpha=0.2)
plt.plot(rew2go_all[:2000], alpha=0.2)
plt.show()
#%%
plt.figure(figsize=[10,5])
plt.plot(value_all[8000:12000], alpha=0.2)
plt.plot(rew2go_all[8000:12000], alpha=0.2)
plt.show()