# PPO.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD
from collections import OrderedDict
import copy
from main import getInitState, getSuccessor, getSuccessors, gameSimul, actions, sample
import matplotlib.pylab as plt

episode_buffer = {}
def episodeLoader(triali, episode_buffer=episode_buffer, savetensor=False):
    if triali not in episode_buffer:
        data = np.load("exp_data\\traj%03d.npz"%triali)
        actseq = data['actseq']  # (T, )
        rewardseq = data['rewardseq']  # (T, )
        stateseq = data['stateseq']  # (T+1, 4, 4)
        score_tot = data['score']
        if savetensor:
            episode_buffer[triali] = torch.tensor(actseq), torch.tensor(rewardseq), \
                                     torch.tensor(stateseq), score_tot
            return torch.tensor(actseq), torch.tensor(rewardseq), \
                                     torch.tensor(stateseq), score_tot
        else:
            episode_buffer[triali] = actseq, rewardseq, stateseq, score_tot
            return actseq, rewardseq, stateseq, score_tot
    else:
        actseq, rewardseq, stateseq, score_tot = episode_buffer[triali]
        return actseq, rewardseq, stateseq, score_tot


def episodeSaver(triali, actseq, rewardseq, stateseq, score_tot, episode_buffer=episode_buffer, savetensor=False):
    if savetensor:
        episode_buffer[triali] = torch.tensor(actseq), torch.tensor(rewardseq), \
                                 torch.tensor(stateseq), score_tot
    else:
        episode_buffer[triali] = actseq, rewardseq, stateseq, score_tot


MAX_LOG2NUM = 16
class policy_CNN(nn.Module):

    def __init__(self, max_log2num=MAX_LOG2NUM):
        super().__init__()
        self.max_log2num = max_log2num
        self.model = nn.Sequential(OrderedDict(
            [("Conv1", nn.Conv2d(self.max_log2num, 128, 2, stride=1, padding=0, dilation=1,)),
             ("ReLU1", nn.LeakyReLU(negative_slope=0.05)),
             ("BN1", nn.BatchNorm2d(128)),
             ("Conv2", nn.Conv2d(128, 128, 2, stride=1, padding=0, dilation=1,)),
             ("ReLU2", nn.LeakyReLU(negative_slope=0.05)),
             ("BN2", nn.BatchNorm2d(128)),
             # ("Conv3", nn.Conv2d(128, 128, 2, stride=1, padding=0, dilation=1,)),
             # ("ReLU3", nn.LeakyReLU(negative_slope=0.05)),
             ("flatten", nn.Flatten(start_dim=1, end_dim=-1)),
             ("Lin4", nn.Linear(128*4, 64)),
             ("ReLU4", nn.LeakyReLU(negative_slope=0.05)),
             ("BN4", nn.BatchNorm1d(64)),
             ("Lin5", nn.Linear(64, 4)),
             ("logsoftmax", nn.LogSoftmax())]))

    def preprocess(self, stateseq):
        logstate = (1 + stateseq).float().log2().floor()
        logstatetsr = F.one_hot(logstate.long(), self.max_log2num).permute([0,3,1,2])
        return logstatetsr.float()

    def forward(self, stateseq):
        logstatetsr = self.preprocess(stateseq)
        return self.model(logstatetsr)

    # def Q_loss_TD_seq(self, stateseq, actseq, rewardseq, batch=100, discount=0.9, 
    #                 log2_loss=True, device="cuda"):
    #     # Getting target Q value with current model.
    #     if batch is None: batch = len(rewardseq)
    #     else: batch = min(batch, len(rewardseq))
    #     Qtab = self.forward(stateseq[:batch+1,:,:].to(device))
    #     QActSel = Qtab[torch.arange(batch, dtype=torch.int64), actseq[:batch].long()]
    #     QnextMax, QMaxAct = Qtab[1:, :].max(dim=1)
    #     curRew = rewardseq[:batch, ].float().to(device)
    #     # loss = (discount * QnextMax + curRew - QActSel).pow(2).mean()
    #     if log2_loss:
    #         loss = F.smooth_l1_loss((discount * QnextMax + curRew + 1).log2(), (QActSel + 1).log2())
    #     else:
    #         loss = F.smooth_l1_loss(discount * QnextMax + curRew, QActSel)
    #     return loss


class Value_CNN(nn.Module):
    """Value baseline network"""
    def __init__(self, max_log2num=MAX_LOG2NUM):
        super().__init__()
        self.max_log2num = max_log2num
        self.model = nn.Sequential(OrderedDict(
            [("Conv1", nn.Conv2d(self.max_log2num, 128, 2, stride=1, padding=0, dilation=1,)),
             ("ReLU1", nn.LeakyReLU(negative_slope=0.05)),
             ("BN1", nn.BatchNorm2d(128)),
             ("Conv2", nn.Conv2d(128, 128, 2, stride=1, padding=0, dilation=1,)),
             ("ReLU2", nn.LeakyReLU(negative_slope=0.05)),
             ("BN2", nn.BatchNorm2d(128)),
             # ("Conv3", nn.Conv2d(128, 128, 2, stride=1, padding=0, dilation=1,)),
             # ("ReLU3", nn.LeakyReLU(negative_slope=0.05)),
             ("flatten", nn.Flatten(start_dim=1, end_dim=-1)),
             ("Lin4", nn.Linear(128*4, 64)),
             ("ReLU4", nn.LeakyReLU(negative_slope=0.05)),
             ("BN4", nn.BatchNorm1d(64)),
             ("Lin5", nn.Linear(64, 1)),
             ]))

    def preprocess(self, stateseq):
        logstate = (1 + stateseq).float().log2().floor()
        logstatetsr = F.one_hot(logstate.long(), self.max_log2num).permute([0,3,1,2])
        return logstatetsr.float()

    def forward(self, stateseq, exp=True):
        logstatetsr = self.preprocess(stateseq)
        return self.model(logstatetsr).exp() if exp else self.model(logstatetsr)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)


#%  Policy based on Q networks
def Pnet_policy(board, Pnet, device="cpu"):
    with torch.no_grad():
        prob = Pnet(torch.tensor(board).unsqueeze(0).to(device))
        choices = torch.multinomial(prob.exp(), num_samples=1) # output is B-by-1
    return choices, 0

#%%
# triali = 10
# data = np.load("exp_data\\traj%03d.npz"%triali)
# stateseq = data["stateseq"]
# actseq = data["actseq"]
# rewardseq = data["rewardseq"]
# score = data["score"]
#%%
# buffer = {}
# def load_traj_store(triali):
#     data = np.load("exp_data\\traj%03d.npz" % triali)
#     stateseq = data["stateseq"]
#     actseq = data["actseq"]
#     rewardseq = data["rewardseq"]
#     score = data["score"]
#     return torch.tensor(stateseq), torch.tensor(actseq), torch.tensor(rewardseq), torch.tensor(score)
#%% #######################################
# """ Policy gradient """
# Pnet = policy_CNN()
# Pnet.cuda()
# Poptim = Adam([*Pnet.parameters()], lr=0.05)
# #%%
# gamma = 0.9
# T = 100
# Poptim.zero_grad()
# stateseq_tsr = torch.tensor(stateseq).cuda()
# surrogate = torch.zeros(1).cuda()
# reward2go = torch.zeros(1).cuda()
# logactprob_mat = Pnet(stateseq_tsr[0: T])
# for t in range(T-1, -1, -1):
#     state_cur = stateseq_tsr[t:t + 1]
#     state_nxt = stateseq_tsr[t + 1:t + 2]
#     reward2go = rewardseq[t] + gamma * reward2go
#     logactprob = Pnet(state_cur)
#     surrogate += logactprob[0, actseq[t]] * reward2go
#
# surrogate.backward()  # retain_graph=True
#
# #%% Simplified Policy Gradient
# B = 50
# T = 100
# gamma = 0.9
# Poptim.zero_grad()
# for triali in range(B):
#     actseq, rewardseq, stateseq, _ = episodeLoader(triali)
#     surrogate = torch.zeros(1).cuda()
#     reward2go = torch.zeros(1).cuda()
#     logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())
#     for t in range(T-1, -1, -1):
#         reward2go = rewardseq[t] + gamma * reward2go
#         surrogate += logactprob_mat[t, actseq[t]] * reward2go
#
#     surrogate.backward()  # retain_graph=True
#
# Poptim.step()
#
#
# #%% Policy Gradient with fixed baseline
# Poptim.zero_grad()
# B = 50
# baseline = 0
# for triali in range(B):
#     actseq, rewardseq, stateseq, _ = episodeLoader(triali)
#     surrogate = torch.zeros(1).cuda()
#     reward2go = torch.zeros(1).cuda()
#     logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())
#     for t in range(T-1, -1, -1):
#         reward2go = rewardseq[t] - baseline + gamma * reward2go
#         surrogate += logactprob_mat[t, actseq[t]] * reward2go
#
#     surrogate.backward()  # retain_graph=True
#
# Poptim.step()
#
# #%% Policy Gradient with Value function (Actor Critic)
# Pnet = policy_CNN()
# Vnet = Value_CNN()
# Pnet.cuda()
# Vnet.cuda()
# Poptim = Adam([*Pnet.parameters()], lr=0.05)
# Voptim = Adam([*Vnet.parameters()], lr=0.05)
#
# # update value and policy using current dataset
# Poptim.zero_grad()
# Voptim.zero_grad()
# B = 50
# baseline = 0
# value_err = torch.zeros(1).cuda()
# surrogate = torch.zeros(1).cuda()
# for triali in range(B):
#     actseq, rewardseq, stateseq, _ = episodeLoader(triali)
#     reward2go = torch.zeros(1).cuda()
#     logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())
#     value_vec = Vnet(stateseq_tsr[0: T+1].cuda())
#     for t in range(T-1, -1, -1):
#         # reward2go = rewardseq[t] - baseline + gamma * reward2go
#         advantage = rewardseq[t] + gamma * value_vec[t + 1] - value_vec[t]  # advantage
#         surrogate += logactprob_mat[t, actseq[t]] * advantage
#         value_err += advantage ** 2
#
# loss = 0.1 * value_err - surrogate  # maximize surrogate, minimize value_err
# loss.backward()  # retain_graph=True
# Poptim.step()
#
#
# #%% Actor Critic with importance sampling
# # https://julien-vitay.net/deeprl/ImportanceSampling.html
# Pnet = policy_CNN().cuda()
# Vnet = Value_CNN().cuda()
#
# Poptim = Adam([*Pnet.parameters()], lr=0.01)
# Voptim = Adam([*Vnet.parameters()], lr=0.01)
#
# T = 200
# B = 50
# beta = 0.1
# epsilon = 0.2
# gamma = 0.9

# update value and policy using current dataset
#%%
def update_A2C_IS(Pnet, Vnet, Poptim, Voptim, onpolicy_buffer):
    Pnet.train()
    Vnet.train()
    Pnet_orig = copy.deepcopy(Pnet)
    Pnet_orig.requires_grad_(False)
    for epi in range(update_epoch):
        Poptim.zero_grad()
        Voptim.zero_grad()
        for runi in range(B):
            surrogate = torch.zeros(1).cuda()
            value_err = torch.zeros(1).cuda()
            entropy_bonus = torch.zeros(1).cuda()

            actseq, rewardseq, stateseq_tsr, _ = episodeLoader(runi, episode_buffer=onpolicy_buffer)

            reward2go = torch.zeros(1).cuda()
            L = min(len(actseq), T)
            logactprob_mat = Pnet(stateseq_tsr[0: L].cuda())
            value_vec = Vnet(stateseq_tsr[0: L + 1].cuda())
            with torch.no_grad():
                logactprob_orig = Pnet_orig(stateseq_tsr[0: L].cuda())

            logactprob_vec = logactprob_mat[torch.arange(L), actseq[0:L].long()]
            logactprob_vec_orig = logactprob_orig[torch.arange(L), actseq[0:L].long()]
            probratio_vec = (logactprob_vec - logactprob_vec_orig).exp()
            cumprobratio_vec = torch.cumprod(probratio_vec, dim=0)

            for t in range(L - 1, -1, -1):
                # reward2go = rewardseq[t] - baseline + gamma * reward2go
                # ratio = (logactprob_mat[t, actseq[t]] - logactprob_orig[t, actseq[t]]).exp()
                advantage = rewardseq[t] + gamma * value_vec[t + 1] - value_vec[t]  # advantage
                surrogate += logactprob_mat[t, actseq[t]] * cumprobratio_vec[t] * advantage
                value_err += advantage ** 2

            entropy_bonus += -(logactprob_mat * logactprob_mat.exp()).sum()  # .sum(dim=1)

            loss = 0.5 * value_err - (surrogate + beta * entropy_bonus)
            loss.backward()  # retain_graph=True
            if runi % gradstep_freq == 0:
                Poptim.step()
                Voptim.step()
                Poptim.zero_grad()
                Voptim.zero_grad()

            if runi % 50 == 0:
                print(
                    f"Epoch {epi:d}-run{runi:d} Loss decomp Valuee L2 {value_err.item():.1f} surrogate {surrogate.item():.1f} entropy_err {entropy_bonus.item():.1f}")
        print(
            f"Epoch {epi:d} Loss decomp Valuee L2 {value_err.item():.1f} surrogate {surrogate.item():.1f} entropy_err {entropy_bonus.item():.1f}")

#%%
def update_PPO(Pnet, Vnet, Poptim, Voptim, onpolicy_buffer,
               K_epochs=40, update_step_freq=3000, writer=None, global_step=0):
    Pnet.train()
    Vnet.train()
    Pnet_orig = copy.deepcopy(Pnet)
    Pnet_orig.requires_grad_(False)
    actseq = []
    rewardseq = []
    stateseq = []
    is_doneseq = []
    for runi in range(len(onpolicy_buffer)):
        actseq_ep, rewardseq_ep, stateseq_ep, _ = episodeLoader(runi, episode_buffer=onpolicy_buffer)

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
            or (runi == len(onpolicy_buffer)-1 and len(actseq) > 10000):
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

            reward2go_vec = torch.tensor(reward2go_vec).cuda()

            with torch.no_grad():
                logactprob_orig = Pnet_orig(stateseq_tsr[0: T].cuda())

            logactprob_vec_orig = logactprob_orig[torch.arange(T), actseq_tsr[0:T].long()]

            for iK in range(K_epochs):
                logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())
                value_vec = Vnet(stateseq_tsr[0: T].cuda(), exp=True).squeeze()
                logactprob_vec = logactprob_mat[torch.arange(T), actseq_tsr[0:T].long()]
                probratio_vec = (logactprob_vec - logactprob_vec_orig).exp()
                # cumprobratio_vec = torch.cumprod(probratio_vec, dim=0)
                advantages = reward2go_vec[0: T] - value_vec.detach()

                surrogate1 = probratio_vec * advantages
                surrogate2 = torch.clamp(probratio_vec, 1 - epsilon, 1 + epsilon) * advantages
                surrogate = torch.min(surrogate1, surrogate2)

                entropy_bonus = -(logactprob_mat * logactprob_mat.exp()).sum(dim=1)  # .sum(dim=1)

                # value_err_vec = (value_vec - reward2go_vec[0: T]) ** 2
                value_err_vec = (value_vec - reward2go_vec[0: T]) ** 2

                loss = 0.5 * value_err_vec - (surrogate + beta * entropy_bonus)

                Poptim.zero_grad()
                Voptim.zero_grad()                
                loss.mean().backward()  # retain_graph=True            
                Poptim.step()
                Voptim.step()
                if iK % 10 ==0:
                    valL2_mean = value_err_vec.mean().item()
                    surr_mean = surrogate.mean().item()
                    entrp_bonus_mean = entropy_bonus.mean().item()
                    print(
                    f"Run{runi:d}-opt{iK:d} Valuee L2 {valL2_mean:.1f} surrogate {surr_mean:.1f} entropy_err {entrp_bonus_mean:.1f}")
                    if writer is not None:
                        writer.add_scalar("optim/value_L2err", valL2_mean, global_step)
                        writer.add_scalar("optim/surrogate", surr_mean, global_step)
                        writer.add_scalar("optim/act_entropy", entrp_bonus_mean, global_step)
                        global_step += 10

            actseq = []
            rewardseq = []
            stateseq = []
            is_doneseq = []

            print(
                f"Run{runi:d} Loss decomp Valuee L2 {value_err_vec.mean().item():.1f} surrogate {surrogate.mean().item():.1f} entropy_err {entropy_bonus.mean().item():.1f}")
    return global_step
        

#%% Training cycle
from expCollector import traj_sampler
B = 300
beta = 0.5
epsilon = 0.2
gamma = 0.99
# update_epoch = 10
# gradstep_freq = 20
Pnet = policy_CNN().cuda()
Vnet = Value_CNN().cuda()
Poptim = Adam([*Pnet.parameters()], lr=0.00015)
Voptim = Adam([*Vnet.parameters()], lr=0.00005)
#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs\\PPO_pilot_new_logval")
global_step = 0
global_step = 9300
global_step = 12600
#%%
update_step_freq = 40000
for cycle in range(55, 65):
    # collect data
    Pnet.eval()
    onpolicy_buffer = {}
    score_list = []
    epsL_list = []
    for runi in tqdm(range(B)):
        stateseq, actseq, rewardseq, score = traj_sampler(Pnet_policy, 
                                    policyArgs={"Pnet": Pnet, "device": "cuda"},
                                    printfreq=-1)
        episodeSaver(runi, actseq, rewardseq, stateseq, score,
                     episode_buffer=onpolicy_buffer, savetensor=False)
        score_list.append(score)
        epsL_list.append(len(actseq))
        if sum(epsL_list) > update_step_freq:
            break

    if writer is not None:
        writer.add_histogram("eval/scores", np.array(score_list), global_step)
        writer.add_histogram("eval/episode_len", np.array(epsL_list), global_step)
        writer.add_scalar("eval/scores_mean", np.array(score_list).mean(), global_step)
        writer.add_scalar("eval/scores_std", np.array(score_list).std(), global_step)
    
    print(f"iteration {cycle:d} summary {np.mean(score_list):.2f}+-{np.std(score_list):.2f}")
    # update model
    # update_A2C_IS(Pnet, Vnet, Poptim, Voptim, onpolicy_buffer)
    global_step = update_PPO(Pnet, Vnet, Poptim, Voptim, onpolicy_buffer,
               update_step_freq=update_step_freq, K_epochs=200, writer=writer, global_step=global_step)
    if cycle % 5 == 0:
        torch.save(Pnet.state_dict(), f"ckpt\\Pnet_iter{cycle:d}_gs{global_step:d}.pt")
        torch.save(Vnet.state_dict(), f"ckpt\\Vnet_iter{cycle:d}_gs{global_step:d}.pt")

#%% load dict
Pnet.load_state_dict(torch.load(f"ckpt\\Pnet_iter{30:d}_gs{9300:d}.pt"))
Vnet.load_state_dict(torch.load(f"ckpt\\Vnet_iter{30:d}_gs{9300:d}.pt"))
global_step = 9300
#%% load dict
cycle, global_step = 50, 14400
cycle, global_step = 45, 12600
Pnet.load_state_dict(torch.load(f"ckpt\\Pnet_iter{cycle:d}_gs{global_step:d}.pt"))
Vnet.load_state_dict(torch.load(f"ckpt\\Vnet_iter{cycle:d}_gs{global_step:d}.pt"))
#%%
for param in Pnet.parameters():
    print(param.norm(),param.grad.norm())

for param in Vnet.parameters():
    print(param.norm(),param.grad.norm())
#%%
# iteration 0 summary 773.05+-383.55
# iteration 1 summary 751.86+-387.57
# actseq, score = gameSimul(Pnet_policy, {"Pnet": Pnet, "device": "cuda"}, printfreq=0)
# iteration 39 summary 1143.06+-478.67
# iteration 40 summary 1156+-
#%%
# Policy agent 

# # Pnet_orig = policy_CNN() # = copy.deepcopy(Pnet)
# # Pnet_orig.load_state_dict(Pnet.state_dict())
# Pnet_orig = copy.deepcopy(Pnet)
# Pnet_orig.requires_grad_(False)

# for epi in range(10):
#     Poptim.zero_grad()
#     Voptim.zero_grad()
#     surrogate = torch.zeros(1).cuda()
#     value_err = torch.zeros(1).cuda()
#     entropy_bonus = torch.zeros(1).cuda()
#     for triali in range(B):
#         actseq, rewardseq, stateseq_tsr, _ = episodeLoader(triali)
#         reward2go = torch.zeros(1).cuda()
#         L = min(len(actseq), T)
#         logactprob_mat = Pnet(stateseq_tsr[0: L].cuda())
#         value_vec = Vnet(stateseq_tsr[0: L+1].cuda())

#         with torch.no_grad():
#             logactprob_orig = Pnet_orig(stateseq_tsr[0: L].cuda())

#         logactprob_vec = logactprob_mat[torch.arange(L), actseq[0:L].long()]
#         logactprob_vec_orig = logactprob_orig[torch.arange(L), actseq[0:L].long()]
#         probratio_vec = (logactprob_vec - logactprob_vec_orig).exp()
#         cumprobratio_vec = torch.cumprod(probratio_vec, dim=0)

#         for t in range(L-1, -1, -1):
#             # reward2go = rewardseq[t] - baseline + gamma * reward2go
#             # ratio = (logactprob_mat[t, actseq[t]] - logactprob_orig[t, actseq[t]]).exp()
#             advantage = rewardseq[t] + gamma * value_vec[t + 1] - value_vec[t]  # advantage
#             surrogate += logactprob_mat[t, actseq[t]] * probratio_vec[t] * advantage
#             value_err += advantage ** 2

#         entropy_bonus += -(logactprob_mat * logactprob_mat.exp()).sum()#.sum(dim=1)

#     loss = 0.5 * value_err - (surrogate + beta * entropy_bonus)
#     loss.backward()  # retain_graph=True
#     Poptim.step()
#     Voptim.step()
#     print(f"Epoch {epi:d} Loss decomp Valuee L2 {value_err.item():.1f} surrogate {surrogate.item():.1f} entropy_err {entropy_bonus.item():.1f}")

