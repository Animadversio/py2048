# PPO.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD
from collections import OrderedDict
from main import getInitState, getSuccessor, getSuccessors, gameSimul, actions, sample
import copy

MAX_LOG2NUM = 16
episode_buffer = {}
def episodeLoader(triali, episode_buffer=episode_buffer):
    if triali not in episode_buffer:
        data = np.load("exp_data\\traj%03d.npz"%triali)
        actseq = torch.tensor(data['actseq'])  # (T, )
        rewardseq = torch.tensor(data['rewardseq'])  # (T, )
        stateseq = torch.tensor(data['stateseq'])  # (T+1, 4, 4)
        score_tot = data['score']
        episode_buffer[triali] = actseq, rewardseq, stateseq, score_tot
        return actseq, rewardseq, stateseq, score_tot
    else:
        actseq, rewardseq, stateseq, score_tot = episode_buffer[triali]
        return actseq, rewardseq, stateseq, score_tot


def episodeSaver(triali, actseq, rewardseq, stateseq, score_tot, episode_buffer=episode_buffer):
    episode_buffer[triali] = torch.tensor(actseq), torch.tensor(rewardseq), \
                             torch.tensor(stateseq), score_tot


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

    def forward(self, stateseq):
        logstatetsr = self.preprocess(stateseq)
        return self.model(logstatetsr).exp()


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
triali = 10
data = np.load("exp_data\\traj%03d.npz"%triali)
stateseq = data["stateseq"]
actseq = data["actseq"]
rewardseq = data["rewardseq"]
score = data["score"]
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
""" Policy gradient """
Pnet = policy_CNN()
Pnet.cuda()
Poptim = Adam([*Pnet.parameters()], lr=0.05)
#%%
gamma = 0.9
T = 100
Poptim.zero_grad()
stateseq_tsr = torch.tensor(stateseq).cuda()
surrogate = torch.zeros(1).cuda()
reward2go = torch.zeros(1).cuda()
logactprob_mat = Pnet(stateseq_tsr[0: T])
for t in range(T-1, -1, -1):
    state_cur = stateseq_tsr[t:t + 1]
    state_nxt = stateseq_tsr[t + 1:t + 2]
    reward2go = rewardseq[t] + gamma * reward2go
    logactprob = Pnet(state_cur)
    surrogate += logactprob[0, actseq[t]] * reward2go

surrogate.backward()  # retain_graph=True

#%% Simplified Policy Gradient
B = 50
T = 100
gamma = 0.9
Poptim.zero_grad()
for triali in range(B):
    actseq, rewardseq, stateseq, _ = episodeLoader(triali)
    surrogate = torch.zeros(1).cuda()
    reward2go = torch.zeros(1).cuda()
    logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())
    for t in range(T-1, -1, -1):
        reward2go = rewardseq[t] + gamma * reward2go
        surrogate += logactprob_mat[t, actseq[t]] * reward2go

    surrogate.backward()  # retain_graph=True

Poptim.step()
#%% Policy Gradient with fixed baseline
Poptim.zero_grad()
B = 50
baseline = 0
for triali in range(B):
    actseq, rewardseq, stateseq, _ = episodeLoader(triali)
    surrogate = torch.zeros(1).cuda()
    reward2go = torch.zeros(1).cuda()
    logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())
    for t in range(T-1, -1, -1):
        reward2go = rewardseq[t] - baseline + gamma * reward2go
        surrogate += logactprob_mat[t, actseq[t]] * reward2go

    surrogate.backward()  # retain_graph=True

Poptim.step()

#%% Policy Gradient with Value function (Actor Critic)
Pnet = policy_CNN()
Vnet = Value_CNN()
Pnet.cuda()
Vnet.cuda()
Poptim = Adam([*Pnet.parameters()], lr=0.05)
Voptim = Adam([*Vnet.parameters()], lr=0.05)

# update value and policy using current dataset
Poptim.zero_grad()
Voptim.zero_grad()
B = 50
baseline = 0
value_err = torch.zeros(1).cuda()
surrogate = torch.zeros(1).cuda()
for triali in range(B):
    actseq, rewardseq, stateseq, _ = episodeLoader(triali)
    reward2go = torch.zeros(1).cuda()
    logactprob_mat = Pnet(stateseq_tsr[0: T].cuda())
    value_vec = Vnet(stateseq_tsr[0: T+1].cuda())
    for t in range(T-1, -1, -1):
        # reward2go = rewardseq[t] - baseline + gamma * reward2go
        advantage = rewardseq[t] + gamma * value_vec[t + 1] - value_vec[t]  # advantage
        surrogate += logactprob_mat[t, actseq[t]] * advantage
        value_err += advantage ** 2

loss = 0.1 * value_err - surrogate  # maximize surrogate, minimize value_err
loss.backward()  # retain_graph=True
Poptim.step()


#%% Actor Critic with importance sampling
# https://julien-vitay.net/deeprl/ImportanceSampling.html
Pnet = policy_CNN().cuda()
Vnet = Value_CNN().cuda()

Poptim = Adam([*Pnet.parameters()], lr=0.01)
Voptim = Adam([*Vnet.parameters()], lr=0.01)

T = 200
B = 50
beta = 0.1
epsilon = 0.2
gamma = 0.9
# update value and policy using current dataset
Pnet_orig = copy.deepcopy(Pnet)
Pnet_orig.requires_grad_(False)

for epi in range(10):
    Poptim.zero_grad()
    Voptim.zero_grad()
    surrogate = torch.zeros(1).cuda()
    value_err = torch.zeros(1).cuda()
    entropy_bonus = torch.zeros(1).cuda()
    for triali in range(B):
        actseq, rewardseq, stateseq_tsr, _ = episodeLoader(triali)
        reward2go = torch.zeros(1).cuda()
        L = min(len(actseq), T)
        logactprob_mat = Pnet(stateseq_tsr[0: L].cuda())
        value_vec = Vnet(stateseq_tsr[0: L+1].cuda())

        with torch.no_grad():
            logactprob_orig = Pnet_orig(stateseq_tsr[0: L].cuda())

        logactprob_vec = logactprob_mat[torch.arange(L), actseq[0:L].long()]
        logactprob_vec_orig = logactprob_orig[torch.arange(L), actseq[0:L].long()]
        probratio_vec = (logactprob_vec - logactprob_vec_orig).exp()
        cumprobratio_vec = torch.cumprod(probratio_vec, dim=0)

        for t in range(L-1, -1, -1):
            # reward2go = rewardseq[t] - baseline + gamma * reward2go
            # ratio = (logactprob_mat[t, actseq[t]] - logactprob_orig[t, actseq[t]]).exp()
            advantage = rewardseq[t] + gamma * value_vec[t + 1] - value_vec[t]  # advantage
            surrogate += logactprob_mat[t, actseq[t]] * probratio_vec[t] * advantage
            value_err += advantage ** 2

        entropy_bonus += -(logactprob_mat * logactprob_mat.exp()).sum()#.sum(dim=1)

    loss = 0.5 * value_err - (surrogate + beta * entropy_bonus)
    loss.backward()  # retain_graph=True
    Poptim.step()
    Voptim.step()
    print(f"Epoch {epi:d} Loss decomp Valuee L2 {value_err.item():.1f} surrogate {surrogate.item():.1f} entropy_err {entropy_bonus.item():.1f}")

#%% Proximal Policy Gradient

#%% Training cycle
from expCollector import traj_sampler
B = 400
T = 200
beta = 5
epsilon = 0.2
gamma = 0.9
update_epoch = 10
gradstep_freq = 20
Pnet = policy_CNN().cuda()
Vnet = Value_CNN().cuda()
Poptim = Adam([*Pnet.parameters()], lr=0.005)
Voptim = Adam([*Vnet.parameters()], lr=0.005)
#%%
for cycle in range(20):
    # collect data
    Pnet.eval()
    onpolicy_buffer = {}
    score_list = []
    for runi in range(B):
        # actseq, score = gameSimul(Pnet_policy, {"Pnet": Pnet, "device": "cuda"}, printfreq=0)
        stateseq, actseq, rewardseq, score = traj_sampler(Pnet_policy, policyArgs={"Pnet": Pnet, "device": "cuda"},
                                                          printfreq=0)
        episodeSaver(runi, actseq, rewardseq, stateseq, score, episode_buffer=onpolicy_buffer)
        score_list.append(score)
    print(f"iteration {cycle:d} summary {np.mean(score_list):.2f}+-{np.std(score_list):.2f}")

    # update model
    Pnet_orig = copy.deepcopy(Pnet)
    Pnet_orig.requires_grad_(False)
    Pnet.train()
    Vnet.train()
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
                surrogate += logactprob_mat[t, actseq[t]] * probratio_vec[t] * advantage
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
# iteration 0 summary 773.05+-383.55
# iteration 1 summary 751.86+-387.57

#%%
# Policy agent 


# collect Episodes

# update value and policy using current dataset
gamma = 0.95
# Value objective 
# Policy objective 
Voptim = Adam([*Vnet.parameters()], lr=0.05)
Poptim = Adam([*Pnet.parameters()], lr=0.05)
R_term = Vnet(state_n)
value_loss = torch.tensor(0.0).cuda()
policy_loss = torch.tensor(0.0).cuda()
Voptim.zero_grad() 
Poptim.zero_grad()
for t in range():
    probs = Pnet(state_t)
    value_loss.backward()    
    policy_loss.backward()

Voptim.step()
Poptim.step()
    # grad descent. 


