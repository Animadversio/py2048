import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
#%%
triali = 0
def episodeLoader(triali):
    data = np.load("exp_data\\traj%03d.npz"%triali)
    actseq = torch.tensor(data['actseq'])  # (T, )
    rewardseq = torch.tensor(data['rewardseq'])  # (T, )
    stateseq = torch.tensor(data['stateseq'])  # (T+1, 4, 4)
    score_tot = data['score']
    return actseq, rewardseq, stateseq, score_tot

class Q_net(torch.nn.Module):
    def __init__(self, dimen=4):
        super().__init__()
        self.D = dimen
        self.model = nn.Sequential(OrderedDict(
            [("Lin1", nn.Linear(dimen*dimen, 64)),
             ("ReLU1", nn.LeakyReLU(negative_slope=0.05)),
             ("BN1", nn.BatchNorm1d(64)),
             ("Lin2", nn.Linear(64, 128)),
             ("ReLU2", nn.LeakyReLU(negative_slope=0.05)),
             ("BN2", nn.BatchNorm1d(128)),
             ("Lin3", nn.Linear(128, 128)),
             ("ReLU3", nn.LeakyReLU(negative_slope=0.05)),
             ("BN3", nn.BatchNorm1d(128)),
             ("Lin4", nn.Linear(128, 64)),
             ("ReLU4", nn.LeakyReLU(negative_slope=0.05)),
             ("BN4", nn.BatchNorm1d(64)),
             ("Lin5", nn.Linear(64, 4)),
             ("ReLU5", nn.LeakyReLU(negative_slope=0.05)),]))

    def preprocess(self, stateseq):
        logstate = stateseq.float().log2()
        logstate[logstate == float("-inf")] = 0
        return logstate

    def forward(self, stateseq):
        logstate = self.preprocess(stateseq)
        Q = self.model(logstate.view(-1, self.D * self.D))
        return torch.pow(2, Q) - 1

    def process_seq(self, stateseq, actseq, rewardseq, batch=100, discount=0.9):
        # Getting target Q value with current model.
        if batch is None: batch = len(rewardseq)
        else: batch = min(batch, len(rewardseq))
        Qtab = self.forward(stateseq[:batch+1,:,:])
        QActSel = Qtab[torch.arange(batch, dtype=torch.int64), actseq[:batch].to(torch.int64)]
        QnextMax, QMaxAct = Qtab[1:, :].max(dim=1)
        curRew = rewardseq[:batch, ].float()
        # loss = (discount * QnextMax + curRew - QActSel).pow(2).mean()
        loss = F.smooth_l1_loss(discount * QnextMax + curRew, QActSel)
        return loss

    def episode_loss(self, stateseq, actseq, rewardseq):
        pass
#%%
Qnet = Q_net()
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
Qnet.model.apply(weights_init)
#%%
from torch.optim import Adam, SGD
optimizer = Adam(list(Qnet.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0001)
#%%
from tqdm import tqdm
Qnet.eval()
_, score = gameSimul(Qnet_policy, {"Qnet": Qnet})
print("Q policy scores: %.1f"%score)
for epoc in tqdm(range(50)):
    Qnet.train()
    print("Starting Epoc %d"%epoc)
    for triali in tqdm(range(1000)):
        print("Starting episode %d"%triali)
        actseq, rewardseq, stateseq, score_tot = episodeLoader(triali)
        for step in range(1000):
            optimizer.zero_grad()
            loss = Qnet.process_seq(stateseq, actseq, rewardseq, batch=200, discount=0.999)
            loss.backward()
            optimizer.step()
            if (step+1)%100 == 0:
                print("Step %d loss %.2e"%(step, loss))
    print("Finish learning all the trajectories! ")
    Qnet.eval()
    tot_loss = 0
    for triali in tqdm(range(1000)):
        actseq, rewardseq, stateseq, score_tot = episodeLoader(triali)
        with torch.no_grad():
            loss = Qnet.process_seq(stateseq, actseq, rewardseq, batch=None)
        tot_loss += loss
    print("Finish learning all the trajectories! Mean loss for an episode is %.1f"%(tot_loss / 1000))
    _, score = gameSimul(Qnet_policy, {"Qnet": Qnet})
    print("Q policy scores: %.1f"%score)


#%%
torch.save(Qnet.state_dict(), "ckpt\\Qnet_1000_50epc.pt", )
#%%
Qnet = Q_net()
Qnet.load_state_dict(torch.load("ckpt\\Qnet_50epc.pt"))
#%% Deployment
from main import getInitState, getSuccessor, getSuccessors, gameSimul, actions, sample
def Qnet_policy(board, Qnet):
    with torch.no_grad():
        Qval = Qnet(torch.tensor(board).unsqueeze(0))
        maxQ, maxact = Qval.max(1)
    return maxact.item(), maxQ.item()

def Qnet_ExpectiMax(board, Qnet, level=1, sampn=3):
    """ExpectiMax policy (Expectation over random fall of blocks), with certain depth termination.
    sampln determine the number """
    bestVal = -1E6
    bestAct = None
    for act in actions:
        nextboards, reward, finished = getSuccessors(board, action=act, clone=True)
        if level > 0 and not finished:
            value_col = []
            sampn = min(sampn, len(nextboards))
            for nextboard in sample(nextboards, sampn):
                sampvalue, nextact = Qnet_ExpectiMax(nextboard, Qnet, level=level-1, sampn=4)
                value_col.append(sampvalue)
            nextvalue = sum(value_col) / sampn
        else:
            with torch.no_grad():
                Qval = Qnet(torch.tensor(nextboards).unsqueeze(0))
                maxQ, maxact = Qval.max(1)
            nextvalue = maxQ.mean().item()  # evaluation function!
        curvalue = nextvalue + reward - (finished) * 1000 # punishment for death
        if curvalue > bestVal: bestVal, bestAct = curvalue, act
    return bestAct, bestVal
#%%
board = np.array([[  2,  64,  16,   2],
       [  8,   4,   2,   4],
       [128, 256, 128,  16],
       [  4,  16,   4,   2]])
Qnet_policy(board, Qnet)

_, score = gameSimul(Qnet_policy, {"Qnet":Qnet}) # Qnet_ExpectiMax


# Q_approx(state)

# return Q # 1 by 4 Q values of the 4 actions.
# def gameSimul(policy, policyArgs={}, initboard=None, initscore=0):
#     board = getInitState() if initboard is None else initboard
#     score = initscore
#     actseq = []
#     while True:
#         # act, bestVal, = RndMax(board, 3)
#         act, bestVal, = policy(board, **policyArgs)
#         # act = choice(actions)
#         actseq.append(act)
#         board, reward, finished = getSuccessor(board, action=act, show=False)
#         score += reward
#         if len(actseq) % 50==0:
#             print("Step %d score %d"%(len(actseq), score))
#             print(board)
#         if finished:
#             print("Game Over, step %d score %d"%(len(actseq), score))
#             break
#     return actseq, score