
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from main import getInitState, getSuccessor, getSuccessors, gameSimul, actions, sample
#%
episode_buffer = {}
def episodeLoader(triali):
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


def episodeSaver(triali, actseq, rewardseq, stateseq, score_tot):
    episode_buffer[triali] = torch.tensor(actseq), torch.tensor(rewardseq), \
                             torch.tensor(stateseq), score_tot



class Q_net(torch.nn.Module):
    def __init__(self, dimen=4):
        super().__init__()
        self.D = dimen
        # self.model = nn.Sequential(OrderedDict(
        #     [("Lin1", nn.Linear(dimen * dimen, 64)),
        #      ("ReLU1", nn.LeakyReLU(negative_slope=0.05)),
        #      ("BN1", nn.BatchNorm1d(64)),
        #      ("Lin2", nn.Linear(64, 128)),
        #      ("ReLU2", nn.LeakyReLU(negative_slope=0.05)),
        #      ("BN2", nn.BatchNorm1d(128)),
        #      ("Lin3", nn.Linear(128, 128)),
        #      ("ReLU3", nn.LeakyReLU(negative_slope=0.05)),
        #      ("BN3", nn.BatchNorm1d(128)),
        #      ("Lin4", nn.Linear(128, 64)),
        #      ("ReLU4", nn.LeakyReLU(negative_slope=0.05)),
        #      ("BN4", nn.BatchNorm1d(64)),
        #      ("Lin5", nn.Linear(64, 4)),
        #      ("ReLU5", nn.LeakyReLU(negative_slope=0.05)), ]))
        self.model = nn.Sequential(OrderedDict(
            [("Lin1", nn.Linear(dimen*dimen, 64)),
             ("ReLU1", nn.LeakyReLU(negative_slope=0.05)),
             ("BN1", nn.BatchNorm1d(64)),
             ("Lin2", nn.Linear(64, 128)),
             ("ReLU2", nn.LeakyReLU(negative_slope=0.05)),
             ("BN2", nn.BatchNorm1d(128)),
             ("Lin3", nn.Linear(128, 4)),
             ("ReLU3", nn.LeakyReLU(negative_slope=0.05)),]))

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
        # loss = F.smooth_l1_loss(discount * QnextMax + curRew, QActSel)
        loss = F.smooth_l1_loss((discount * QnextMax + curRew + 1).log2(), (QActSel + 1).log2())
        return loss

MAX_LOG2NUM = 16
import torch.nn.functional as F
class Q_CNN(torch.nn.Module):
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
             ("Conv3", nn.Conv2d(128, 256, 2, stride=1, padding=0, dilation=1,)),
             ("ReLU3", nn.LeakyReLU(negative_slope=0.05)),
             ("flatten", nn.Flatten(start_dim=1, end_dim=-1)),
             ("Lin4", nn.Linear(256, 4))]))

    def preprocess(self, stateseq):
        logstate = (1 + stateseq).float().log2().floor()
        logstatetsr = F.one_hot(logstate.long(), self.max_log2num).permute([0,3,1,2])
        return logstatetsr.float()

    def forward(self, stateseq):
        logstatetsr = self.preprocess(stateseq)
        Q = self.model(logstatetsr)
        return torch.pow(2, Q) - 1

    def Q_loss_TD_seq(self, stateseq, actseq, rewardseq, batch=100, discount=0.9, 
                    log2_loss=True, device="cuda"):
        # Getting target Q value with current model.
        if batch is None: batch = len(rewardseq)
        else: batch = min(batch, len(rewardseq))
        Qtab = self.forward(stateseq[:batch+1,:,:].to(device))
        QActSel = Qtab[torch.arange(batch, dtype=torch.int64), actseq[:batch].long()]
        QnextMax, QMaxAct = Qtab[1:, :].max(dim=1)
        curRew = rewardseq[:batch, ].float().to(device)
        # loss = (discount * QnextMax + curRew - QActSel).pow(2).mean()
        if log2_loss:
            loss = F.smooth_l1_loss((discount * QnextMax + curRew + 1).log2(), (QActSel + 1).log2())
        else:
            loss = F.smooth_l1_loss(discount * QnextMax + curRew, QActSel)
        return loss

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)

#%  Policy based on Q networks
def Qnet_policy(board, Qnet, device="cpu"):
    with torch.no_grad():
        Qval = Qnet(torch.tensor(board).unsqueeze(0).to(device))
        maxQ, maxact = Qval.max(1)
    return maxact.item(), maxQ.item()


def Qnet_ExpectiMax(board, Qnet, level=1, sampn=3):
    """ExpectiMax policy (Expectation over random fall of blocks), with certain depth termination.
    sampln determine the number 
    """
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

#%% Training loop routines
def train_episode_iter(Qnet, optimizer, epoc_num=50, epoc_start=0):
    """Go through the episodes one by one and do batch q learning for each of them"""
    Qnet.eval()
    _, score = gameSimul(Qnet_policy, {"Qnet": Qnet})
    print("Q policy scores: %.1f"%score)
    for epoc in tqdm(range(epoc_num)):
        Qnet.train()
        print("Starting Epoc %d"%epoc)
        for triali in tqdm(range(1000)):
            print("Starting episode %d"%triali)
            actseq, rewardseq, stateseq, score_tot = episodeLoader(triali)
            for step in range(1000):
                optimizer.zero_grad()
                loss = Qnet.Q_loss_TD_seq(stateseq, actseq, rewardseq, batch=200, discount=0.999)
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
                loss = Qnet.Q_loss_TD_seq(stateseq, actseq, rewardseq, batch=None)
            tot_loss += loss
        print("Finish learning all the trajectories! Mean loss for an episode is %.1f"%(tot_loss / 1000))
        actseq, score = gameSimul(Qnet_policy, {"Qnet": Qnet})
        print("Q policy scores: %.1f, %d steps"%(score, len(actseq)))

#%
def train_mix_episode_iter(Qnet, optimizer, epoc_num=50, writer=None, epoc_start=0,
                           device="cuda", log2_loss=False):
    """Go through the episodes one by one and do batch q learning for each of them"""
    write_rec = not (writer is None)
    Qnet.eval()
    actseq, score = gameSimul(Qnet_policy, {"Qnet": Qnet, "device": device})
    print("Q policy scores: %.1f, %d steps"%(score, len(actseq)))
    for epoc in tqdm(range(epoc_num)):
        Qnet.train()
        print("Starting Epoc %d"%epoc)
        perm_idx = np.random.permutation(1000)
        for triali in tqdm(range(1000)):
            actseq, rewardseq, stateseq, score_tot = episodeLoader(perm_idx[triali])
            optimizer.zero_grad()
            loss = Qnet.Q_loss_TD_seq(stateseq, actseq, rewardseq,
                                      batch=200, discount=0.999, device=device, log2_loss=log2_loss)
            loss.backward()
            optimizer.step()
            if (triali+1) % 100 == 0:
                print("Trial i %d loss %.2e"%(triali, loss))
        Qnet.eval()
        score_col = []
        actlen_col = []
        for i in range(10):
            actseq, score = gameSimul(Qnet_policy, {"Qnet": Qnet, "device": device}, printfreq=0)
            score_col.append(score)
            actlen_col.append(len(actseq))
            # print("Epoch %d Q net policy scores: %.1f, %d steps" % (epoc, score, len(actseq)))
        print("Epoch %d Q net policy score distrib (%.1f+-%.1f): %s" %
              (epoc, np.mean(score_col), np.std(score_col), score_col))
        if write_rec:
            writer.add_scalar("perf/mean_score", np.mean(score_col), epoc)
            writer.add_scalar("perf/mean_stepN", np.mean(actlen_col), epoc)
        print("Finish learning all the trajectories! ")
        Qnet.eval()
        tot_loss = 0
        for triali in range(1000):
            actseq, rewardseq, stateseq, score_tot = episodeLoader(triali)
            with torch.no_grad():
                loss = Qnet.Q_loss_TD_seq(stateseq, actseq, rewardseq,
                                      batch=None, device=device, log2_loss=log2_loss)
            tot_loss += loss
        print("Finish learning all the trajectories! Mean loss for an episode is %.1e" % (tot_loss / 1000))
        if write_rec:
            writer.add_scalar("eval/train_loss", tot_loss / 1000, epoc)

#%%
max_buffer_cnt = 1E4
from expCollector import traj_sampler
def train_roll_episode(Qnet, optimizer, epoc_num=50, writer=None, epoc_start=0,
                       device="cuda", log2_loss=False, simul_per_Nstep=5000):
    """Go through the episodes one by one and do batch q learning for each of them"""
    write_rec = not (writer is None)

    buffer_cnt = 1000
    buffer_idx = buffer_cnt
    max_buffer_cnt = 1E4

    Qnet.eval()
    actseq, score = gameSimul(Qnet_policy, {"Qnet": Qnet, "device": device})
    print("Q policy scores: %.1f, %d steps"%(score, len(actseq)))
    for epoc in tqdm(range(epoc_num)):
        Qnet.train()
        print("Starting Epoc %d"%epoc)
        perm_idx = np.random.permutation(buffer_cnt)
        for triali in tqdm(range(simul_per_Nstep)):
            actseq, rewardseq, stateseq, score_tot = episodeLoader(perm_idx[triali])
            optimizer.zero_grad()
            loss = Qnet.Q_loss_TD_seq(stateseq, actseq, rewardseq,
                                      batch=200, discount=0.999, device=device, log2_loss=log2_loss)
            loss.backward()
            optimizer.step()
            if (triali+1) % 100 == 0:
                print("Trial i %d loss %.2e"%(triali, loss))
                if write_rec:
                    writer.add_scalar("eval/loss", loss, epoc * simul_per_Nstep + triali)
        Qnet.eval()
        score_col = []
        actlen_col = []
        for i in range(20):
            # actseq, score = gameSimul(Qnet_policy, {"Qnet": Qnet, "device": device}, printfreq=0)
            stateseq, actseq, rewardseq, score = traj_sampler(Qnet_policy, policyArgs={"Qnet": Qnet, "device": device}, printfreq=0)
            episodeSaver(buffer_idx, actseq, rewardseq, stateseq, score)
            buffer_cnt = min(buffer_cnt + 1, max_buffer_cnt)
            buffer_idx = (buffer_idx + 1) % max_buffer_cnt

            score_col.append(score)
            actlen_col.append(len(actseq))
            # print("Epoch %d Q net policy scores: %.1f, %d steps" % (epoc, score, len(actseq)))
        print("Epoch %d Q net policy score distrib (%.1f+-%.1f): %s" %
              (epoc, np.mean(score_col), np.std(score_col), score_col))
        if write_rec:
            writer.add_scalar("perf/mean_score", np.mean(score_col), epoc * simul_per_Nstep)
            writer.add_scalar("perf/mean_stepN", np.mean(actlen_col), epoc * simul_per_Nstep)

        print("Finish learning all the trajectories! ")
        Qnet.eval()
        tot_loss = 0
        for triali in range(1000):
            actseq, rewardseq, stateseq, score_tot = episodeLoader(triali)
            with torch.no_grad():
                loss = Qnet.Q_loss_TD_seq(stateseq, actseq, rewardseq,
                                      batch=None, device=device, log2_loss=log2_loss)
            tot_loss += loss
        print("Finish learning all the trajectories! Mean loss for an episode is %.1e" % (tot_loss / 1000))
        if write_rec:
            writer.add_scalar("eval/train_loss", tot_loss / 1000, epoc * simul_per_Nstep)
#%%
if __name__ == "__main__":
    # %% Experiment with CNN + roll buffer
    Qnet = Q_CNN().cuda()
    optimizer = SGD(list(Qnet.parameters()), lr=3e-3, )
    writer = SummaryWriter("logs\\CNN3lyr_L1_rollbuffer")
    train_roll_episode(Qnet, optimizer, writer=writer, epoc_num=100, log2_loss=False,
                       simul_per_Nstep=1000)
    torch.save(Qnet.state_dict(), "ckpt\\Qnet_L1loss_rollbuff_100epc.pt", )


    Qnet = Q_CNN().cuda()
    optimizer = SGD(list(Qnet.parameters()), lr=4e-3, )
    writer = SummaryWriter("logs\\CNN3lyr_logL1_rollbuffer")
    train_roll_episode(Qnet, optimizer, writer=writer, epoc_num=100, log2_loss=True,
                       simul_per_Nstep=1000)
    torch.save(Qnet.state_dict(), "ckpt\\Qnet_logL1loss_rollbuff_100epc.pt", )
    print()
    #%%
    Qnet = Q_net()
    Qnet.model.apply(weights_init)
    optimizer = SGD(list(Qnet.parameters()), lr=1e-2, )
    train_mix_episode_iter(Qnet, optimizer, epoc_num=50)

    #%% Experiment with CNN
    Qnet = Q_CNN().cuda()
    optimizer = SGD(list(Qnet.parameters()), lr=2e-3, )
    writer = SummaryWriter("logs//CNN3lyr_L1")
    train_mix_episode_iter(Qnet, optimizer, writer=writer, epoc_num=100, log2_loss=False)
    torch.save(Qnet.state_dict(), "ckpt\\Qnet_L1loss_100epc.pt", )

    #% Experiment with CNN
    Qnet = Q_CNN().cuda()
    optimizer = SGD(list(Qnet.parameters()), lr=2e-3, )
    writer = SummaryWriter("logs//CNN3lyr_logL1")
    train_mix_episode_iter(Qnet, optimizer, writer=writer, epoc_num=100, log2_loss=True)
    torch.save(Qnet.state_dict(), "ckpt\\Qnet_logL1loss_100epc.pt", )




    #%%
    board = np.array( [[  2,  64,  16,   2],
                       [  8,   4,   2,   4],
                       [128, 128, 256,  16],
                       [  4,  16,   4,   2]])

    Qnet = Q_CNN()
    # Qnet.model.apply(weights_init)
    Qnet_policy(board, Qnet)
    _, score = gameSimul(Qnet_policy, {"Qnet":Qnet}) # Qnet_ExpectiMax



    #%%
    board = np.array( [[  2,  64,  16,   2],
                       [  8,   4,   2,   4],
                       [128, 128, 256,  16],
                       [  4,  16,   4,   2]])
    Qnet = Q_net()
    Qnet.model.apply(weights_init)
    #%%
    Qnet.eval()
    Qnet.forward(torch.tensor(board).unsqueeze(0))
    # Q_approx(state)


    #%% Original MLP + SGD + episode iteration
    Qnet = Q_net()
    Qnet.model.apply(weights_init)
    # optimizer = Adam(list(Qnet.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0001)
    optimizer = SGD(list(Qnet.parameters()), lr=1e-2, )
    train_episode_iter(Qnet, optimizer, epoc_num=50)
    torch.save(Qnet.state_dict(), "ckpt\\Qnet_1000_50epc.pt", )
    #%%
    Qnet = Q_net()
    Qnet.load_state_dict(torch.load("ckpt\\Qnet_50epc.pt"))

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