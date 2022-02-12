# PPO.py
import torch
import torch.nn as nn
import torch.nn.functional as F
MAX_LOG2NUM = 16

class policy_CNN(torch.nn.Module):
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
             ("Lin4", nn.Linear(256, 4)),
             ("softmax", nn.SoftMax())]))

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


def Value_CNN():
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
             ("Conv3", nn.Conv2d(128, 256, 2, stride=1, padding=0, dilation=1,)),
             ("ReLU3", nn.LeakyReLU(negative_slope=0.05)),
             ("flatten", nn.Flatten(start_dim=1, end_dim=-1)),
             ("Lin4", nn.Linear(256, 128)),
             ("ReLU4", nn.LeakyReLU(negative_slope=0.05)),
             ("BN4", nn.BatchNorm2d(128)),
             ("Lin5", nn.Linear(128, 1)),]))

    def preprocess(self, stateseq):
        logstate = (1 + stateseq).float().log2().floor()
        logstatetsr = F.one_hot(logstate.long(), self.max_log2num).permute([0,3,1,2])
        return logstatetsr.float()

    def forward(self, stateseq):
        logstatetsr = self.preprocess(stateseq)
        return self.model(logstatetsr)





def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)

#%  Policy based on Q networks
def Pnet_policy(board, Pnet, device="cpu"):
    with torch.no_grad():
        prob = Pnet(torch.tensor(board).unsqueeze(0).to(device))
        choices = torch.multinomial(prob, num_samples=1) # output is B-by-1
    return choices




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


