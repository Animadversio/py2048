
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD
from collections import OrderedDict


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


import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class Map3DCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Map3DCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(2, 2, 2), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).unsqueeze(1).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.unsqueeze(1)))


class MapCNN(BaseFeaturesExtractor):
  """
  :param observation_space: (gym.Space)
  :param features_dim: (int) Number of features extracted.
      This corresponds to the number of unit for the last layer.
  """

  def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, init_weights=None):
    super(MapCNN, self).__init__(observation_space, features_dim)
    # We assume CxHxW images (channels first)
    # Re-ordering will be done by pre-preprocessing or wrapper
    n_input_channels = observation_space.shape[0]
    self.cnn = nn.Sequential(
      nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
      nn.LeakyReLU(),
      nn.Flatten(),
    )

    # Compute shape by doing one forward pass
    with th.no_grad():
      n_flatten = self.cnn(
        th.as_tensor(observation_space.sample()[None]).float()
      ).shape[1]

    self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

  def forward(self, observations: th.Tensor) -> th.Tensor:
    return self.linear(self.cnn(observations))
