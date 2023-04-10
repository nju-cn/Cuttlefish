import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class ActorCritic(torch.nn.Module):
    def __init__(self, actionNum, k):   # k is the length of historical configuration decisions
        super(ActorCritic, self).__init__()
        self.critic_fps_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)
        self.critic_resolution_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)
        self.critic_fc0 = nn.Linear(2*k, 256)
        self.critic_fc1 = nn.Linear(256, 1)
        self.actor_fps_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)
        self.actor_resolution_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)
        self.actor_fc0 = nn.Linear(2 * k, 256)
        self.actor_fc1 = nn.Linear(256, actionNum)

        #self.pi4=  nn.Linear(1000,256)  #feature map
        # self.merge_net=nn.Linear(6*256,256)
        #
        # self.d=nn.Linear(256,actionNum)
        # self.v=nn.Linear(256,1)
        set_init([self.critic_fps_conv, self.critic_resolution_conv, self.critic_fc0, self.critic_fc1,
                  self.actor_fps_conv, self.actor_resolution_conv, self.actor_fc0, self.actor_fc1])
        # self.distribution = torch.distributions.Categorical

    def forward(self, x, feature_map):

        conv_1 = F.relu6(self.critic_fps_conv(x[0]))
        conv_2 = F.relu6(self.critic_resolution_conv(x[1]))

        cat_tensor = torch.cat((conv_1[:, 0, :], conv_2[:, 0, :], x[2]), 1)
        fc0 = F.relu6(self.critic_fc0(cat_tensor))
        values = F.relu6(self.critic_fc1(fc0))

        conv_1 = F.relu6(self.actor_fps_conv(x[0]))
        conv_2 = F.relu6(self.actor_resolution_conv(x[1]))

        cat_tensor = torch.cat((conv_1[:, 0, :], conv_2[:, 0, :], x[2]), 1)
        fc0 = F.relu6(self.actor_fc0(cat_tensor))
        logits = F.relu6(self.actor_fc1(fc0))

        return logits, values

