import os
import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class Temporal_I3D_SGA_STD(nn.Module):
    def __init__(self,dropout_rate):

        super(Temporal_I3D_SGA_STD, self).__init__()
        self.Regressor=nn.Sequential(nn.Linear(4096,512),nn.ReLU(), nn.Dropout(dropout_rate),nn.Linear(512, 32),nn.Dropout(dropout_rate),nn.Linear(32, 2))
        self.Softmax=nn.Softmax(dim=-1)

    def train(self,mode=True):
        super(Temporal_I3D_SGA_STD, self).train(mode)
        return self

    def forward(self,x,act=True,extract=False):
        logits=self.Regressor(x)
        if act:
            logits=self.Softmax(logits)
        return logits

