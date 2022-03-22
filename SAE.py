import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable                                                                                     # 使用GPU来训练
import math

class SAE(nn.Module):
    def __init__(self, size, n_latent):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(size, n_latent)
        self.fc2 = nn.Linear(n_latent, size)
        self.activation = nn.Sigmoid()                                                                                  # 中间的嵌入向量层进行了归一化处理

    def forward(self, x0):                                                                                              # 原文中有一个偏置项
        x1 = self.activation(self.fc1(x0))
        x2 = self.fc2(x1)

        return x2