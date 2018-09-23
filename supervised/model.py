import gym
import gym_exchange
import random
import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
import functools
from functools import partial
from copy import deepcopy
import datetime
from itertools import count
import math
import logging
import matplotlib.pyplot as plt
import numpy as np
from random import choice
import time
from utils import device, train_dqn
import seaborn as sns
from collections import deque, Counter
from supervised.environment import *


class Model(torch.nn.Module):

    def __init__(self, input_size, rnn_hidden_size,
                 output_size, batch_size, num_layers):
        super(Model, self).__init__()
        self.rnn = torch.nn.GRU(input_size, rnn_hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.h_0 = None
        # self.h_0 = self.initialize_hidden(rnn_hidden_size, batch_size,
        #                                  num_layers)
        self.linear = nn.Linear(rnn_hidden_size, output_size)

    def forward(self, x):

        # YES ONE LINER POSSIBLE...
        # if self.h_0 is None:
        #     out, self.h_0 = self.rnn(x)
        # else:
        #     out, self.h_0 = self.rnn(x, self.h_0)
        out, self.h_0 = self.rnn(x)

        x = out[:, -1:, :].squeeze().cuda()
        return self.linear(x)

    def initialize_hidden(self, rnn_hidden_size, batch_size, num_layers):
        # n_layers * n_directions, batch_size, rnn_hidden_size
        return torch.randn(num_layers, batch_size, rnn_hidden_size,
                           requires_grad=True, device=device, dtype=torch.float64)

