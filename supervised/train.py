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
import torch.optim as optim
from supervised.environment import *
from supervised.data import TickerData
from torch.utils.data import Dataset, DataLoader



def train_validate_split(length, split_pct=PCT_TRAIN):
    '''
    :param length: length of the total dataset
    :param split_pct: percentage to use for training
    :return np.array of indices for train, test
    '''
    total = np.arange(length)
    np.random.shuffle(total)  # inplace

    cutoff = int(length * split_pct)
    train, validate = total[:cutoff], total[cutoff:]

    return train, validate


def get_dl(ticker, num_state_space, batch_size):
    ticker = str.upper(ticker)
    ticker_file = f'iexfinance/iexdata/{ticker}'
    ticker_df = pd.read_csv(ticker_file)

    train_data_length = len(ticker_df) - num_state_space
    train, validate = train_validate_split(train_data_length)

    ticker_dataset = partial(TickerData, ticker=ticker,
                             num_state_space=num_state_space)

    train_dataloader = DataLoader(ticker_dataset(shuffled_index=train),
                                  num_workers=1, batch_size=batch_size)
    val_dataloader = DataLoader(ticker_dataset(shuffled_index=validate),
                                num_workers=1, batch_size=batch_size)

    return train_dataloader, val_dataloader


# models can be multiples, so are criterions, optimizers, schedulers...
def train_model(models, criterions, optimizers, schedulers, num_epochs=5):
    since = time.time()

    if not hasattr(models, '__iter__'):
        models, criterions, optimizers, schedulers = [models], [criterions], [optimizers], [schedulers]

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'validate']:
            if phase == 'train':
                for scheduler, model in zip(schedulers, models):
                    scheduler.step()
                    model.train()
            else:
                for model in models:
                    model.eval()

            # Fix here
            train_dl, val_dl = get_dl('aapl', NUM_STATE_SPACE, BATCH_SIZE)
            dl = train_dl if phase == 'train' else val_dl

            running_losses = [torch.tensor([0.0], dtype=torch.float).cuda()
                              for _ in range(len(models))]

            for cur_idx, (x, y) in enumerate(dl):

                x, y = map(lambda x: x.cuda(), (x, y))

                for optimizer in optimizers:
                    optimizer.zero_grad()

                outs = [model(x) for model in models]

                losses = [criterion(out, y.squeeze())
                          for criterion, out in zip(criterions, outs)]

                if phase == 'train':
                    for loss, optimizer in zip(losses, optimizers):
                        loss.backward(retain_graph=True)
                        optimizer.step()

                # I want to see outs at some point...

                for loss, running_loss in zip(losses, running_losses):
                    running_loss += loss.item() * x.size(0)
                    print(' Average loss: {:.4f}'.format(running_loss.item() /
                                                         (x.size(0) * (cur_idx + 1))))

            # Could be ...
            # epoch_losses = [running_loss / len(dl) for running_loss in running_losses]
            epoch_losses = []
            for running_loss in running_losses:
                epoch_losses.append(running_loss / len(dl))

            for n_iter, epoch_loss in enumerate(epoch_losses):
                print('{}th Model: {} loss: {:.4f}'.format(n_iter, phase,
                                                           epoch_loss.cpu().item()))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

