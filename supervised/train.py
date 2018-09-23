import pandas as pd
import torch
from functools import partial
import numpy as np
import time
from collections import Counter
from supervised.environment import *
from supervised.dataset import TickerData
from torch.utils.data import DataLoader


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


# Need to create inputs to `get_dl`... look at `Fix here:82`
# models can be multiples, so are criterions, optimizers, schedulers...
def train_model(models, criterions, optimizers, schedulers, num_epochs=5):

    def count_outcomes(outs_list, counters_list):
        # Record keeping for counting the max arguments
        for out, counter in zip(outs_list, counters_list):
            # out is from a batch, so sort by index 1
            # argmax can be used, but for later uses, just sort it...
            idxs = out.sort(1, descending=True)[1][:, 0].tolist()
            for idx in idxs:
                counter[idx] += 1
        return counters_list

    since = time.time()

    if not hasattr(models, '__iter__'):
        models, criterions, optimizers, schedulers = [models], [criterions], [optimizers], [schedulers]

    counters = [Counter() for _ in range(len(models))]

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

                counters = count_outcomes(outs, counters)

                losses = [criterion(out, y.squeeze())
                          for criterion, out in zip(criterions, outs)]

                if phase == 'train':
                    for loss, optimizer in zip(losses, optimizers):
                        loss.backward(retain_graph=True)
                        optimizer.step()

                for loss, running_loss, counter in zip(losses, running_losses, counters):
                    running_loss += loss.item() * x.size(0)
                    print('\rAverage loss: {:.4f}, Outs count: {}'.format(running_loss.item()/(x.size(0)*(cur_idx + 1)),
                                                                          counter), end='')

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

