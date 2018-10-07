import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from supervised.environment import *
from supervised.utils import iterable


def digitize(y):
    bins = np.linspace(-MAX_POSSIBLE_VALUE, MAX_POSSIBLE_VALUE,
                       NUM_DISCRETE_RETURNS-1)
    return np.digitize(y, bins).item()


# EVENTUALLY, PORTFOLIO BE THE ONLY INTERFACE
class TickerData(Dataset):
    def __init__(self, ticker, num_state_space, shuffled_index):
        '''
        :param ticker: string
        :param num_state_space: number of days used as an input `x`
        :param shuffled_index: an iterable of indices
        '''
        self.ticker = str.upper(ticker)
        self.num_state_space = num_state_space
        self.x, self.y = self.load_df(self.ticker, num_state_space)
        self.index = shuffled_index

    @classmethod
    def load_df(cls, ticker, num_state_space):
        '''
        classmethod for easy use to other inheriting classes
        '''
        df = pd.read_csv(f'iexfinance/iexdata/{ticker}') \
            .drop('date', axis=1)
        close_delta = np.log(df.close) - np.log(df.close.shift(1))
        close_delta[0] = 0.0

        stacked = [close_delta[i:num_state_space + i]
                   for i in range(len(df)-num_state_space)]

        # pd.DataFrame is necessary
        stacked = pd.DataFrame(np.column_stack(stacked))

        target_delta = np.array(close_delta[num_state_space:])

        return stacked, target_delta

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        index = self.index[index]
        x = torch.FloatTensor(self.x[index]).unsqueeze(-1)
        y = torch.FloatTensor([self.y[index]])
        return x, y


class TickerDataDiscreteReturn(TickerData):
    def __getitem__(self, index):
        index = self.index[index]
        x = torch.DoubleTensor(self.x[index]).unsqueeze(-1)
        y = torch.LongTensor([digitize(self.y[index])])
        return x, y


# hmm looks like it doesn't need to inherit...
class PortfolioData(TickerData):
    def __init__(self, tickers, num_state_space, shuffled_index, transform=None):
        '''
        :param tickers: an iterable of strings
        :param num_state_space: number of days used as an input `x`
        :param shuffled_index: an iterable of indices
        '''
        assert iterable(tickers), 'tickers must be an iterable'
        self.tickers = [str.upper(ticker) for ticker in tickers]
        self.num_state_space = num_state_space
        self.index = shuffled_index
        self.xs, self.ys = self.load_tickers()
        self.transform = transform

    def load_tickers(self):
        # xs will be of dimension 3
        xs, ys = [], []
        for ticker in self.tickers:
            x, y = self.load_df(ticker, self.num_state_space)
            xs += [x.values[np.newaxis, ...]]
            ys += [y[np.newaxis, ...]]

        # After rollings, we have tickers packed in the last axis
        #     For example, if two tickers, we would have
        #     xs_concat in a shape of [?, self.num_state_space, 2]
        xs_concat = np.concatenate(xs)
        xs_concat = np.rollaxis(xs_concat, 2, 0)
        xs_concat = np.rollaxis(xs_concat, 2, 1)

        ys_concat = np.concatenate(ys)
        ys_concat = np.rollaxis(ys_concat, 1, 0)

        return xs_concat, ys_concat

    def __getitem__(self, index):
        index = self.index[index]
        x = self.xs[index]
        y = self.ys[index]

        if self.transform:
            x, y = self.transform(x, y)

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y
