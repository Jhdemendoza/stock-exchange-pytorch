import torch
import numpy as np
import pandas as pd
import pickle
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


class TickersData(Dataset):
    def __init__(self, ticker_list, last_file_path, y_transform=None):
        '''
        :param ticker_list: iterable tickers
        :param last_file_path: pickle_file (e.g. _train.pickle, _test.pickle)
        '''
        self.tickers = ticker_list
        self.x = self.read_in_pickles('_x' + last_file_path)
        self.y = self.read_in_pickles('_y' + last_file_path)

        self._sanity_check()

        if y_transform is not None:
            self.y = y_transform(self.y).astype(np.float64)

    def read_in_pickles(self, last_file_path):
        numpy_tickers = []
        for ticker in self.tickers:
            with open('data/ohlc_processed/' + ticker + last_file_path, 'rb') as f:
                data = pickle.load(f)
                numpy_tickers += [data]
        numpy_tickers = np.concatenate(numpy_tickers, axis=1)
        numpy_tickers = numpy_tickers.astype(np.float64)
        return numpy_tickers

    def _sanity_check(self):
        assert self.x.shape[0] == self.y.shape[0], 'x.shape != y.shape: {} vs {}'.format(
            x.shape, y.shape)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


class TickerDataSimple(Dataset):
    def __init__(self, ticker, x, y):
        '''
        :param ticker: string
        :param x: np.array of x
        :param y: np.array of y
        '''
        self.ticker = ticker
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y