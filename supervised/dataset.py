import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from supervised.environment import *


def digitize(y):
    bins = np.linspace(-MAX_POSSIBLE_VALUE, MAX_POSSIBLE_VALUE,
                       NUM_DISCRETE_RETURNS-1)
    return np.digitize(y, bins).item()


class TickerData(Dataset):
    def __init__(self, ticker, num_state_space, shuffled_index):
        self.ticker = str.upper(ticker)
        self.num_state_space = num_state_space
        self.x, self.y = self.load_df()
        self.index = shuffled_index

    def load_df(self):
        df = pd.read_csv(f'iexfinance/iexdata/{self.ticker}') \
            .drop('date', axis=1)
        close_delta = np.log(df.close) - np.log(df.close.shift(1))
        close_delta[0] = 0.0
        stacked = [close_delta[i:self.num_state_space + i]
                   for i in range(len(df) - self.num_state_space)]
        stacked = pd.DataFrame(np.column_stack(stacked))
        target_delta = np.array(close_delta[self.num_state_space:])

        #   If target is to be sum of days
        # close_delta = np.log(df.close.shift(-1)) - np.log(df.close)
        # close_delta[-1] = 0.0
        # stacked = [close_delta[i:40 + i]
        #            for i in range(len(df)-40)]
        # stacked = pd.DataFrame(np.column_stack(stacked))
        # target_delta = np.log(df.close.shift(-self.num_state_space)) - \
        #                np.log(df.close)

        return stacked, target_delta

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        index = self.index[index]
        x = torch.DoubleTensor(self.x[index]).unsqueeze(-1)
        y = torch.LongTensor([digitize(self.y[index])])
        return x, y