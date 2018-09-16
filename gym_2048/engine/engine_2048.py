import collections
import datetime
import itertools
import functools
import numpy as np
import pandas as pd
import six


class Ticker:
    def __init__(self, ticker, start_date, num_days_iter,
                 today=None, num_actions=21, test=False):
        self.ticker = str.upper(ticker)
        self.start_date = start_date
        self.num_days_iter = num_days_iter
        self.df = self._load_df(test)
        self.action_space = np.linspace(-1, 1, num_actions)
        self.today = 0 if today is None else today
        self._data_valid()

    def _load_df(self, test):
        if test:
            ticker_data = self._load_test_df()
        else:
            ticker_data = pd.read_csv(f'iexfinance/iexdata/{self.ticker}')
            ticker_data = ticker_data[ticker_data.date >= self.start_date]
            ticker_data.reset_index(inplace=True)
            # This is really cheating but...
            # This part should become a new function eventually
            ticker_data = ticker_data.drop('date', axis=1).pct_change().shift(-1)
            # It would be better if dates are saved as a pd.Series?

        zeros = pd.DataFrame(np.zeros((len(ticker_data), 2)),
                             columns=['position', 'pnl'])

        # It's probably better to transpose, then let columns be dates, but wtf...
        df = pd.concat([ticker_data, zeros], axis=1) # .set_index('date')
        df.drop('index', axis=1, inplace=True)
        return df

    def _load_test_df(self):
        date_col = [datetime.date.today() + datetime.timedelta(days=i)
                    for i in range(self.num_days_iter)]
        temp_df = pd.DataFrame(np.ones((self.num_days_iter, 6)),
                               columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        temp_df.iloc[:, 0] = date_col
        return temp_df

    def _data_valid(self):
        assert len(self.df) >= self.num_days_iter, \
                f'DataFrame shape: {self.df.shape}, num_days_iter: {self.num_days_iter}'

    def get_state(self, delta_t=0):
        return self.df.iloc[self.today + delta_t, -4:]

    # 1. Reward is tricky
    # 2. Should invalid action be penalized?
    def step(self, action):
        if not self.done():
            # Record pnl
            # This implementation of reward is such a hogwash!!
            #     but recall, Deepmind Atari 2600 solution does something similar...
            #     assigning credit is always hard...
            # Pandas complain here, "A value is trying to be set on a copy of a slice from a DataFrame"
            #     but the suggested solution is actually misleading... so leaving it as is
            pd.set_option('mode.chained_assignment', None)
            self.df.pnl[self.today] = reward = 0.0 if self.today == 0 else \
                                                 self.df.position[self.today - 1] * self.df.close[self.today]
            #                                    np.log(self.df.close[self.today] /
            #                                           self.df.close[self.today - 1]) * self.df.position[self.today-1]

            # Think about accumulating for the scores...

            # Record position
            # Feel like we should force the action to be valid...
            #     Otherwise, the action-reward function becomes too complex for
            #     the network to learn.
            if self.valid_action(action):
                new_position = self.action_space[action]
                self.df.position[self.today] = self.df.position[self.today-1] + new_position \
                                               if self.today != 0 else new_position

            else:
                self.df.position[self.today] = self.df.position[self.today - 1]
                reward = -10.0

            self.today += 1
            # Think about how to re-allocate the reward
            return reward, False
        else:
            return 0.0, True

    def valid_action(self, action):
        if self.today == 0: return True
        current_position = self.df.position[self.today - 1]
        return -1.0 <= current_position + self.action_space[action] <= 1.0

    def reset(self):
        self.today = 0
        # take care of df position & pnl
        self.df.position = self.df.pnl = 0.0

    # NOT THE MOST EFFICIENT...
    def done(self):
        return self.today > self.num_days_iter


def iterable(arg):
    return (isinstance(arg, collections.Iterable) and
            not isinstance(arg, six.string_types))


class Engine:
    def __init__(self, tickers, start_date, num_days_iter,
                 today=None, seed=None):
        if seed: np.random.seed(seed)
        if not iterable(tickers): tickers = [tickers]
        self.tickers = self._get_tickers_objs(tickers, start_date, num_days_iter, today)
        self.reset_game()

    def reset_game(self):
        self.score, self.done = 0.0, False
        list(map(lambda ticker: ticker.reset(), self.tickers))

    @staticmethod
    def _get_tickers_objs(tickers, start_date, num_days_iter, today):
        return [Ticker(ticker, start_date, num_days_iter, today) for ticker in tickers]

    # return state
    def get_state(self):
        return list(map(lambda ticker: ticker.get_state(), self.tickers))

    def moves_available(self):
        raise NotImplementedError

    # take a step
    def step(self, actions):
        if not iterable(actions): actions = [actions]
        assert len(self.tickers) == len(actions)

        rewards, dones = zip(*(itertools.starmap(lambda ticker, action: ticker.step(action),
                                                 zip(self.tickers, actions))))

        self.score = functools.reduce(lambda x, y: x + y, rewards, 0)
        self.done = functools.reduce(lambda x, y: x | y, dones, False)

        return self.score, self.done

    def __repr__(self):
        tickers = [f'ticker_{i}:{ticker.ticker}, ' for i, ticker in enumerate(self.tickers)]
        return str(tickers)


