import collections
import datetime
import itertools
from itertools import count
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six

plt.ion()


class Ticker:
    def __init__(self, ticker, start_date, num_days_iter,
                 today=None, num_actions=3, test=False,
                 action_space_min=-1.0, action_space_max=1.0):
        self.ticker = str.upper(ticker)
        self.start_date = start_date
        self.num_days_iter = num_days_iter
        self.df, self.dates = self._load_df(test)
        self.action_space = np.linspace(action_space_min, action_space_max, num_actions)
        self.today = 0 if today is None else today
        self._data_valid()
        self.current_position = self.accumulated_pnl = 0.0

    def _load_df(self, test):
        if test:
            ticker_data = self._load_test_df()
        else:
            ticker_data = pd.read_csv(f'iexfinance/iexdata/{self.ticker}')
            ticker_data = ticker_data[ticker_data.date >= self.start_date]

        ticker_data.reset_index(inplace=True)
        # This is really cheating but...
        dates_series = ticker_data['date']
        ticker_data.drop('date', axis=1, inplace=True)
        # This part should become a function eventually
        ticker_data_delta = ticker_data.pct_change()
        add_str_delta = lambda x: x + '_delta'
        ticker_data_delta.rename(add_str_delta, axis='columns', inplace=True)
        ticker_data_delta.iloc[0, :] = 0.0

        zeros = pd.DataFrame(np.zeros((len(ticker_data), 2)),
                             columns=['position', 'pnl'])

        # It's probably better to transpose, then let columns be dates, but wtf...
        df = pd.concat([ticker_data, ticker_data_delta, zeros], axis=1)
        df.drop(['index', 'index_delta'], axis=1, inplace=True)

        return df, dates_series

    def _load_test_df(self):
        date_col = [datetime.date.today() + datetime.timedelta(days=i)
                    for i in range(self.num_days_iter)]
        aranged_values = [np.repeat(i, 6) for i in range(1, self.num_days_iter+1)]
        temp_df = pd.DataFrame(aranged_values,
                               columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        temp_df.iloc[:, 0] = date_col
        return temp_df

    def _data_valid(self):
        assert len(self.df) >= self.num_days_iter, \
                f'DataFrame shape: {self.df.shape}, num_days_iter: {self.num_days_iter}'
        assert len(self.df) == len(self.dates), \
                f'df.shape: {self.df.shape}, dates.shape:{self.dates.shape}'

    def get_state(self, delta_t=0):
        today_market_data_position = np.array(self.df.iloc[self.today+delta_t, -4:-2])
        today_market_data_position[-1] = self.current_position
        return today_market_data_position

    # 1. Reward is tricky
    # 2. Should invalid action be penalized?
    def step(self, action):
        if not self.done():
            # Record pnl
            # This implementation of reward is such a hogwash!!
            #     but recall, Deepmind's DQN solution does something similar...
            #     assigning credit is always hard...
            # Pandas complain here, "A value is trying to be set on a copy of a slice from a DataFrame"
            #     but the suggested solution is actually misleading... so leaving it as is
            pd.set_option('mode.chained_assignment', None)
            self.df.pnl[self.today] = reward = 0.0 if self.today == 0 else \
                                               self.current_position * self.df.close_delta[self.today]

            # Think about accumulating the scores...
            self.accumulated_pnl += reward

            self.df.position[self.today] = self.current_position = self.action_space[action]

            # new_position_delta = self.action_space[action] if self.today == 0 or \
            #                                                   self.valid_action(action) else 0.0
            # self.current_position = self.df.position[self.today] = \
            #                                 new_position_delta if self.today == 0 else \
            #                                 self.df.position[self.today-1] + new_position_delta

            # If we penalize invalid moves, the model learns to avoid it, but this deters
            #     learning experiences. It only focuses on position such that it avoids penalty.
            #     Penalty of -0.1 is certainly too high for this matter
            #
            # if self.valid_action(action):
            #     new_position_delta = self.action_space[action]
            #     self.current_position = self.df.position[self.today] = \
            #                                     new_position_delta if self.today == 0 else \
            #                                     self.df.position[self.today-1] + new_position_delta
            #
            # else:
            #     self.current_position = self.df.position[self.today] = self.df.position[self.today-1]
            #     reward = -0.1 # This is tricky, should consider the distribution of the returns

            self.today += 1
            # Think about how to re-allocate the reward
            return reward, False
        else:
            self.current_position = 0.0
            return 0.0, True

    def valid_action(self, action):
        if self.today == 0: return True
        # current_position = self.df.position[self.today-1]
        # return -1.0 <= current_position + self.action_space[action] <= 1.0
        # The above approach causes shitty troubles...

    def reset(self):
        self.today = 0
        self.df.position = self.df.pnl = 0.0
        self.current_position = self.accumulated_pnl = 0.0

    # NOT THE MOST EFFICIENT...
    def done(self):
        return self.today > self.num_days_iter

    def render(self, axis):
        market_data, position = self.get_state()
        # axis[0].scatter(self.today, self.df.pnl[self.today-1])
        axis[0].set_ylabel(f'Daily price: {self.ticker}')
        axis[0].set_xlabel('Time step')
        axis[0].plot(np.arange(self.today), self.df.close[:self.today])
        # axis[1].scatter(self.today, position)
        # axis[2].scatter(self.today, self.accumulated_pnl)
        axis[1].set_ylabel(f'Daily return from Agent')
        axis[1].set_xlabel('Time step')
        axis[1].scatter(self.today, self.df.pnl[self.today-1])
        plt.pause(0.001)


def iterable(arg):
    return (isinstance(arg, collections.Iterable) and
            not isinstance(arg, six.string_types))


class Engine:
    def __init__(self, tickers, start_date, num_days_iter,
                 today=None, seed=None, num_action_space=3, render=False, *args, **kwargs):
        if seed: np.random.seed(seed)
        if not iterable(tickers): tickers = [tickers]
        self.tickers = self._get_tickers(tickers, start_date, num_days_iter,
                                         today, num_action_space, *args, **kwargs)
        self.reset_game()

        if render:
            # Somehow ax_list should be grouped in two always...
            # Or is there another way of getting one axis per row and then add?
            fig_height = 3 * len(self.tickers)
            self.fig, self.ax_list = plt.subplots(len(tickers), 2, figsize=(10, fig_height))

    def reset_game(self):
        list(map(lambda ticker: ticker.reset(), self.tickers))

    def _get_tickers(self, tickers, start_date, num_days_iter,
                     today, num_action_space, *args, **kwargs):
        return [Ticker(ticker, start_date, num_days_iter, today, num_action_space, *args, **kwargs)
                for ticker in tickers]

    def _render(self, render):
        if render:
            if len(self.tickers) == 1:
                self.tickers[0].render(self.ax_list)
            else:
                for axis, ticker in zip(self.ax_list, self.tickers):
                    ticker.render(axis)

    def get_state(self, delta_t=0):
        # Note: np.arary(...) could also be used
        return list(map(lambda ticker: ticker.get_state(delta_t), self.tickers))

    def moves_available(self):
        raise NotImplementedError

    def step(self, actions):
        if not iterable(actions): actions = [actions]
        assert len(self.tickers) == len(actions), f'{len(self.tickers)}, {len(actions)}'

        rewards, dones = zip(*(itertools.starmap(lambda ticker, action: ticker.step(action),
                                                 zip(self.tickers, actions))))

        # This is somewhat misleading
        score = functools.reduce(lambda x, y: x + y, rewards, 0.0)
        done = functools.reduce(lambda x, y: x | y, dones, False)

        return score, done

    def render(self):
        # This is possibly unnecessary b/c of changes
        self._render(True)

    def __repr__(self):
        tickers = [f'ticker_{i}: {ticker.ticker}, ' for i, ticker in enumerate(self.tickers)]
        return str(tickers)

    def _data_valid(self):
        raise NotImplementedError


class Portfolio(Engine):
    def __init__(self, tickers, start_date, num_days_iter,
                 today=None, seed=None, render=False,
                 action_space_min=0.0, action_space_max=1.0):
        num_action_space = len(tickers)
        super().__init__(tickers, start_date, num_days_iter, today, seed, num_action_space, render,
                         action_space_min=action_space_min, action_space_max=action_space_max)
        self.action_space = np.linspace(action_space_min, action_space_max, num_action_space)
        self.position_df = self._get_position_df(tickers, num_action_space, action_space_min, action_space_max)

    # Returns a dataframe of all the possible position distributions
    # positions are divided into increments inversely proportional to the length of tickers
    def _get_position_df(self, tickers, num_action_spaces,
                         action_space_min, action_space_max):

        def all_possible_combinations(values, increase, proportion_remaining, idx, ret_list):
            if idx == len(values) - 1:
                values[idx] = max(0, proportion_remaining)
                ret_list.append(list(values))
                values[idx] = 0
                return
            for t in count():
                add_to_this_idx = t * increase
                if add_to_this_idx <= proportion_remaining:
                    values[idx] = add_to_this_idx
                    all_possible_combinations(values, increase,
                                              proportion_remaining - add_to_this_idx, idx + 1, ret_list)
                    values[idx] = 0.0
                else:
                    break

        ticker_length = len(tickers)
        inc = (action_space_max - action_space_min) / (num_action_spaces - 1)
        so_far = [0.0 for _ in range(ticker_length)]
        # Total positions sum to 1
        ret, remain = [], 1.0
        all_possible_combinations(so_far, inc, remain, 0, ret)
        return pd.DataFrame(ret).T

    def moves_available(self):
        return self.position_df.shape[1]

    def step(self, action_index):
        assert 0 <= action_index < self.moves_available(), \
            f'action_index: {action_index}, moves_avail: {self.moves_available()}'

        positions = self.position_df[action_index]
        actions = [np.argwhere(self.action_space == position).item()
                   for position in positions]
        return super(Portfolio, self).step(actions)


