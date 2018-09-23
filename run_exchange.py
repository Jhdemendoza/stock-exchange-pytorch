from copy import deepcopy
import datetime
from collections import Counter
from itertools import count
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from reinforcement import train_dqn
import math

try:
    os.makedirs('logs')
except OSError:
    print('--- log folder exists')

FILE_NAME = 'logs/training_{}.log'.format('_'.join(str(datetime.datetime.now()).split(' ')))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(FILE_NAME)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Refactor name...
class RunExchange:
    def __init__(self, env, replay_memory, policy, target, optimizer, num_running_days,
                 batch_size=32, epsilon=1.0, min_epsilon=0.1,
                 n_train=1000, update_every=100, log_every=10,
                 gamma=0.999, double_dqn=True, mode='train'):
        self.env = env
        self.replay_memory = replay_memory
        self.policy = policy
        self.target = target
        self.optimizer = optimizer
        self.num_running_days = num_running_days
        self.batch_size = batch_size
        self.max_epsilon = self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.n_train = n_train
        self.update_every = update_every
        self.log_every = log_every
        self.gamma = gamma
        self.double_dqn = double_dqn
        self.rewards = []
        self.losses = []
        self.mode = mode

        # if mode == 'train':
        self.fig, self.axis = self.get_figure_and_axis()

    @classmethod
    def get_figure_and_axis(cls):
        plt.style.use(['ggplot']) # 'fivethirtyeight'])
        plt.ion()
        return plt.subplots(2, 1)

    def get_running_state(self):
        return np.zeros(self.num_running_days) # .tolist()

    def add_new_state(self, running_state_orig, new_state_to_add):
        if isinstance(new_state_to_add, list):
            new_state_to_add = new_state_to_add[0]
        running_state = pd.Series(running_state_orig).shift(-1)
        # Assign new price to index == last_elem - 1
        running_state.iloc[-1] = new_state_to_add.item(0)
        # running_state.iloc[-2] = new_state_to_add.item(0)
        # Assign new position to index == last_elem
        # running_state.iloc[-1] = new_state_to_add.item(1)
        assert len(running_state_orig) == len(running_state)
        return running_state.tolist()

    # Refactor !!
    def test_exchange(self, testing_interval, no_action_index):

        running_state = self.add_new_state(self.get_running_state(), self.env.reset())

        for _ in range(self.num_running_days - 1):
            next_state, reward, done, _ = self.env.step(no_action_index)
            running_state = self.add_new_state(running_state, next_state)
            assert reward == 0, f'Reward is somehow {reward}'
            # assert running_state[-1] == 0.0, f'Position is somehow {running_state[-1]}'

        episode_rewards = []
        actions = []

        for _ in range(testing_interval):
            action = self.policy.act(running_state, 0.0)

            next_state, reward, done, _ = self.env.step(action)
            env.render()

            # update running_state for the next use
            running_state = self.add_new_state(running_state, next_state)
            episode_rewards += [reward]
            actions += [action]

        return episode_rewards, actions

    # Refactor the damn name...
    def run_exchange(self):

        def adjust_epsilon():
            MULTIPLIER = 5.0
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
                                                math.exp(-1.0 * i_episode * MULTIPLIER / self.n_train)

        def log_(episode_loss_avg, episode_reward, action_counters):
            logger.info('---------------------')
            logger.info('Ending {} episodes, epsilon: {:.5f}'.format(i_episode, self.epsilon))
            logger.info('Episode Loss               : {:.5f}'.format(episode_loss_avg))
            logger.info('Episode Rewards            : {:.5f}'.format(episode_reward))
            logger.info('Actions Counted:           : {}'.format(action_counters))

            self.axis[0].plot(self.rewards)
            # self.axis[0].scatter(len(self.rewards), self.rewards[-1])
            # sns.distplot(self.rewards, ax=self.axis[1])

            self.axis[1].plot(self.losses)
            # self.axis[2].scatter(len(self.losses), self.losses[-1])
            # sns.distplot(self.losses, ax=self.axis[3])
            plt.pause(0.001)

        if self.mode == 'test':
            self.epsilon = self.min_epsilon = 1e-7

        for i_episode in range(1, self.n_train + 1):

            # Can be just a number, but let's keep it for now...
            episode_loss = []
            episode_reward = 0.0
            actions = Counter()

            running_state = self.add_new_state(self.get_running_state(), self.env.reset())

            for _ in count(1):

                adjust_epsilon()

                action = self.policy.act(running_state, self.epsilon)
                actions[action] += 1

                next_state, reward, done, info = self.env.step(action)

                # copy t_0 so that we don't change t_1 variable
                running_state_0 = deepcopy(running_state)
                running_state = self.add_new_state(running_state, next_state)

                # Error Check!
                # make it a series then del?
                assert not pd.Series(running_state).hasnans, pd.Series(running_state).isnull()
                assert running_state_0[1:-1] == running_state[:-2], \
                        '{} vs {}'.format(running_state_0[1:], running_state[:-1])

                self.replay_memory.push(running_state_0, action, running_state, reward)
                episode_reward += reward

                if self.mode == 'train':
                    loss = train_dqn(self.policy, self.target, self.replay_memory,
                                     self.batch_size, self.optimizer, self.gamma, self.double_dqn)
                    if loss is not None:
                        episode_loss += [loss.item()]

                if done:
                    self.rewards += [episode_reward]
                    self.losses += [np.mean(episode_loss)]

                    if i_episode % self.log_every == 0:
                        log_(np.mean(episode_loss), episode_reward, actions)

                    if i_episode % self.update_every == 0:
                        self.target.load_state_dict(self.policy.state_dict())

                    del episode_loss
                    del running_state

                    break


