from copy import deepcopy
import datetime
from collections import Counter
from itertools import count
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from reinforcement.train import train_dqn, train_ddpg
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

        self.fig, self.axis = self.get_figure_and_axis()

    @classmethod
    def get_figure_and_axis(cls):
        plt.style.use(['ggplot'])  # 'fivethirtyeight'])
        plt.ion()
        return plt.subplots(2, 1)

    # Refactor !!
    def test_exchange(self, testing_interval, no_action_index):

        state = self.env.reset()

        for _ in range(self.num_running_days - 1):
            next_state, reward, done, _ = self.env.step(no_action_index)
            state = next_state

        episode_rewards = []
        actions = []

        for _ in range(testing_interval):
            action = self.policy.act(state, 0.0)

            next_state, reward, done, _ = self.env.step(action)
            self.env.render()

            state = next_state
            episode_rewards += [reward]
            actions += [action]

        return episode_rewards, actions

    # Refactor the name...
    def train_exchange_dqn(self):

        def adjust_epsilon():
            MULTIPLIER = 3.0
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

            state = self.env.reset()

            for _ in count(1):

                adjust_epsilon()

                action = self.policy.act(state, self.epsilon)
                actions[action] += 1

                next_state, reward, done, info = self.env.step(action)

                self.replay_memory.push(state, action, reward, next_state)
                state = next_state

                episode_reward += reward

                if self.mode == 'train':
                    loss = train_dqn(self.policy, self.target, self.replay_memory,
                                     self.batch_size, self.optimizer, self.gamma,
                                     self.double_dqn)
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

                    break


# Refactor name...
class RunExchangeContinuous:
    def __init__(self, env, replay_memory, ddpg, num_running_days,
                 batch_size=32, n_train=1000, update_every=100, log_every=10,
                 mode='train'):
        self.env = env
        self.replay_memory = replay_memory
        self.ddpg_agent = ddpg
        self.num_running_days = num_running_days
        self.batch_size = batch_size
        self.n_train = n_train
        self.update_every = update_every
        self.log_every = log_every
        self.rewards = []
        self.value_losses = []
        self.policy_losses = []
        self.mode = mode

        self.fig, self.axis = self.get_figure_and_axis()

    @classmethod
    def get_figure_and_axis(cls):
        plt.style.use(['ggplot'])  # 'fivethirtyeight'])
        plt.ion()
        return plt.subplots(3, 1)

    # Refactor
    def train_exchange_ddpg(self):

        def log_(episode_value_loss, episode_policy_loss, episode_reward,
                 action_avg, action_std):
            logger.info('---------------------')
            logger.info('Ending {}th episodes,              '.format(i_episode))
            logger.info('Episode Value Loss         : {:.5f}'.format(episode_value_loss))
            logger.info('Episode Policy Loss        : {:.5f}'.format(episode_policy_loss))
            logger.info('Episode Rewards            : {:.5f}'.format(episode_reward))
            logger.info('Episode Action Average     : {:.5f}'.format(action_avg))
            logger.info('Episode Action Stdev       : {:.5f}'.format(action_std))

            self.axis[0].plot(self.rewards)
            # self.axis[0].scatter(len(self.rewards), self.rewards[-1])
            # sns.distplot(self.rewards, ax=self.axis[1])

            self.axis[1].plot(self.value_losses)
            self.axis[2].plot(self.policy_losses)
            # self.axis[2].scatter(len(self.losses), self.losses[-1])
            # sns.distplot(self.losses, ax=self.axis[3])
            plt.pause(1e-4)

        for i_episode in range(1, self.n_train + 1):
            state = self.env.reset()
            self.ddpg_agent.reset_noise()
            episode_reward = 0.0
            episode_value_loss = 0.0
            episode_policy_loss = 0.0
            actions = []

            for step in count(1):
                action = self.ddpg_agent.select_action(state, step)
                actions += [action]

                next_state, reward, done, _ = self.env.step(action)
                self.replay_memory.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                temp_result = train_ddpg(self.ddpg_agent, self.replay_memory, self.batch_size)

                if temp_result is not None:
                    value_loss, policy_loss = temp_result
                    episode_value_loss += value_loss.item()
                    episode_policy_loss += policy_loss.item()

                if done:
                    actions_avg = np.average(actions)
                    actions_std = np.std(actions)
                    if i_episode % self.log_every == 0:
                        log_(episode_value_loss, episode_policy_loss,
                             episode_reward, actions_avg, actions_std)

                    self.rewards += [episode_reward]
                    self.value_losses += [episode_value_loss]
                    self.policy_losses += [episode_policy_loss]
                    break

