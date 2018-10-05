import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reinforcement import device
from gym_exchange.gym_engine import iterable


from torch.optim import Adam
from reinforcement.utils import Update, OUNoise


class Actor(nn.Module):
    def __init__(self, num_input, num_hidden, num_action_space, init_w=3e-3):
        super(Actor, self).__init__()

        self.s0 = None
        if iterable(num_input):
            len_num_input = len(num_input)
            if len_num_input == 2:
                self.s0 = nn.LSTM(num_input[1], num_hidden)
            elif len_num_input == 1:
                num_input = num_input[0]

        if self.s0 is None:
            self.s1 = nn.Sequential(
                nn.Linear(num_input, num_hidden),
                nn.LayerNorm(num_hidden),
                nn.ReLU(inplace=True))

        self.s2 = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.LayerNorm(num_hidden),
            nn.ReLU(inplace=True))
        self.out = nn.Linear(num_hidden, num_action_space)
        # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py
        # We are doing tanh afterall!
        self.out.weight.data.mul_(0.1)
        self.out.bias.data.mul_(0.1)
        # https://github.com/pemami4911/awesome-hyperparams
        # weight uses uniform for low dim inputs...

    def forward(self, x):
        if self.s0:
            x = self.s0(x)[0][:, -1, :]
        else:
            x = self.s1(x)
        x = self.s2(x)
        return F.tanh(self.out(x))

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state).detach()
        return action.cpu().numpy()[0]


class Critic(nn.Module):
    def __init__(self, num_input, num_hidden, num_action_space, init_w=3e-3):
        super(Critic, self).__init__()

        self.s0 = None
        if iterable(num_input):
            len_num_input = len(num_input)
            if len_num_input == 2:
                self.s0 = nn.LSTM(num_input[1], num_hidden)
                # Taking leeway
                #    Note: num_input is used in s1, so doing this...
                num_input = num_hidden
            if len_num_input == 1:
                num_input = num_input[0]

        self.s1 = nn.Sequential(
            nn.Linear(num_input + num_action_space, num_hidden),
            nn.LayerNorm(num_hidden),
            nn.ReLU(inplace=True))

        self.s2 = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.LayerNorm(num_hidden),
            nn.ReLU(inplace=True))

        self.value = nn.Linear(num_hidden, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.mul_(0.1)

    def forward(self, state, action):
        if self.s0:
            x = self.s0(state)[0][:, -1, :]
        else:
            x = state
        x = torch.cat((x, action), 1)
        x = self.s1(x)
        x = self.s2(x)
        return self.value(x)


class DDPG(nn.Module):
    def __init__(self, num_input, num_hidden, num_action_space, gym_env, args):
        super(DDPG, self).__init__()
        self.args = args

        self.actor = Actor(num_input, num_hidden, num_action_space)
        self.actor_target = Actor(num_input, num_hidden, num_action_space)
        Update.hard_update(self.actor, self.actor_target)

        self.critic = Critic(num_input, num_hidden, num_action_space)
        self.critic_target = Critic(num_input, num_hidden, num_action_space)
        Update.hard_update(self.critic, self.critic_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=args.actor_learning_rate)
        self.optim_critic = Adam(self.critic.parameters(), lr=args.critic_learning_rate)

        # add options through args at some point
        self.noise = OUNoise(gym_env.action_space)

        self.value_loss_fn = nn.MSELoss()

    def select_action(self, state, t=0):
        self.actor.eval()
        # should detach from here...
        action = self.actor.select_action(state)
        action += self.get_noise(t)
        self.actor.train()
        return np.clip(action, -1.0, 1.0)

    def reset_noise(self):
        self.noise.reset()

    def get_noise(self, t):
        return self.noise.get_noise(t)

    def get_policy_loss(self, state):
        action = self.actor(state)
        return -self.critic(state, action).mean()

    def get_value_loss(self, state, action, reward, next_state, done):
        next_action = self.actor_target(next_state).detach()
        target_value = self.critic_target(next_state, next_action).detach()

        expected_value = reward + (1.0 - done) * target_value * self.args.gamma

        pred_value = self.critic(state, action)

        value_loss = self.value_loss_fn(pred_value, expected_value)
        return value_loss

    def forward(self, x):
        raise NotImplementedError

    def update(self, value_loss, policy_loss):
        self.optim_critic.zero_grad()
        value_loss.backward()
        self.optim_critic.step()

        self.optim_actor.zero_grad()
        policy_loss.backward()
        self.optim_actor.step()

        Update.soft_update(self.critic, self.critic_target, self.args.tau)
        Update.soft_update(self.actor, self.actor_target, self.args.tau)
