import gym
import numpy as np


# Modified, originally from
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, gym_env_action_space, mu=0.0, theta=0.15,
                 max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        # Discrete, We need gym_env.action_space.n
        # if isinstance(gym_env_action_space, gym.spaces.Discrete):
        #     self.action_dim = gym_env_action_space.n
        # else:
        #     self.action_dim = gym_env_action_space.shape[0]
        self.action_dim = gym_env_action_space.shape[0]
        # I think low, high should be -1, 1 if normalized in our manner
        # Something to think about ...
        self.low = gym_env_action_space.low
        self.high = gym_env_action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def get_noise(self, t=0):
        # update sigma
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) \
                     * min(1.0, t / self.decay_period)
        x = self.state
        dx = self.theta * (self.mu - x) \
             + self.sigma * np.random.randn(self.action_dim)

        # self.state == noise
        self.state = x + dx
        return np.float32(self.state)


class Update:
    @classmethod
    def soft_update(cls, source, target, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau)
                                    + source_param.data * tau)
    @classmethod
    def hard_update(cls, source, target):
        target.load_state_dict(source.state_dict())


class NormalizedActions(gym.ActionWrapper):
    def lb_ub(self):
        return self.action_space.low, self.action_space.high

    def action(self, action):
        lb, ub = self.lb_ub()
        scaled_action = lb + (action + 1.0) * (ub - lb) / 2
        return np.clip(scaled_action, lb, ub)

    def reverse_action(self, scaled_action):
        lb, ub = self.lb_ub()
        action = 2 * (scaled_action - lb) / (ub - lb) - 1
        return np.clip(action, lb, ub)