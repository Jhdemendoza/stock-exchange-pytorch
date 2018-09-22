import numpy as np
import torch
import torch.nn as nn
from utils.environment import device

FEATURES = 128


class DuelingDQN(nn.Module):
    # Eventually, n_tickers should create n number of values and advantages
    def __init__(self, n_input_features, n_action_space, n_tickers=None):
        super(DuelingDQN, self).__init__()
        self.n_action_space = n_action_space
        self.feature = nn.Sequential(
            nn.Linear(n_input_features, FEATURES),
            nn.ReLU(),
            nn.Linear(FEATURES, FEATURES),
            nn.ReLU(),
            nn.Linear(FEATURES, FEATURES),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(FEATURES, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(FEATURES, 32),
            nn.ReLU(),
            nn.Linear(32, n_action_space)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()

    def predict(self, x):
        return self.forward(x).detach().sort(dim=1, descending=True)

    def act(self, x, epsilon):
        if not torch.is_tensor(x):
            x = torch.tensor([x], dtype=torch.float32, device=device)

        if np.random.rand() > epsilon:
            x = self.feature(x).detach()
            # print(self.advantage(x).detach().sort(dim=1, descending=True))
            x = self.advantage(x).detach().argmax().item()
            return x
        else:
            return np.random.randint(self.n_action_space)