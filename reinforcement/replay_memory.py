from collections import deque, namedtuple
import numpy as np
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

TransitionDone = namedtuple('TransitionDone',
                            ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, max_length):
        self.max_length = max_length
        self.memory = deque()

    def push(self, *args):
        while len(self.memory) >= self.max_length:
            self.memory.popleft()
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __repr__(self):
        if len(self) == 0:
            return 'ReplayMemory.memory: EMPTY'
        return 'ReplayMemory.memory: {}...'.format(self.memory[0])

    def __len__(self):
        return len(self.memory)


class ReplayMemoryWithDone(ReplayMemory):
    def push(self, *args):
        while len(self.memory) >= self.max_length:
            self.memory.popleft()
        self.memory.append(TransitionDone(*args))

    def __repr__(self):
        if len(self) == 0:
            return 'ReplayMemory.memory: EMPTY'
        return 'ReplayMemory.memory: {}...'.format(self.memory[0])


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)