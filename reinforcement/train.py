import numpy as np
import torch
import torch.nn.functional as F

from reinforcement.replay_memory import Transition, TransitionDone
from reinforcement.environment import device


def load_game_from_replay_memory(replay_memory, batch_size, done_flag=False):
    transitions = replay_memory.sample(batch_size)
    if done_flag:
        batch = TransitionDone(*zip(*transitions))
    else:
        batch = Transition(*zip(*transitions))
    return batch


def batch_to_tensor(given_batch, action_batch=False):
    dtype = torch.long if action_batch else torch.float32
    batch = list(map(lambda x: torch.tensor(x, device=device, dtype=dtype
                                            ).unsqueeze(0), given_batch))
    return torch.cat(batch, 0)


def train_dqn(policy_q, target_q, replay_memory, batch_size,
              optimizer, gamma, double_dqn):

    # Keep replay_memory length large enough to sample from...
    if len(replay_memory) < batch_size * 30:
        return

    batch = load_game_from_replay_memory(replay_memory, batch_size)

    state_batch = batch_to_tensor(batch.state)
    action_batch = batch_to_tensor(batch.action, action_batch=True)
    reward_batch = batch_to_tensor(batch.reward).unsqueeze(1)
    next_state_batch = batch_to_tensor(batch.next_state)

    state_action_values = policy_q(state_batch).gather(1, action_batch.unsqueeze(1))

    if double_dqn:
        next_state_action = policy_q(next_state_batch).detach().max(1)[1].unsqueeze(1)
        next_state_values_action_unspecified = target_q(next_state_batch).detach()
        next_state_values = next_state_values_action_unspecified.gather(1, next_state_action)
    else:
        next_state_values = target_q(next_state_batch).max(1)[0].detach().unsqueeze(1)

    expected_state_action_values = next_state_values * gamma + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_q.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


# Wonder if I should wrap this into a class...
def train_ddpg(ddpg_agent, replay_buffer, batch_size):
    if len(replay_buffer) < batch_size * 10:
        return

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    value_loss = ddpg_agent.get_value_loss(state, action, reward, next_state, done)
    policy_loss = ddpg_agent.get_policy_loss(state)

    # This can't be the best way...
    # Need to think about what's the best design...
    ddpg_agent.update(value_loss, policy_loss)

    return value_loss, policy_loss