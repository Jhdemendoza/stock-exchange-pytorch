import torch
import torch.nn.functional as F

from reinforcement.replay_memory import Transition
from reinforcement.environment import device


def train_dqn(policy_q, target_q, replay_memory, batch_size,
              optimizer, gamma, double_dqn):

    if len(replay_memory) < batch_size * 30:
        return

    def load_game_from_replay_memory():
        transitions = replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def batch_to_tensor(given_batch, action_batch=False):
        dtype = torch.long if action_batch else torch.float32
        batch = list(map(lambda x: torch.tensor(x, device=device, dtype=dtype
                                                ).unsqueeze(0), given_batch))
        return torch.cat(batch, 0)

    batch = load_game_from_replay_memory()

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
