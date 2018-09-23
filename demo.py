from reinforcement import DuelingDQN
import gym
import gym_stock_exchange
import pandas as pd
import torch
import numpy as np


def get_running_state(num_days):
    return np.zeros(num_days).tolist()


def add_new_state(running_state_orig, new_state_to_add):
    if isinstance(new_state_to_add, list):
        new_state_to_add = new_state_to_add[0]

    running_state = pd.Series(running_state_orig).shift(-1)

    # Assign new price to index == last_elem - 1
    running_state.iloc[-2] = new_state_to_add.item(0)
    # Assign new position to index == last_elem
    running_state.iloc[-1] = new_state_to_add.item(1)

    assert len(running_state_orig) == len(running_state)

    return running_state.tolist()


def test_exchange(env, policy_q, testing_interval=100):
    running_state = get_running_state(NUM_RUNNING_DAYS)

    state = env.reset()
    running_state = add_new_state(running_state, state)

    for _ in range(NUM_RUNNING_DAYS - 1):
        # recall step 2 makes position unchanged given
        # action_space == 5
        next_state, reward, done, _ = env.step(1)
        running_state = add_new_state(running_state, next_state)

        assert reward == 0, \
            f'Reward is somehow {reward}'
        assert running_state[-1] == 0.0, \
            f'Position is somehow {running_state[-1]}'

    episode_rewards = []
    actions = []

    for _ in range(testing_interval):
        action = policy_q.act(running_state, 0.0)

        next_state, reward, done, _ = env.step(action)
        env.render()

        # update running_state for the next use
        running_state = add_new_state(running_state, next_state)
        episode_rewards += [reward]
        actions += [action]

    return episode_rewards, actions


# Actual days will be num_days - 1, plus one spot for position variable
NUM_RUNNING_DAYS = 40

if __name__ == '__main__':

    env = gym.make('game-stock-exchange-v0')
    env.create_engine('aapl', '2014-01-01', 1000, num_action_space=3, render=True)

    policy_q, target_q = DuelingDQN(NUM_RUNNING_DAYS, 3).cuda(), \
                         DuelingDQN(NUM_RUNNING_DAYS, 3).cuda()

    try:
        policy_q.load_state_dict(torch.load('my_duel_policy_vanilla.pt'))
        target_q.load_state_dict(torch.load('my_duel_target_vanilla.pt'))
    except FileNotFoundError:
        print('--- Files Not Found ---')

    rewards, actions = test_exchange(env, policy_q, 1000)
    # print(f'rewards: {rewards}')
    print(f'actions: {actions}')

