import argparse
import gym
import gym_exchange
import pickle

from reinforcement.run_exchange import RunExchange

import torch
import torch.optim as optim

from reinforcement.models_dqn import DuelingDQN
from reinforcement import ReplayMemory


parser = argparse.ArgumentParser(description='Hyper-parameters for the DQN training')
parser.add_argument('--epsilon',              default=1.0, type=float)
parser.add_argument('--min_epsilon',          default=0.05, type=float)
parser.add_argument('--update_every',         default=20, type=int)
parser.add_argument('--log_every',            default=2, type=int)
parser.add_argument('--n_train',              default=200, type=int)
parser.add_argument('--batch_size',           default=32, type=int)
parser.add_argument('--gamma',                default=0.9, type=float)
parser.add_argument('--replay_memory',        default='replay_memory.p', type=str)
parser.add_argument('--replay_memory_length', default=100000, type=int)
parser.add_argument('--learning_rate',        default=1e-7, type=float)
parser.add_argument('--mode',                 default='train', type=str, choices=['train', 'test'])

# num_action_space not TRUE
parser.add_argument('--num_action_space',     default=3, type=int)
parser.add_argument('--num_running_days',     default=20, type=int)

args = parser.parse_args()

if __name__ == '__main__':

    assert args.mode == 'train', '--- Currently not supported. Use test_reinforce_dqn.py instead ---'

    env = gym.make('game-stock-exchange-v0')

    args.num_action_space = env.moves_available()
    args.n_input_features = 52

    policy_q, target_q = DuelingDQN(args.n_input_features, args.num_action_space).cuda(), \
                         DuelingDQN(args.n_input_features, args.num_action_space).cuda()

    try:
        policy_q.load_state_dict(torch.load('my_duel_policy_vanilla.pt'))
        target_q.load_state_dict(torch.load('my_duel_target_vanilla.pt'))
    except FileNotFoundError:
        print('--- Exception Raised: Files for model states not found...')

    # This is not supported for this version
    # Just leaving as is for now ...
    try:
        rm = pickle.load(open(args.replay_memory, 'rb'))
    except FileNotFoundError:
        rm = ReplayMemory(args.replay_memory_length)

    optimizer = optim.RMSprop(policy_q.parameters(), eps=args.learning_rate)


    player = RunExchange(env, rm, policy_q, target_q,
                         optimizer, args.num_running_days,
                         args.batch_size, args.epsilon,
                         args.min_epsilon,
                         args.n_train, args.update_every, args.log_every,
                         gamma=args.gamma, mode=args.mode)

    try:
        player.train_exchange_dqn()
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
    finally:
        # This is now unnecessary...
        if args.mode == 'train':
            print('Saving...')
            torch.save(policy_q.state_dict(), 'my_duel_policy_vanilla.pt')
            torch.save(target_q.state_dict(), 'my_duel_target_vanilla.pt')
