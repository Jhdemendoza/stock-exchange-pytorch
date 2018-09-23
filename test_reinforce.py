import argparse
import gym
import gym_stock_exchange

from reinforcement.run_exchange import RunExchange

import torch

from reinforcement.models_dqn import DuelingDQN


parser = argparse.ArgumentParser(description='Hyper-parameters for the DQN training')
parser.add_argument('--epsilon',              default=1.0, type=float)
parser.add_argument('--min_epsilon',          default=0.05, type=float)
parser.add_argument('--update_every',         default=0, type=int)
parser.add_argument('--log_every',            default=2, type=int)
parser.add_argument('--n_train',              default=0, type=int)
parser.add_argument('--n_test',               default=200, type=int)
parser.add_argument('--batch_size',           default=32, type=int)
parser.add_argument('--gamma',                default=0.0, type=float)
parser.add_argument('--learning_rate',        default=1e-7, type=float)
parser.add_argument('--mode',                 default='test', type=str)
parser.add_argument('--num_action_space',     default=3, type=int)
parser.add_argument('--num_running_days',     default=40, type=int)

args = parser.parse_args()

if __name__ == '__main__':

    policy_q, target_q = DuelingDQN(args.num_running_days, args.num_action_space).cuda(), \
                         DuelingDQN(args.num_running_days, args.num_action_space).cuda()

    try:
        policy_q.load_state_dict(torch.load('my_duel_policy_vanilla.pt'))
        target_q.load_state_dict(torch.load('my_duel_target_vanilla.pt'))
    except FileNotFoundError:
        print('--- Exception Raised: Files not found...')

    optimizer = replay_memory = None

    render = args.mode == 'test'
    env = gym.make('game-stock-exchange-v0')
    env.create_engine('aapl', '2014-01-01', 1000,
                      num_action_space=args.num_action_space, render=render)

    player = RunExchange(env, replay_memory, policy_q, target_q, optimizer, args.num_running_days,
                         args.batch_size, args.epsilon,
                         args.min_epsilon,
                         args.n_train, args.update_every, args.log_every,
                         gamma=args.gamma, mode=args.mode)

    try:
        player.test_exchange(args.n_test, args.num_action_space//2)
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
    finally:
        if args.mode == 'train':
            print('Saving...')
            torch.save(policy_q.state_dict(), 'my_duel_policy_vanilla.pt')
            torch.save(target_q.state_dict(), 'my_duel_target_vanilla.pt')
