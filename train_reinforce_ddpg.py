import argparse
import gym
import gym_exchange

from reinforcement.run_exchange import RunExchangeContinuous

from reinforcement.models_ddpg import DDPG
from reinforcement import ReplayBuffer


parser = argparse.ArgumentParser(description='Hyper-parameters for DDPG training')
parser.add_argument('--update_every',         default=20, type=int)
parser.add_argument('--log_every',            default=2, type=int)
parser.add_argument('--n_train',              default=200, type=int)
parser.add_argument('--batch_size',           default=128, type=int)
parser.add_argument('--replay_buffer_length', default=100000, type=int)
parser.add_argument('--actor_learning_rate',  default=1e-4, type=float)
parser.add_argument('--critic_learning_rate', default=1e-3, type=float)
parser.add_argument('--mode',                 default='train', type=str, choices=['train', 'test'])
parser.add_argument('--hidden_dim',           default=256, type=int)
parser.add_argument('--num_running_days',     default=20, type=int)
parser.add_argument('--gamma',                default=0.99, type=float)
parser.add_argument('--tau',                  default=1e-4, type=float)


args = parser.parse_args()


if __name__ == '__main__':

    assert args.mode == 'train', '--- Currently not supported. Use test_reinforce.py instead ---'

    env = gym.make('game-stock-exchange-continuous-v0')
    # env = gym.make('Pendulum-v0')
    # env = gym.make('MountainCarContinuous-v0')

    ddpg = DDPG(env.observation_space.shape, args.hidden_dim,
                env.action_space.shape[0], env, args).cuda()

    rb = ReplayBuffer(args.replay_buffer_length)

    player = RunExchangeContinuous(env, rb, ddpg, args.num_running_days,
                                   args.batch_size, args.n_train,
                                   args.update_every, args.log_every,
                                   args.mode)

    try:
        player.train_exchange_ddpg()
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
    finally:
        # This is now unnecessary...
        if args.mode == 'train':
            print('Saving...')
            # torch.save(policy_q.state_dict(), 'my_duel_policy_vanilla.pt')
            # torch.save(target_q.state_dict(), 'my_duel_target_vanilla.pt')
