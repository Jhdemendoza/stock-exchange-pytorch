import argparse
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from supervised import train_model, Model
from supervised.environment import *


parser = argparse.ArgumentParser(description='Hyper-parameters for the DQN training')
parser.add_argument('--n_train',              default=5000, type=int)
parser.add_argument('--batch_size',           default=32, type=int)
parser.add_argument('--learning_rate',        default=1e-7, type=float)
parser.add_argument('--mode',                 default='train', type=str, choices=['train', 'test'])
parser.add_argument('--num_running_days',     default=40, type=int)
parser.add_argument('--num_discrete_returns', default=10, type=int)

args = parser.parse_args()


if __name__ == '__main__':

    gru_model = Model(1, GRU_OUTPUT_SIZE,
                      args.num_discrete_returns,
                      args.batch_size, 2).double().cuda()
    try:
        gru_model.load_state_dict(torch.load('gru_model.pt'))
    except FileNotFoundError:
        print('--- File Not Found ---')

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = optim.SGD(gru_model.parameters(),
                             lr=args.learning_rate,
                             momentum=0.9, weight_decay=1e-7)

    def sin_lr(x):
        return np.abs(np.sin((x + 0.01) * 0.2))

    exp_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_fn,
                                                   lr_lambda=[sin_lr])

    try:
        train_model(gru_model, loss_fn, optimizer_fn, exp_lr_scheduler, args.n_train)
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
    finally:
        print('Saving...')
        torch.save(gru_model.state_dict(), 'gru_model.pt')
