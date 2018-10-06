import argparse
import functools
import numpy as np
import torch
import torch.nn as nn

from torch import optim
from supervised import train_model_continuous, ContinuousModel
from supervised.train import get_dl
from supervised.dataset import PortfolioData


parser = argparse.ArgumentParser(description='Hyper-parameters for the DQN training')
parser.add_argument('--n_train',              default=5000, type=int)
parser.add_argument('--batch_size',           default=32, type=int)
parser.add_argument('--learning_rate',        default=1e-7, type=float)
parser.add_argument('--mode',                 default='train', type=str, choices=['train', 'test'])
parser.add_argument('--num_running_days',     default=40, type=int)
parser.add_argument('--num_discrete_returns', default=10, type=int)

args = parser.parse_args()

args.tickers = ['aapl', 'googl', 'pg']
args.input_dim = len(args.tickers)

if __name__ == '__main__':

    gru_model = ContinuousModel(args.input_dim,
                                rnn_hidden_size=20,
                                output_size=args.dim_linear_output,
                                num_layers=4,
                                final_output=args.input_dim).cuda()
    try:
        gru_model.load_state_dict(torch.load('gru_model.pt'))
    except FileNotFoundError:
        print('--- File Not Found ---')

    loss_fn = nn.MSELoss()
    optimizer_fn = optim.SGD(gru_model.parameters(),
                             lr=args.learning_rate,
                             momentum=0.9,
                             weight_decay=1e-5)

    def sin_lr(x):
        return np.abs(np.sin((x + 0.01) * 0.2))

    exp_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_fn, lr_lambda=[sin_lr])

    data_loader = functools.partial(get_dl,
                                    tickers=args.tickers,
                                    num_state_space=args.num_running_days,
                                    batch_size=args.batch_size,
                                    DataClass=PortfolioData)

    try:
        train_model_continuous(gru_model,
                               loss_fn,
                               optimizer_fn,
                               exp_lr_scheduler,
                               args.n_train,
                               data_loader)
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
    finally:
        print('Saving...')
        torch.save(gru_model.state_dict(), 'gru_model.pt')
