import argparse
import functools
import numpy as np
import torch
import torch.nn as nn

from torch import optim
from supervised import train_model_continuous, ContinuousModelBasicConvolution, sin_lr
from supervised.train import get_dl
from supervised.dataset import PortfolioData


parser = argparse.ArgumentParser(description='Hyper-parameters for the DQN training')
parser.add_argument('--n_train',              default=2, type=int)
parser.add_argument('--batch_size',           default=32, type=int)
parser.add_argument('--learning_rate',        default=1e-5, type=float)
parser.add_argument('--mode',                 default='train', type=str, choices=['train', 'test'])
parser.add_argument('--num_running_days',     default=40, type=int)
parser.add_argument('--dim_linear_output',    default=10, type=int)
parser.add_argument('--dim_layers',           default=4, type=int)
parser.add_argument('--dim_conv_hidden_size', default=32, type=int)

args = parser.parse_args()

args.tickers = ['aapl', 'amd', 'msft', 'intc', 'd', 'sbux', 'atvi', 'ibm', 'ual', 'vrsn', 't', 'mcd', 'vz']
args.input_dim = len(args.tickers)
args.conv_kernel_size = 4

if __name__ == '__main__':

    convolution_model = ContinuousModelBasicConvolution(input_shape=(args.input_dim, args.num_running_days),
                                                        conv_hidden_size=args.dim_conv_hidden_size,
                                                        conv_kernel_size=args.conv_kernel_size,
                                                        linear_output_size=args.dim_linear_output,
                                                        final_output_size=args.input_dim).cuda()
    try:
        convolution_model.load_state_dict(torch.load('convolution_model.pt'))
    except FileNotFoundError:
        print('--- File Not Found ---')

    loss_fn = nn.MSELoss()
    optimizer_fn = optim.SGD(convolution_model.parameters(),
                             lr=args.learning_rate,
                             momentum=0.9,
                             weight_decay=1e-6)

    exp_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_fn, lr_lambda=[sin_lr])

    data_loader = functools.partial(get_dl,
                                    tickers=args.tickers,
                                    num_state_space=args.num_running_days,
                                    batch_size=args.batch_size,
                                    DataClass=PortfolioData)

    try:
        train_model_continuous(convolution_model,
                               loss_fn,
                               optimizer_fn,
                               exp_lr_scheduler,
                               args.n_train,
                               data_loader)
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
    finally:
        print('Saving...')
        torch.save(convolution_model.state_dict(), 'convolution_model.pt')
