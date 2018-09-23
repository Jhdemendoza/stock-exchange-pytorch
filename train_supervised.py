import torch
import torch.nn as nn
import numpy as np
from torch import optim
from supervised import train_model, Model
from supervised.environment import *

def train():
    gru_model = Model(1, 64, NUM_DISCRETE_RETURNS, BATCH_SIZE, 2).double().cuda()
    try:
        gru_model.load_state_dict(torch.load('gru_model.pt'))
    except FileNotFoundError:
        print('--- File Not Found ---')

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = optim.SGD(gru_model.parameters(),
                             lr=1e-6, momentum=0.9, weight_decay=1e-7)

    def sin_lr(x):
        return np.abs(np.sin((x + 0.01) * 0.2))

    exp_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_fn,
                                                   lr_lambda=[sin_lr])

    try:
        train_model(gru_model, loss_fn, optimizer_fn, exp_lr_scheduler, 5000)
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
    finally:
        print('Saving...')
        torch.save(gru_model.state_dict(), 'gru_model.pt')


if __name__ == '__main__':
    train()