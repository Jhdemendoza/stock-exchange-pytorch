import torch
import torch.nn as nn


class Model(torch.nn.Module):
    # batch_size should be refactored ...
    def __init__(self, input_size, rnn_hidden_size,
                 output_size, batch_size, num_layers):
        super(Model, self).__init__()
        self.rnn = torch.nn.GRU(input_size, rnn_hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.linear = nn.Linear(rnn_hidden_size, output_size)
        self.h_0 = None

    def forward(self, x):
        out, self.h_0 = self.rnn(x)
        x = out[:, -1:, :].squeeze().cuda()
        return self.linear(x)

