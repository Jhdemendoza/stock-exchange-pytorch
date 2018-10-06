import torch
import torch.nn as nn


class DiscreteModel(torch.nn.Module):
    def __init__(self, input_size, rnn_hidden_size,
                 output_size, num_layers):
        super(DiscreteModel, self).__init__()
        self.rnn = torch.nn.GRU(input_size, rnn_hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.linear = nn.Linear(rnn_hidden_size, output_size)
        self.h_0 = None

    def forward(self, x):
        out, self.h_0 = self.rnn(x)
        x = out[:, -1:, :].squeeze()
        return self.linear(x)


class ContinuousModel(torch.nn.Module):
    def __init__(self, input_size, rnn_hidden_size,
                 output_size, num_layers, final_output):
        super(ContinuousModel, self).__init__()
        self.rnn = torch.nn.GRU(input_size,
                                rnn_hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.linear1 = nn.Linear(rnn_hidden_size, output_size)
        self.linear2 = nn.Linear(output_size, final_output)
        self.h_0 = None

    def forward(self, x):
        out, self.h_0 = self.rnn(x)
        x = out[:, -1:, :].squeeze()
        x = torch.tanh(self.linear1(x))
        return torch.tanh(self.linear2(x))