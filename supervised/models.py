import torch
import torch.nn as nn


class DiscreteModelBasicGru(torch.nn.Module):
    def __init__(self, input_size, rnn_hidden_size,
                 output_size, num_layers):
        super(DiscreteModelBasicGru, self).__init__()
        self.rnn1 = torch.nn.GRU(input_size, rnn_hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.rnn2 = torch.nn.GRU(rnn_hidden_size, rnn_hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.linear = nn.Linear(rnn_hidden_size, output_size)
        self.h_0 = None

    def forward(self, x):
        out, self.h_0 = self.rnn(x)
        x = out[:, -1:, :].squeeze()
        return self.linear(x)


class ContinuousModelBasicGru(torch.nn.Module):
    def __init__(self, input_size, rnn_hidden_size,
                 linear_output_size, num_layers, final_output):
        '''
        :param input_size: number of tickers if not transformed
        :param rnn_hidden_size: rnn hidden dimension
        :param linear_output_size: linear output size
        :param num_layers: GRU layers dimension
        :param final_output: number of tickers (same as input_size unless transformed)
        '''
        super(ContinuousModelBasicGru, self).__init__()
        self.rnn = torch.nn.GRU(input_size,
                                rnn_hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.linear1 = nn.Linear(rnn_hidden_size, linear_output_size)
        self.linear2 = nn.Linear(linear_output_size, final_output)
        self.h_0 = None

    def forward(self, x):
        out, self.h_0 = self.rnn(x)
        x = out[:, -1:, :].squeeze()
        x = torch.tanh(self.linear1(x))
        return torch.tanh(self.linear2(x))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ContinuousModelBasicConvolution(torch.nn.Module):
    def __init__(self,
                 input_shape,
                 conv_hidden_size,
                 conv_kernel_size,
                 linear_output_size,
                 final_output_size):
        '''
        :param input_shape: tuple of (num_tickers, num_running_days)
        :param conv_hidden_size: hidden dimension
        :param conv_kernel_size: kernel dimension for 1d convolution
        :param linear_output_size: linear output dimension
        :param final_output_size: num_tickers
        '''
        assert isinstance(input_shape, tuple)
        super(ContinuousModelBasicConvolution, self).__init__()
        self.c1 = nn.Conv1d(input_shape[0], conv_hidden_size, conv_kernel_size)
        self.b1 = nn.BatchNorm1d(conv_hidden_size)
        self.c2 = nn.Conv1d(conv_hidden_size, conv_hidden_size, conv_kernel_size)
        self.b2 = nn.BatchNorm1d(conv_hidden_size)
        self.c3 = nn.Conv1d(conv_hidden_size, conv_hidden_size, conv_kernel_size)
        self.b3 = nn.BatchNorm1d(conv_hidden_size)

        self.flatten = Flatten()

        flatten_dim = self._get_conv_output(input_shape)

        self.linear1 = nn.Linear(flatten_dim, linear_output_size)
        self.linear2 = nn.Linear(linear_output_size, final_output_size)

    def _forward_features(self, x):
        x = torch.relu(self.b1(self.c1(x)))
        x = torch.relu(self.b2(self.c2(x)))
        x = torch.relu(self.b3(self.c3(x)))
        return x

    def _get_conv_output(self, shape):
        batch_size = 1
        _input = torch.rand(batch_size, *shape)
        _output = self._forward_features(_input).detach()
        n_size = _output.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        # x is being permuted...
        x = self._forward_features(x.permute(0, 2, 1))
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        return torch.tanh(self.linear2(x))

