import torch
import torch.nn as nn

from functools import reduce


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
        self.c1 = nn.Conv1d(input_shape[0], conv_hidden_size, conv_kernel_size, stride=2)
        self.b1 = nn.BatchNorm1d(conv_hidden_size)
        self.c2 = nn.Conv1d(conv_hidden_size, conv_hidden_size, conv_kernel_size, stride=1)
        self.b2 = nn.BatchNorm1d(conv_hidden_size)
        # self.c3 = nn.Conv1d(conv_hidden_size, conv_hidden_size, conv_kernel_size//2, stride=1)
        # self.b3 = nn.BatchNorm1d(conv_hidden_size)

        self.flatten = Flatten()

        flatten_dim = self._get_conv_output(input_shape)

        self.linear1 = nn.Linear(flatten_dim, linear_output_size)
        self.linear2 = nn.Linear(linear_output_size, final_output_size)

    def _forward_features(self, x):
        x = torch.relu(self.b1(self.c1(x)))
        # x = torch.relu(self.c1(x))
        x = torch.relu(self.b2(self.c2(x)))
        # x = torch.relu(self.c2(x))
        # x = torch.relu(self.b3(self.c3(x)))
        # x = torch.relu(self.c3(x))
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


class AutoEncoder(nn.Module):
    def __init__(self, input_size, final_output_size):
        assert isinstance(input_size, tuple)
        self.input_size = input_size
        linear_length = reduce(lambda x, y: x * y, input_size)

        super(AutoEncoder, self).__init__()
        self.flatten = Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(linear_length, final_output_size * 8),
            nn.ReLU(True),
            nn.Linear(final_output_size * 8, final_output_size * 4),
            nn.ReLU(True),
            nn.Linear(final_output_size * 4, final_output_size * 2),
            nn.ReLU(True),
            nn.Linear(final_output_size * 2, final_output_size))

        self.predictor = nn.Sequential(
            nn.Linear(final_output_size, final_output_size * 8),
            nn.ReLU(True),
            nn.Linear(final_output_size * 8, final_output_size),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(final_output_size, final_output_size * 2),
            nn.ReLU(True),
            nn.Linear(final_output_size * 2, final_output_size * 4),
            nn.ReLU(True),
            nn.Linear(final_output_size * 4, final_output_size * 8),
            nn.ReLU(True),
            nn.Linear(final_output_size * 8, linear_length))

    def forward(self, x):
        f = self.flatten(x)
        e = self.encoder(f)
        p = self.predictor(e.detach())
        d = self.decoder(e)
        r = d.reshape(x.size(0), *self.input_size)

        return r, p, x

