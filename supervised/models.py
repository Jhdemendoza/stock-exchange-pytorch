import torch
import torch.nn as nn


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


class Encoder(nn.Module):
    def __init__(self, input_shape, final_output_size):
        super(Encoder, self).__init__()
        self.c1 = nn.Conv1d(input_shape[0], final_output_size * 2, (4,), 4)
        self.c2 = nn.Conv1d(final_output_size * 2, final_output_size * 1, (1,), 1)
        self.c3 = nn.Conv1d(final_output_size * 1, 1, (1,), 1)

    def forward(self, x):
        x = self.c1(x)
        x = torch.relu(x)
        x = self.c2(x)
        x = torch.relu(x)
        x = self.c3(x)
        return torch.tanh(x)


class Decoder(nn.Module):
    def __init__(self, input_shape, final_output_size):
        super(Decoder, self).__init__()
        self.c1 = nn.ConvTranspose1d(1,
                                     final_output_size * 1, (1,), stride=1)
        self.c2 = nn.ConvTranspose1d(final_output_size * 1,
                                     final_output_size * 2, (1,), 1)
        self.c3 = nn.ConvTranspose1d(final_output_size * 2,
                                     input_shape[0], (4,), 4)

    def forward(self, x):
        x = self.c1(x)
        x = torch.relu(x)
        x = self.c2(x)
        x = torch.relu(x)
        x = self.c3(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, final_output_size):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_shape, final_output_size)
        self.decoder = Decoder(input_shape, final_output_size)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)