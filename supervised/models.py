import numpy as np
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, ticker_dim, data_point_dim, shift_dim, transform_dim, output_dim):
        '''
        :param ticker_dim     : dim of tickers used for the data
        :param data_point_dim : dim of a given data point (e.g. ohlc+volume = 5)
        :param shift_dim      : dim of shifts in time scales (e.g. 4 different shifts backs for returns)
        :param transform_dim  : dim of different transforms (e.g. 4 different sckit-transforms)
        :param output_dim     : dim of outputs, should be a multiple of ticker_dim (e.g. ticker_dim * 2)
        '''
        assert output_dim % ticker_dim == 0, 'output_dim should be divisible by ticker_dim'

        self.input_dim = ticker_dim * shift_dim * data_point_dim * transform_dim
        self.output_dim = output_dim

        self.conv_channel = output_dim // ticker_dim

        super(Classifier, self).__init__()

        self.l1 = nn.Linear(self.input_dim, output_dim).double()

        self.c1 = nn.Conv1d(1,
                            self.conv_channel,
                            kernel_size=shift_dim * data_point_dim,
                            stride=shift_dim * data_point_dim).double()
        self.c2 = nn.Conv1d(self.conv_channel,
                            self.conv_channel,
                            kernel_size=transform_dim,
                            stride=transform_dim).double()

        self.linear_repeat_dim, self.conv_repeat_dim = self._return_repeat_dim()

        self.c3 = nn.Conv1d(self.conv_channel, 1, (1,)).double()

    def _compute_conv_output_shape(self):
        temp_input = torch.ones((1, 1, self.input_dim), requires_grad=False).double()
        return self.c2(self.c1(temp_input)).shape

    def _compute_repeat_dim(self):
        temp_input = torch.ones((1, self.input_dim), requires_grad=False).double()
        linear_output_shape = self.l1(temp_input).unsqueeze(1).shape

        conv_output_shape = self._compute_conv_output_shape()

        return [conv / linear for conv, linear in
                zip(conv_output_shape, linear_output_shape)]

    def _return_repeat_dim(self):
        conv_div_linear_dims = self._compute_repeat_dim()

        linear_output_repeat_dim = [item if item >= 1.0 else 1.0 for item in conv_div_linear_dims]
        conv_output_repeat_dim = [1.0 if item >= 1.0 else 1 / item for item in conv_div_linear_dims]

        linear_output_repeat_dim, conv_output_repeat_dim = list(map(lambda x: np.array(x).astype(int),
                                                                    [linear_output_repeat_dim, conv_output_repeat_dim]))

        return linear_output_repeat_dim, conv_output_repeat_dim

    def forward(self, input):
        linear = self.l1(input)
        linear = torch.sigmoid(linear).unsqueeze(1)
        linear = linear.repeat(*self.linear_repeat_dim)

        conv = input.unsqueeze(1)
        conv = torch.sigmoid(self.c1(conv))
        conv = torch.sigmoid(self.c2(conv))
        conv = conv.repeat(*self.conv_repeat_dim)

        out = linear + conv
        return torch.sigmoid(self.c3(out)).squeeze(1)

    def show_parameter_shapes(self):
        '''
        Use this function as a reminder of horrible param sizes...
        '''
        return [param.shape for child in self.children() for param in child.parameters()]


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