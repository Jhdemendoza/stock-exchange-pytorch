import collections
import six
import scipy.stats as stats
import numpy as np


def iterable(arg):
    return (isinstance(arg, collections.Iterable) and not
            isinstance(arg, six.string_types))


def print_distribution(output):
    if output.dim() != 2:
        return
    out = output.detach().cpu().numpy()
    out = out.reshape(-1, out.shape[-1])
    print('\r{} '.format(stats.describe(out)), end='')


def sin_lr(x):
    return np.abs(np.sin((x + 0.01) * 0.2))
