import numpy as np

lst = [4, 8, 16, 32, 64]


def extend(p):
    return np.concatenate([p[:, -1:, :], p, p[:, :1, :]], axis=1)


def diff(p, axis):
    if axis == 0:
        out = np.zeros_like(p)
        out[:-1] = p[1:] - p[:-1]
        out[1:] += p[1:] - p[:-1]
        out[1:-1] /= 2
        return out
    return np.roll(p, 1, axis=axis) - p


def diff2(p, axis):
    assert axis != 0
    return (2 * p - np.roll(p, 1, axis=axis) - np.roll(p, -1, axis=axis)) / 2


def clean(p):
    return p[:, 1:-1, :]
