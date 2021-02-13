import math


def sel(t, y):
    e = t - y
    return (e.transpose() * e).data[0][0] / 2


def sigmoid(z):
    a = z[:, :]
    for i in range(a.m()):
        for j in range(a.n()):
            x = a.data[i][j]
            a.data[i][j] = 1 / (1 + math.exp(-x))
    return a


def sigmoid_der(z):
    d = z[:, :]
    for i in range(d.m()):
        for j in range(d.n()):
            x = d.data[i][j]
            d.data[i][j] = 1 / (1 + math.exp(-x)) * 1 / (1 + math.exp(x))
    return d
