import math
import random


class Matrix:
    def __init__(self, *args):
        if len(args) == 1:
            self.data = args[0]
        else:
            self.data = [[0 for _ in range(args[1])] for _ in range(args[0])]

    def __str__(self):
        return "{} * {} matrix {}".format(self.m(), self.n(), str(self.data))

    def __mul__(self, other):
        if self.n() != other.m():
            raise ValueError("Matrix sizes do not match: multiplying {} and {}".format(self, other))
        res = Matrix(self.m(), other.n())

        for j in range(res.m()):
            for i in range(res.n()):
                for k in range(self.n()):
                    res.data[j][i] += self.data[j][k] * other.data[k][i]
        return res

    def __add__(self, other):
        if self.m() != other.m() or self.n() != other.n():
            raise ValueError("Matrix sizes do not match:", self, other)
        res = Matrix(self.m(), self.n())
        for i in range(self.m()):
            for j in range(self.n()):
                res.data[i][j] = self.data[i][j] + other.data[i][j]
        return res

    def __sub__(self, other):
        if self.m() != other.m() or self.n() != other.n():
            raise ValueError("Matrix sizes do not match:", self, other)
        res = Matrix(self.m(), self.n())
        for i in range(self.m()):
            for j in range(self.n()):
                res.data[i][j] = self.data[i][j] - other.data[i][j]
        return res

    def m(self):
        return len(self.data)

    def n(self):
        return len(self.data[0])

    def transpose(self):
        t = Matrix(self.n(), self.m())
        for j in range(self.m()):
            for i in range(self.n()):
                t.data[i][j] = self.data[j][i]
        return t

    def hadamard_mul(self, other):
        res = Matrix(self.data)
        for i in range(res.m()):
            for j in range(res.n()):
                res.data[i][j] *= other.data[i][j]
        return res

    def scalar_mul(self, x):
        res = Matrix(self.data)
        for i in range(res.m()):
            for j in range(res.n()):
                res.data[i][j] *= x
        return res

    def scalar_div(self, x):
        return self.scalar_mul(1 / x)

    def __getitem__(self, item):
        if isinstance(item[0], slice):
            res = self.data[item[0]]
        else:
            res = [self.data[item[0]]]

        for i in range(len(res)):
            if isinstance(item[1], slice):
                res[i] = res[i][item[1]]
            else:
                res[i] = [res[i][item[1]]]
        if not isinstance(item[0], slice) and not isinstance(item[1], slice):
            return res
        else:
            return Matrix(res)

    def __setitem__(self, key, value):
        self.data[key[0] - 1][key[1] - 1] = value

    def __delitem__(self, key):
        self.data[key[0] - 1][key[1] - 1] = 0


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


weights = [
    # Matrix(neurons, last layer + bias)
    Matrix(3, 3),
    Matrix(1, 4)
]

# randomise initial weights
random.seed(0)
for w in weights:
    for row in w.data:
        for x in range(len(row)):
            row[x] = random.uniform(-5, 5)


dataset = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

cost_avg_prev = 0

iterations = 2000
for iteration in range(iterations):
    print("Iteration {}/{}".format(iteration + 1, iterations))

    # reset weight gradient
    weights_gradient = []
    for weight in weights:
        weights_gradient.append(Matrix(weight.m(), weight.n()))

    cost_avg = 0

    for d in range(len(dataset)):
        data = dataset[d]
        activation = []
        derivatives = []

        # FORWARDS PASS
        activation.append(Matrix([data[0]]).transpose())
        # derivatives.append(Matrix([[1] for _ in range(len(data))]))

        # loop through layers
        for w in weights:
            # copy activation from last layer
            z = activation[-1]

            # add bias as a node
            z.data.append([1])

            # calculate weighted inputs
            z = w * z

            # calculate activation and cache derivatives for backwards pass
            activation.append(sigmoid(z))
            derivatives.append(sigmoid_der(z))

        # BACKWARDS PASS
        t = Matrix([data[1]]).transpose()
        cost = sel(t, activation[-1])
        cost_avg += cost / len(dataset)
        print("Input: {}\tOutput: {}\tCost: {}".format(activation[0][:-1, :], activation[-1], cost))

        # calculate error of output layer
        errors = []
        errors.insert(0, derivatives[-1].hadamard_mul(activation[-1] - t))

        # calculate errors of hidden layers backwards
        for i in reversed(range(len(weights) - 1)):
            errors.insert(0, derivatives[i].hadamard_mul(weights[i + 1][:, :-1].transpose() * errors[0]))

        # calculate weights gradients
        for i in range(len(weights_gradient)):
            weights_gradient[i] += (errors[i] * activation[i].transpose()).scalar_div(len(dataset))

    # update weights
    for i in range(len(weights)):
        weights[i] -= weights_gradient[i].scalar_mul(10)

    print("Average cost (last | current | diff): {} | {} | {}\n".format(cost_avg_prev, cost_avg, cost_avg - cost_avg_prev))
    cost_avg_prev = cost_avg

print("See you next time")
