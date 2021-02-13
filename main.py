import random

from matrix import *
from util import *


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

iterations = 500
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
