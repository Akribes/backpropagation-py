import random

from matrix import *
from util import *


class NeuralNetwork:
    def __init__(self, structure, learning_rate=10, log_level=0):
        self.learning_rate = learning_rate
        self.log_level = log_level

        self.weights = []
        for i in range(len(structure) - 1):
            self.weights.append(Matrix(structure[i + 1], structure[i] + 1))

    def randomise_weights(self, seed=0):
        random.seed(seed)
        for w in self.weights:
            for row in w.data:
                for x in range(len(row)):
                    row[x] = random.uniform(-5, 5)

    cost_avg_prev = 0
    cost_avg = 0

    def evaluate(self, x):
        activation = [Matrix([x]).transpose()]

        # loop through layers
        for w in self.weights:
            # copy activation from last layer
            z = activation[-1]

            # add bias as a node
            z.data.append([1])

            # calculate weighted inputs
            z = w * z

            # calculate activation and cache derivatives for backwards pass
            activation.append(sigmoid(z))
        return activation[-1].transpose().data[0]

    def train(self, dataset, iterations):
        for iteration in range(iterations):
            if self.log_level >= 1:
                print("Iteration {}/{}".format(iteration + 1, iterations))

            # reset weight gradient
            weights_gradient = []
            for weight in self.weights:
                weights_gradient.append(Matrix(weight.m(), weight.n()))

            self.cost_avg = 0

            for d in range(len(dataset)):
                data = dataset[d]
                activation = []
                derivatives = []

                # FORWARDS PASS
                activation.append(Matrix([data[0]]).transpose())
                # derivatives.append(Matrix([[1] for _ in range(len(data))]))

                # loop through layers
                for w in self.weights:
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
                self.cost_avg += cost / len(dataset)

                if self.log_level >= 2:
                    print("Input: {}\tOutput: {}\tCost: {}".format(activation[0][:-1, :], activation[-1], cost))

                # calculate error of output layer
                errors = []
                errors.insert(0, derivatives[-1].hadamard_mul(activation[-1] - t))

                # calculate errors of hidden layers backwards
                for i in reversed(range(len(self.weights) - 1)):
                    errors.insert(0, derivatives[i].hadamard_mul(self.weights[i + 1][:, :-1].transpose() * errors[0]))

                # calculate weights gradients
                for i in range(len(weights_gradient)):
                    weights_gradient[i] += (errors[i] * activation[i].transpose()).scalar_div(len(dataset))

            # update weights
            for i in range(len(self.weights)):
                self.weights[i] -= weights_gradient[i].scalar_mul(self.learning_rate)

            if self.log_level >= 1:
                print("Average cost (last | current | diff): {} | {} | {}\n".format(
                    self.cost_avg_prev, self.cost_avg, self.cost_avg - self.cost_avg_prev))
            self.cost_avg_prev = self.cost_avg
