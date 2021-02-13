from nn import *

dataset = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

net = NeuralNetwork((2, 3, 1))
net.randomise_weights()
net.train(dataset, 3000)
print(net.evaluate((1, 0)))
