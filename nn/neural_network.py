import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # activation function
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return total