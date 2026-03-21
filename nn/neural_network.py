import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return total

class Layer:
    def __init__(self, weights, bias):
        self.neurons = []
        for i in range(len(weights)):
            self.neurons.append(Neuron(weights[i], bias[i]))
    
    def feedforward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.feedforward(inputs))
        return outputs