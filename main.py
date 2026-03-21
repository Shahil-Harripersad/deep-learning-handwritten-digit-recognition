from nn.neural_network import Neuron
from utils.mnist_data_loader import load_mnist_data
from utils.utils import flatten, one_hot_encode, softmax, cross_entropy
from os.path import join
import numpy as np


def main():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")

    input = flatten(x_train[0])
    label = y_train[0]
    true_labels = one_hot_encode(label)
    print(f"True label: {true_labels}")

    # create layer of neurons
    layer = []
    for _ in range(10):
        weights = np.random.rand(28*28)
        bias = np.random.rand()
        layer.append(Neuron(weights, bias))

    # feedforward through the layer
    outputs = [neuron.feedforward(input) for neuron in layer]

    # use softmax to convert outputs to probabilities
    probabilities = softmax(outputs)

    # prediction
    prediction = np.argmax(probabilities)

    # calculate loss
    loss = cross_entropy(probabilities, true_labels)

    print(f"Label: {label}\n Probabilities: {probabilities}\n Prediction: {prediction}\n Loss: {loss}")


if __name__ == "__main__":
    main()