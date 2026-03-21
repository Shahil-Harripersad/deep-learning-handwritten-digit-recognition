from nn.neural_network import Neuron, Layer
from utils.mnist_data_loader import load_mnist_data
from utils.utils import flatten, one_hot_encode, softmax, cross_entropy
import numpy as np

LEARNING_RATE = 0.01

def main():
    # LOAD DATA
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")

    image = flatten(x_train[0])
    label = y_train[0]
    true_labels = one_hot_encode(label)
    print(f"True label: {true_labels}")

    # CREATE LAYER
    weights = [np.random.rand(28*28) for _ in range(10)]
    bias = np.random.rand(10)
    layer = Layer(weights, bias)

    outputs = layer.feedforward(image)

    # use softmax to convert outputs to probabilities
    probabilities = softmax(outputs)

    # prediction
    prediction = np.argmax(probabilities)

    # calculate loss
    loss = cross_entropy(true_labels, probabilities)

    print(f"Label: {label}\n Probabilities: {probabilities}\n Prediction: {prediction}\n Loss: {loss}")
    print(f"probability of correct class: {probabilities[label]}")

    # CALCULATE GRADIENTS
    error = probabilities - true_labels
    gradient_weights = np.outer(error, image)
    gradient_bias = error

    # UPDATE WEIGHTS AND BIAS
    weights = weights - LEARNING_RATE * gradient_weights
    bias = bias - LEARNING_RATE * gradient_bias

    new_layer = Layer(weights, bias)

    print(f"Updated weights: {weights}\n Updated bias: {bias}")


if __name__ == "__main__":
    main()