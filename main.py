from nn.neural_network import Neuron, Layer
from utils.mnist_data_loader import load_mnist_data
from utils.utils import flatten, normalize, one_hot_encode, softmax, cross_entropy
import numpy as np

LEARNING_RATE = 0.01

def main():
    # LOAD DATA
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")

    image = flatten(x_train[0])
    image = normalize(image)
    label = y_train[0]
    true_labels = one_hot_encode(label)
    print(f"True label: {true_labels}")

    # CREATE LAYER
    weights = [np.random.rand(28*28) for _ in range(10)]
    bias = np.random.rand(10)

    for epoch in range(10):
        print(f"Epoch {epoch + 1}")

        layer = Layer(weights, bias)
        probabilities, prediction = layer.feedforward(image)
        loss = cross_entropy(true_labels, probabilities)

        print(f"Label: {label} | Prediction: {prediction} | Loss: {loss:.2f} | prob correct class: {probabilities[label]:.2f}")

        # CALCULATE GRADIENTS
        error = probabilities - true_labels
        gradient_weights = np.outer(error, image)
        gradient_bias = error

        # UPDATE WEIGHTS AND BIAS
        weights = weights - LEARNING_RATE * gradient_weights
        bias = bias - LEARNING_RATE * gradient_bias


if __name__ == "__main__":
    main()