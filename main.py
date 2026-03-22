from nn.neural_network import Neuron, Layer
from utils.mnist_data_loader import load_mnist_data
from utils.utils import (
    flatten,
    normalize,
    one_hot_encode,
    cross_entropy,
    plot_training_metrics,
)
import numpy as np

LEARNING_RATE = 0.01
EPOCHS = 10

def main():
    # LOAD DATA
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")

    inputs = []
    for image, label in zip(x_train, y_train):
        flattened_image = flatten(image)
        normalized_image = normalize(flattened_image)
        inputs.append((normalized_image, label))

    # CREATE LAYER
    weights = np.random.rand(10, 28 * 28)
    bias = np.random.rand(10)
    losses = []
    correct_class_probabilities = []

    # TRAINING LOOP
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        epoch_loss = 0.0
        epoch_correct_class_probability = 0.0
        shuffled_indices = np.random.permutation(len(inputs))

        for index in shuffled_indices:
            image, label = inputs[index]
            true_labels = one_hot_encode(label)

            layer = Layer(weights, bias)
            probabilities, prediction = layer.feedforward(image)
            loss = cross_entropy(true_labels, probabilities)
            epoch_loss += loss
            epoch_correct_class_probability += probabilities[label]

            # CALCULATE GRADIENTS
            error = probabilities - true_labels
            gradient_weights = np.outer(error, image)
            gradient_bias = error

            # UPDATE WEIGHTS AND BIAS
            weights = weights - LEARNING_RATE * gradient_weights
            bias = bias - LEARNING_RATE * gradient_bias

        average_loss = epoch_loss / len(inputs)
        average_correct_class_probability = epoch_correct_class_probability / len(inputs)
        losses.append(average_loss)
        correct_class_probabilities.append(average_correct_class_probability)
        print(
            f"Average loss: {average_loss:.4f} | "
            f"Average prob correct class: {average_correct_class_probability:.4f}"
        )

    plot_training_metrics(losses, correct_class_probabilities)


if __name__ == "__main__":
    main()
