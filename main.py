import argparse
from pathlib import Path

import numpy as np

from nn.neural_network import NeuralNetwork
from utils.mnist_data_loader import load_mnist_data
from utils.utils import (
    flatten,
    normalize,
    one_hot_encode,
    cross_entropy,
    plot_training_metrics,
)

LEARNING_RATE = 0.01
EPOCHS = 10
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10
MODEL_PATH = Path("model.npz")

def prepare_inputs(images, labels):
    inputs = []
    for image, label in zip(images, labels):
        flattened_image = flatten(image)
        normalized_image = normalize(flattened_image)
        inputs.append((normalized_image, label))
    return inputs


def evaluate_model(network, test_inputs):
    total_test_loss = 0.0
    correct_predictions = 0

    for image, label in test_inputs:
        true_labels = one_hot_encode(label)
        _, _, probabilities, prediction = network.feedforward(image)
        loss = cross_entropy(true_labels, probabilities)
        total_test_loss += loss

        if prediction == label:
            correct_predictions += 1

    average_test_loss = total_test_loss / len(test_inputs)
    test_accuracy = correct_predictions / len(test_inputs)
    return average_test_loss, test_accuracy


def train_model(network, training_inputs):
    losses = []
    correct_class_probabilities = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        epoch_loss = 0.0
        epoch_correct_class_probability = 0.0
        shuffled_indices = np.random.permutation(len(training_inputs))

        for index in shuffled_indices:
            image, label = training_inputs[index]
            true_labels = one_hot_encode(label)

            probabilities, prediction = network.train(image, true_labels, LEARNING_RATE)
            loss = cross_entropy(true_labels, probabilities)
            epoch_loss += loss
            epoch_correct_class_probability += probabilities[label]

        average_loss = epoch_loss / len(training_inputs)
        average_correct_class_probability = epoch_correct_class_probability / len(training_inputs)
        losses.append(average_loss)
        correct_class_probabilities.append(average_correct_class_probability)
        print(
            f"Average loss: {average_loss:.4f} | "
            f"Average prob correct class: {average_correct_class_probability:.4f}"
        )

    plot_training_metrics(losses, correct_class_probabilities)

def build_network():
    return NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model and save it.")
    parser.add_argument("--test", action="store_true", help="Load the saved model and test it.")
    args = parser.parse_args()

    if not args.train and not args.test:
        parser.error("Use --train or --test.")

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")

    network = build_network()

    if args.train:
        training_inputs = prepare_inputs(x_train, y_train)
        train_model(network, training_inputs)
        network.save_model(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    if args.test:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        network.load_model(MODEL_PATH)
        test_inputs = prepare_inputs(x_test, y_test)
        average_test_loss, test_accuracy = evaluate_model(network, test_inputs)
        print(f"Test loss: {average_test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
