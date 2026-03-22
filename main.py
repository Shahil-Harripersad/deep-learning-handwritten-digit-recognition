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
    save_training_plot,
)

LEARNING_RATE = 0.01
EPOCHS = 10
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10
OUTPUTS_DIRECTORY = Path("outputs")
MODEL_PATH = OUTPUTS_DIRECTORY / "model.npz"
PLOT_PATH = OUTPUTS_DIRECTORY / "training_plot.png"

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
    accuracies = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        epoch_loss = 0.0
        correct_predictions = 0
        shuffled_indices = np.random.permutation(len(training_inputs))

        for index in shuffled_indices:
            image, label = training_inputs[index]
            true_labels = one_hot_encode(label)

            probabilities, prediction = network.train(image, true_labels, LEARNING_RATE)
            loss = cross_entropy(true_labels, probabilities)
            epoch_loss += loss

            if prediction == label:
                correct_predictions += 1

        average_loss = epoch_loss / len(training_inputs)
        average_accuracy = correct_predictions / len(training_inputs)
        losses.append(average_loss)
        accuracies.append(average_accuracy)
        print(
            f"Average loss: {average_loss:.4f} | "
            f"Average accuracy: {average_accuracy:.4f}"
        )

    return losses, accuracies

def build_network():
    return NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model and save it.")
    parser.add_argument("--test", action="store_true", help="Load the saved model and test it.")
    parser.add_argument("--gui", action="store_true", help="Open a drawing window and predict digits.")
    args = parser.parse_args()

    if not args.train and not args.test and not args.gui:
        parser.error("Use --train, --test, or --gui.")

    if args.gui:
        from gui import launch_gui

        launch_gui()
        return

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")

    network = build_network()

    if args.train:
        training_inputs = prepare_inputs(x_train, y_train)
        losses, accuracies = train_model(network, training_inputs)
        OUTPUTS_DIRECTORY.mkdir(exist_ok=True)
        network.save_model(MODEL_PATH)
        save_training_plot(losses, accuracies, PLOT_PATH)
        print(f"Model saved to {MODEL_PATH}")
        print(f"Training plot saved to {PLOT_PATH}")

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
