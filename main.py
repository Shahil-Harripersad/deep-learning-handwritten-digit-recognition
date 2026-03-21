from nn.neural_network import Neuron
from utils.mnist_data_loader import MnistDataloader
from os.path import join
import numpy as np


def main():
    input_path = 'data'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    print(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")

    # flatten each image to 1d vector of size 784
    x_train = np.array(x_train).reshape(-1, 28*28)
    x_test = np.array(x_test).reshape(-1, 28*28)

    # single neuron
    output = np.dot(x_train[0], np.random.rand(28*28)) + np.random.rand()
    print(f"Label: {y_train[0]}, Output: {output}")

    # layer of neurons
    layer = []
    for _ in range(10):
        weights = np.random.rand(28*28)
        bias = np.random.rand()
        layer.append(Neuron(weights, bias))

    # feedforward through the layer
    outputs = [neuron.feedforward(x_train[0]) for neuron in layer]
    print(f"Label: {y_train[0]}, Outputs: {outputs}")


if __name__ == "__main__":
    main()