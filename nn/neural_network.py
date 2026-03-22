import numpy as np

from utils.utils import softmax


def relu(values):
    return np.maximum(0, values)


def relu_derivative(values):
    return (values > 0).astype(float)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_to_hidden_weights = np.random.randn(hidden_size, input_size) * 0.01
        self.hidden_bias = np.zeros(hidden_size)
        self.hidden_to_output_weights = np.random.randn(output_size, hidden_size) * 0.01
        self.output_bias = np.zeros(output_size)

    def feedforward(self, image):
        hidden_inputs = np.dot(self.input_to_hidden_weights, image) + self.hidden_bias
        hidden_outputs = relu(hidden_inputs)

        output_inputs = np.dot(self.hidden_to_output_weights, hidden_outputs) + self.output_bias
        output_probabilities = softmax(output_inputs)
        predicted_label = np.argmax(output_probabilities)

        return hidden_inputs, hidden_outputs, output_probabilities, predicted_label

    def train(self, image, true_labels, learning_rate):
        hidden_inputs, hidden_outputs, output_probabilities, predicted_label = self.feedforward(image)

        output_error = output_probabilities - true_labels
        hidden_to_output_weight_gradients = np.outer(output_error, hidden_outputs)
        output_bias_gradients = output_error

        hidden_error = np.dot(self.hidden_to_output_weights.T, output_error) * relu_derivative(hidden_inputs)
        input_to_hidden_weight_gradients = np.outer(hidden_error, image)
        hidden_bias_gradients = hidden_error

        self.hidden_to_output_weights = (
            self.hidden_to_output_weights - learning_rate * hidden_to_output_weight_gradients
        )
        self.output_bias = self.output_bias - learning_rate * output_bias_gradients
        self.input_to_hidden_weights = (
            self.input_to_hidden_weights - learning_rate * input_to_hidden_weight_gradients
        )
        self.hidden_bias = self.hidden_bias - learning_rate * hidden_bias_gradients

        return output_probabilities, predicted_label
