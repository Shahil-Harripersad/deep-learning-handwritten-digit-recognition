import numpy as np


def normalize(image):
    return image / 255.0


def flatten(image):
    return np.array(image).reshape(-1)


def one_hot_encode(label, num_classes=10):
    one_hot_vector = np.zeros(num_classes)
    one_hot_vector[label] = 1
    return one_hot_vector


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred))
