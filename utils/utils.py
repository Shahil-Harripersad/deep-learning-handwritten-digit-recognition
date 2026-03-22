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


def relu(values):
    return np.maximum(0, values)


def relu_derivative(values):
    return (values > 0).astype(float)


def save_training_plot(losses, accuracies, plot_path):
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(losses) + 1)

    figure, loss_axis = plt.subplots(figsize=(10, 5))
    accuracy_axis = loss_axis.twinx()

    loss_line = loss_axis.plot(epochs, losses, color="tab:blue", label="Loss", linewidth=2)
    accuracy_line = accuracy_axis.plot(
        epochs,
        accuracies,
        color="tab:orange",
        label="Accuracy",
        linewidth=2,
    )

    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss", color="tab:blue")
    accuracy_axis.set_ylabel("Accuracy", color="tab:orange")
    loss_axis.set_title("Training Loss and Accuracy")
    loss_axis.grid(True, alpha=0.3)

    lines = loss_line + accuracy_line
    labels = [line.get_label() for line in lines]
    loss_axis.legend(lines, labels, loc="center right")

    figure.tight_layout()
    figure.savefig(plot_path)
    plt.close(figure)
