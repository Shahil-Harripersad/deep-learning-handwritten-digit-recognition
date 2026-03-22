# 🧠 Digit Recognizer (From Scratch Neural Network)

A fully from-scratch implementation of a neural network trained on the MNIST dataset to recognize handwritten digits (0–9), built using only NumPy.

This project focuses on understanding **how neural networks actually work under the hood**, without relying on deep learning frameworks like PyTorch or TensorFlow.

---

## Demo

### Draw Your Own Digit



https://github.com/user-attachments/assets/f3142e32-5216-4b9b-9406-39393103ba83



The model predicts digits in real-time based on user input.

---

## Project Overview

This project implements a basic feedforward neural network:

```text
784 (input pixels) → Hidden Layer (ReLU) → 10 outputs (softmax)
```

* Input: 28×28 grayscale image (flattened to 784)
* Output: Probability distribution over digits 0–9

---

## How It Works

### Forward Pass

1. Input image is flattened into a vector
2. Passed through a hidden layer with ReLU activation
3. Passed through an output layer
4. Softmax converts outputs into probabilities

---

### Loss Function

* **Cross-Entropy Loss**
* Measures how far predictions are from the true label

---

### Backpropagation

Gradients are computed manually:

* Output layer error: `predicted - true`
* Gradients flow backward through the network
* Weights and biases updated using gradient descent

---

### Training

* Model trained on MNIST dataset
* Loss decreases and accuracy improves over epochs
* Inputs normalized for stable learning

---

## Evaluation

* Evaluated on test dataset
* Tracks:

  * Accuracy
  * Average loss

---

## GUI

A simple GUI allows you to:

* Draw digits with your mouse
* Get real-time predictions from the model

This makes the model interactive and easy to test.

---

## Project Structure

```text
digit-recognizer/
├── nn/                # Neural network implementation
├── utils/             # Data loading and helper functions
├── gui/               # Drawing interface
├── main.py            # Training + evaluation entry point
```

---

## How to Run

```bash
pip install -r requirements.txt
python main.py --train
python main.py --test
python main.py --gui
```

---

## Key Learnings

* Neural networks are just layered linear transformations + non-linearity
* Softmax + cross-entropy provides a strong learning signal
* Backpropagation enables efficient weight updates
* Hidden layers allow learning of complex patterns
* Input normalization is critical for stable training

---

## Takeaway

This project demonstrates that deep learning is not magic. It is a combination of:

> matrix operations + gradients + iteration

Understanding this from scratch builds strong intuition for more advanced models.

---

## Future Improvements

* Add more hidden layers
* Implement batching and optimization techniques
* Visualize learned weights and activations
* Extend to more complex datasets
