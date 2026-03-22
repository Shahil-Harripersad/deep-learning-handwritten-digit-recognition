from pathlib import Path
import tkinter as tk

import numpy as np

from nn.neural_network import NeuralNetwork

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10
MODEL_PATH = Path("outputs") / "model.npz"
CANVAS_SIZE = 280
GRID_SIZE = 28
CELL_SIZE = CANVAS_SIZE // GRID_SIZE
BRUSH_RADIUS = 12


def build_network():
    return NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)


def resize_to_28x28(image):
    return image.reshape(GRID_SIZE, CELL_SIZE, GRID_SIZE, CELL_SIZE).max(axis=(1, 3))


def center_digit(image):
    row_positions, column_positions = np.where(image > 0)

    if len(row_positions) == 0 or len(column_positions) == 0:
        return image

    top_row = row_positions.min()
    bottom_row = row_positions.max() + 1
    left_column = column_positions.min()
    right_column = column_positions.max() + 1

    cropped_digit = image[top_row:bottom_row, left_column:right_column]
    centered_image = np.zeros((GRID_SIZE, GRID_SIZE))

    digit_height, digit_width = cropped_digit.shape
    row_start = (GRID_SIZE - digit_height) // 2
    column_start = (GRID_SIZE - digit_width) // 2

    centered_image[
        row_start:row_start + digit_height,
        column_start:column_start + digit_width,
    ] = cropped_digit

    return centered_image


class DigitRecognizerGui:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.network = build_network()
        self.network.load_model(MODEL_PATH)

        self.window = tk.Tk()
        self.window.title("Digit Recognizer")
        self.window.resizable(False, False)

        self.canvas_data = np.zeros((CANVAS_SIZE, CANVAS_SIZE))

        self.title_label = tk.Label(
            self.window,
            text="Draw a digit from 0 to 9",
            font=("Arial", 16),
        )
        self.title_label.pack(pady=(12, 8))

        self.canvas = tk.Canvas(
            self.window,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="#666666",
        )
        self.canvas.pack(padx=12, pady=8)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<Button-1>", self.draw_digit)

        self.result_label = tk.Label(
            self.window,
            text="Prediction: -",
            font=("Arial", 14),
        )
        self.result_label.pack(pady=(8, 4))

        self.confidence_label = tk.Label(
            self.window,
            text="Confidence: -",
            font=("Arial", 12),
        )
        self.confidence_label.pack(pady=(0, 10))

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(pady=(0, 12))

        self.predict_button = tk.Button(
            self.button_frame,
            text="Predict",
            command=self.predict_digit,
            width=12,
        )
        self.predict_button.pack(side=tk.LEFT, padx=6)

        self.clear_button = tk.Button(
            self.button_frame,
            text="Clear",
            command=self.clear_canvas,
            width=12,
        )
        self.clear_button.pack(side=tk.LEFT, padx=6)

    def draw_digit(self, event):
        x_position = event.x
        y_position = event.y

        self.canvas.create_oval(
            x_position - BRUSH_RADIUS,
            y_position - BRUSH_RADIUS,
            x_position + BRUSH_RADIUS,
            y_position + BRUSH_RADIUS,
            fill="white",
            outline="white",
        )

        for row in range(max(0, y_position - BRUSH_RADIUS), min(CANVAS_SIZE, y_position + BRUSH_RADIUS + 1)):
            for column in range(max(0, x_position - BRUSH_RADIUS), min(CANVAS_SIZE, x_position + BRUSH_RADIUS + 1)):
                distance = (column - x_position) ** 2 + (row - y_position) ** 2
                if distance <= BRUSH_RADIUS ** 2:
                    self.canvas_data[row, column] = 1.0

    def predict_digit(self):
        resized_image = resize_to_28x28(self.canvas_data)
        centered_image = center_digit(resized_image)
        flattened_image = centered_image.reshape(-1)

        _, _, probabilities, prediction = self.network.feedforward(flattened_image)
        confidence = probabilities[prediction]

        self.result_label.config(text=f"Prediction: {prediction}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2%}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas_data.fill(0.0)
        self.result_label.config(text="Prediction: -")
        self.confidence_label.config(text="Confidence: -")

    def run(self):
        self.window.mainloop()


def launch_gui():
    app = DigitRecognizerGui()
    app.run()
