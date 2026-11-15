"""
Neural Network Architectures for Digit Recognition (Digits 1-9).
Implements:
1. PyTorch Convolutional Neural Network (DigitCNN) if torch is available.
2. Scikit-Learn Deep Multi-Layer Perceptron (DigitMLP) as a robust, dependency-free alternative.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class DigitCNN(nn.Module):
        """
        Lightweight CNN classifier for single-channel 28x28 handwritten digits (classes 1-9).
        """
        def __init__(self, num_classes: int = 9):
            super(DigitCNN, self).__init__()
            self.num_classes = num_classes
            self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)

            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

        def set_temperature(self, temp: float) -> None:
            self.temperature.data = torch.tensor([temp], dtype=torch.float32)

        def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                scaled_logits = logits / self.temperature
                probs = F.softmax(scaled_logits, dim=-1)
            return probs


from sklearn.neural_network import MLPClassifier


class DigitNeuralNetwork:
    """
    Multi-Layer Perceptron Neural Network for handwritten digit recognition (classes 1-9).
    Architecture: Input (784) -> Hidden 1 (256, ReLU) -> Hidden 2 (128, ReLU) -> Output (9, Softmax).
    Trained with Adam optimizer and Cross-Entropy Loss.
    """

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (256, 128), random_state: int = 42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=30,
            batch_size=128,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=random_state
        )
        self.temperature: float = 1.0
        self.is_trained: bool = False

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the neural network.
        x_train: (N, 28, 28) or (N, 784) in [0, 1]
        y_train: (N,) labels in {1..9}
        """
        x_flat = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_flat, y_train)
        self.is_trained = True

    def set_temperature(self, temp: float) -> None:
        """Sets the calibration temperature for predicted probability distributions."""
        self.temperature = max(0.01, float(temp))

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Extracts unnormalized log-probabilities (logits) from the MLP."""
        x_flat = x.reshape(x.shape[0], -1)
        raw_probs = self.model.predict_proba(x_flat)
        # Convert probabilities back to logits via log(p + eps)
        eps = 1e-12
        logits = np.log(np.clip(raw_probs, eps, 1.0 - eps))
        return logits

    def predict_probabilities(self, x: np.ndarray) -> np.ndarray:
        """Computes calibrated softmax probabilities over classes 1-9."""
        logits = self.predict_logits(x)
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (predicted_digits, confidences). Digits are 1-9."""
        probs = self.predict_probabilities(x)
        top_indices = np.argmax(probs, axis=-1)
        # Classes are ordered 1..9 in model.classes_
        digits = self.model.classes_[top_indices]
        confs = np.max(probs, axis=-1)
        return digits, confs

    def predict_patch(self, patch: np.ndarray) -> Dict[int, float]:
        """Infers calibrated probabilities for a single 28x28 patch."""
        patch_2d = patch.reshape(1, 28, 28)
        probs = self.predict_probabilities(patch_2d)[0]
        return {int(cls): float(probs[i]) for i, cls in enumerate(self.model.classes_)}
