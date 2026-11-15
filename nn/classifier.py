"""
High-Level Inference Wrapper for Sudoku Digit Recognition.
Loads trained neural network weights and provides inference methods for 28x28 patches.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import joblib

from nn.model import DigitNeuralNetwork


class SudokuDigitClassifier:
    """Inference wrapper for the trained Digit Neural Network."""

    def __init__(self, weights_path: Optional[str] = "nn/weights/digit_nn.joblib"):
        self.nn = DigitNeuralNetwork()
        self.is_loaded = False

        if weights_path is not None and os.path.exists(weights_path):
            try:
                data = joblib.load(weights_path)
                self.nn.model = data["model"]
                self.nn.set_temperature(data.get("temperature", 1.0))
                self.nn.is_trained = True
                self.is_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load weights from {weights_path}: {e}")
                self.is_loaded = False

    def predict_patch(self, patch: np.ndarray) -> Dict[int, float]:
        """
        Infers probabilities for a single 28x28 patch.
        Returns a dictionary mapping digits 1..9 to calibrated probabilities.
        """
        if not self.is_loaded:
            raise RuntimeError("Model weights not loaded. Please train or provide valid weights.")
        return self.nn.predict_patch(patch)

    def predict_batch(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infers top digit and probabilities for a batch of patches (N, 28, 28).
        """
        if not self.is_loaded:
            raise RuntimeError("Model weights not loaded.")
        return self.nn.predict(patches)
