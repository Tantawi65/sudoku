"""
Probability Calibration and Evaluation Metrics.
Implements Temperature Scaling and Expected Calibration Error (ECE).
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across n_bins.
    probs: (N, C) predicted probabilities
    labels: (N,) true labels (0-indexed or 1-indexed)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.sum(in_bin) / total_samples

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class TemperatureScaler:
    """Finds optimal temperature T > 0 to calibrate model logits on validation data."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Optimizes temperature T on validation logits to minimize Negative Log Likelihood.
        logits: (N, C) float array
        labels: (N,) int array with class indices 0..C-1
        """
        def nll_objective(t: np.ndarray) -> float:
            scaled_logits = logits / t[0]
            # Numerically stable log-softmax
            max_l = np.max(scaled_logits, axis=1, keepdims=True)
            log_sum_exp = max_l + np.log(np.sum(np.exp(scaled_logits - max_l), axis=1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp
            # Select log prob of true labels
            nll = -np.mean(log_probs[np.arange(len(labels)), labels])
            return float(nll)

        res = minimize(nll_objective, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self.temperature

    def scale_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Applies fitted temperature to logits and returns calibrated probabilities."""
        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
