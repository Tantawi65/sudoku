"""
Logistic Regression Baseline Classifier for MNIST Digits 1-9.
Provides a standard linear classification benchmark against the CNN.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LogisticRegressionBaseline:
    """Logistic Regression baseline classifier on flattened 28x28 images."""

    def __init__(self, max_iter: int = 150, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=random_state,
            n_jobs=1
        )
        self.is_trained = False

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains logistic regression on flattened images.
        x_train: (N, 784) in range [0, 1]
        y_train: (N,) with classes 1-9
        """
        x_flat = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_flat, y_train)
        self.is_trained = True

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
        """Evaluates baseline accuracy and returns performance metrics."""
        if not self.is_trained:
            raise RuntimeError("Baseline model has not been trained yet.")

        x_flat = x_test.reshape(x_test.shape[0], -1)
        y_pred = self.model.predict(x_flat)
        y_prob = self.model.predict_proba(x_flat)

        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=list(range(1, 10)))
        report = classification_report(y_test, y_pred, labels=list(range(1, 10)), output_dict=True)

        return {
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "probabilities": y_prob
        }
