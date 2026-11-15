"""
Neural Network Training & Baseline Benchmark Pipeline.
Trains the digit classification neural network on MNIST (digits 1-9),
calibrates output probabilities using temperature scaling,
and benchmarks performance against the Logistic Regression baseline.
"""

import os
import json
import time
from typing import Dict, Tuple
import numpy as np
import joblib

from data.make_data import prepare_mnist
from nn.model import DigitNeuralNetwork
from nn.baseline import LogisticRegressionBaseline
from nn.calibrate import TemperatureScaler, compute_ece
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def run_training_pipeline() -> Dict[str, object]:
    """Coordinates full training, temperature scaling calibration, and baseline comparison."""
    print("=" * 60)
    print("NEURAL NETWORK TRAINING AND CALIBRATION PIPELINE")
    print("=" * 60)

    # 1. Load data partitions
    print("\n[Step 1] Loading preprocessed MNIST partitions (digits 1-9)...")
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_mnist()
    print(f"  Training samples:   {x_train.shape[0]}")
    print(f"  Validation samples: {x_val.shape[0]}")
    print(f"  Test samples:       {x_test.shape[0]}")

    # 2. Train Neural Network
    print("\n[Step 2] Training Neural Network (784 -> 256 -> 128 -> 9, ReLU, Adam)...")
    nn_model = DigitNeuralNetwork(hidden_layer_sizes=(256, 128), random_state=42)
    start_train = time.perf_counter()
    nn_model.train(x_train, y_train)
    train_duration = time.perf_counter() - start_train
    print(f"  Neural Network training completed in {train_duration:.2f} seconds.")

    # 3. Calibrate on Validation Set
    print("\n[Step 3] Calibrating Probabilities on Validation Partition via Temperature Scaling...")
    val_logits = nn_model.predict_logits(x_val)
    # Map validation labels 1-9 to 0-8 for calibration NLL
    val_labels_idx = y_val - 1

    scaler = TemperatureScaler()
    opt_temp = scaler.fit(val_logits, val_labels_idx)
    nn_model.set_temperature(opt_temp)
    print(f"  Optimal Temperature Parameter T = {opt_temp:.3f}")

    # 4. Evaluate Neural Network on Test Set
    print("\n[Step 4] Evaluating Neural Network on Unseen Test Partition...")
    test_logits = nn_model.predict_logits(x_test)
    test_labels_idx = y_test - 1

    # Uncalibrated probabilities (T = 1.0)
    uncal_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
    uncal_probs /= np.sum(uncal_probs, axis=1, keepdims=True)

    # Calibrated probabilities (T = opt_temp)
    cal_probs = scaler.scale_probabilities(test_logits)

    nn_preds = np.argmax(cal_probs, axis=1) + 1  # 1-9
    nn_acc = accuracy_score(y_test, nn_preds)
    ece_uncal = compute_ece(uncal_probs, test_labels_idx)
    ece_cal = compute_ece(cal_probs, test_labels_idx)
    cm_nn = confusion_matrix(y_test, nn_preds, labels=list(range(1, 10)))

    print(f"  Test Accuracy:             {nn_acc * 100:.2f}%")
    print(f"  Uncalibrated ECE:         {ece_uncal:.4f}")
    print(f"  Calibrated ECE:           {ece_cal:.4f} (ECE reduced by {max(0, (ece_uncal - ece_cal) / ece_uncal * 100):.1f}%)")

    # 5. Benchmark against Logistic Regression Baseline
    print("\n[Step 5] Benchmarking against Logistic Regression Baseline...")
    baseline = LogisticRegressionBaseline(max_iter=150, random_state=42)
    start_base = time.perf_counter()
    baseline.train(x_train, y_train)
    base_duration = time.perf_counter() - start_base
    base_metrics = baseline.evaluate(x_test, y_test)
    base_acc = base_metrics["accuracy"]
    base_ece = compute_ece(base_metrics["probabilities"], test_labels_idx)

    print(f"  Baseline Accuracy:        {base_acc * 100:.2f}%")
    print(f"  Baseline ECE:             {base_ece:.4f}")
    print(f"  Baseline training time:   {base_duration:.2f} seconds")

    # 6. Save Model Artifacts
    os.makedirs("nn/weights", exist_ok=True)
    weights_path = "nn/weights/digit_nn.joblib"
    joblib.dump({
        "model": nn_model.model,
        "temperature": opt_temp
    }, weights_path)
    print(f"\nSaved trained model and calibration parameters to: {weights_path}")

    # Summary JSON
    results = {
        "neural_network": {
            "accuracy": float(nn_acc),
            "ece_uncalibrated": float(ece_uncal),
            "ece_calibrated": float(ece_cal),
            "optimal_temperature": float(opt_temp),
            "training_time_seconds": float(train_duration),
            "confusion_matrix": cm_nn.tolist()
        },
        "baseline_logistic_regression": {
            "accuracy": float(base_acc),
            "ece": float(base_ece),
            "training_time_seconds": float(base_duration),
            "confusion_matrix": base_metrics["confusion_matrix"].tolist()
        }
    }
    results_path = "nn/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {results_path}")

    print("\n" + "=" * 60)
    print(f"{'Model':<30} | {'Accuracy':<12} | {'ECE':<10}")
    print("-" * 60)
    print(f"{'Neural Network (Calibrated)':<30} | {nn_acc * 100:.2f}%{'':<6} | {ece_cal:.4f}")
    print(f"{'Logistic Regression Baseline':<30} | {base_acc * 100:.2f}%{'':<6} | {base_ece:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_training_pipeline()
