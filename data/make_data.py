"""
Data Pipeline and Dataset Preparation.
Downloads MNIST, filters digits 1-9 (drops 0), performs 50k/10k validation split,
and generates test Sudoku puzzles with 9x9 image mosaics and controlled injected misreads.
"""

import os
import random
import gzip
import requests
import numpy as np
from typing import Dict, List, Tuple
from data.generator import SudokuGenerator

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist(raw_dir: str) -> None:
    """Downloads raw MNIST gzip files from official PyTorch AWS mirror."""
    os.makedirs(raw_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    for filename, url in MNIST_URLS.items():
        filepath = os.path.join(raw_dir, f"{filename}.gz")
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
            tmp_path = filepath + ".tmp"
            print(f"Downloading {filename}...", flush=True)
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=128 * 1024):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp_path, filepath)
            print(f"  Finished {filename} ({os.path.getsize(filepath)} bytes)", flush=True)


def parse_mnist_images(gz_path: str) -> np.ndarray:
    """Reads MNIST image file from gzip into numpy float32 array normalized to [0, 1]."""
    with gzip.open(gz_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28).astype(np.float32) / 255.0


def parse_mnist_labels(gz_path: str) -> np.ndarray:
    """Reads MNIST label file from gzip into numpy int64 array."""
    with gzip.open(gz_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.int64)


def prepare_mnist(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downloads, filters digits 1-9 (drops digit 0),
    splits training into 50k train / 10k validation,
    and saves to processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)
    out_file = os.path.join(processed_dir, "mnist_filtered.npz")

    if os.path.exists(out_file):
        data = np.load(out_file)
        print(f"Dataset already prepared: {out_file}", flush=True)
        print(f"  Train samples: {data['x_train'].shape[0]} (digits 1-9)", flush=True)
        print(f"  Val samples:   {data['x_val'].shape[0]}", flush=True)
        print(f"  Test samples:  {data['x_test'].shape[0]}", flush=True)
        return (
            data["x_train"], data["y_train"],
            data["x_val"], data["y_val"],
            data["x_test"], data["y_test"]
        )

    download_mnist(raw_dir)

    print("Parsing MNIST raw files...", flush=True)
    x_train_raw = parse_mnist_images(os.path.join(raw_dir, "train_images.gz"))
    y_train_raw = parse_mnist_labels(os.path.join(raw_dir, "train_labels.gz"))
    x_test_raw = parse_mnist_images(os.path.join(raw_dir, "test_images.gz"))
    y_test_raw = parse_mnist_labels(os.path.join(raw_dir, "test_labels.gz"))

    # Filter out digit 0 (keep only digits 1 to 9)
    train_mask = y_train_raw > 0
    x_train_f = x_train_raw[train_mask]
    y_train_f = y_train_raw[train_mask]

    test_mask = y_test_raw > 0
    x_test = x_test_raw[test_mask]
    y_test = y_test_raw[test_mask]

    # Deterministic split: 50,000 train / remaining validation
    np.random.seed(seed)
    indices = np.random.permutation(len(x_train_f))
    train_size = min(50000, int(len(x_train_f) * 0.8333))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    x_train, y_train = x_train_f[train_idx], y_train_f[train_idx]
    x_val, y_val = x_train_f[val_idx], y_train_f[val_idx]

    np.savez_compressed(
        out_file,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test
    )
    print(f"Saved filtered dataset: Train={x_train.shape[0]}, Val={x_val.shape[0]}, Test={x_test.shape[0]}", flush=True)
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_digit_mosaic(
    puzzle_grid: List[List[int]],
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Renders a 9x9 image mosaic (252x252) from a Sudoku grid using test MNIST images.
    Blank cells (0) are filled with zeros.
    Returns (mosaic_image, clue_ground_truth).
    """
    random.seed(seed)
    np.random.seed(seed)

    digit_pool: Dict[int, List[int]] = {d: [] for d in range(1, 10)}
    for idx, label in enumerate(y_test):
        digit_pool[int(label)].append(idx)

    mosaic = np.zeros((252, 252), dtype=np.float32)
    clues = {}

    for r in range(9):
        for c in range(9):
            val = puzzle_grid[r][c]
            if val in range(1, 10):
                clues[(r, c)] = val
                sample_idx = random.choice(digit_pool[val])
                img = x_test[sample_idx]
                mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img

    return mosaic, clues


def inject_misreads(
    mosaic: np.ndarray,
    clues: Dict[Tuple[int, int], int],
    puzzle_grid: List[List[int]],
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_misreads: int = 1,
    seed: int = 42
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """
    Injects `num_misreads` ambiguous/colliding digit images into the mosaic.
    Replaces a clue image with another digit's image that collides in row/col/box.
    """
    random.seed(seed)
    corrupted_mosaic = np.copy(mosaic)
    clue_keys = list(clues.keys())
    random.shuffle(clue_keys)

    digit_pool = {d: np.where(y_test == d)[0] for d in range(1, 10)}

    injected_logs = []
    injected_count = 0

    for r, c in clue_keys:
        if injected_count >= num_misreads:
            break

        true_val = clues[(r, c)]
        candidates = []
        for i in range(9):
            if i != c and puzzle_grid[r][i] != 0:
                candidates.append(puzzle_grid[r][i])
            if i != r and puzzle_grid[i][c] != 0:
                candidates.append(puzzle_grid[i][c])

        colliding_digit = candidates[0] if candidates else ((true_val % 9) + 1)

        sample_idx = random.choice(digit_pool[colliding_digit])
        replacement_img = x_test[sample_idx]
        corrupted_mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = replacement_img

        injected_logs.append({
            "cell": (r, c),
            "true_digit": true_val,
            "injected_digit": colliding_digit
        })
        injected_count += 1

    return corrupted_mosaic, injected_logs


if __name__ == "__main__":
    prepare_mnist()
