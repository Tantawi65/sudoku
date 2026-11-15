# Handwritten Sudoku Reader and Solver
**Search + Constraint Satisfaction Problem (CSP) + Neural Network Integration**

---

## 1. Project Overview
This project implements an intelligent system capable of reading, verifying, correcting, and solving handwritten Sudoku puzzles from image inputs. When handwritten digits are misread by the neural network classifier, constraint conflicts arise in the Sudoku grid (such as duplicate numbers within the same row, column, or 3x3 block). The system handles this challenge by tightly coupling three core artificial intelligence paradigms:

1. **Neural Network (CNN):** Classifies handwritten digit patches (1–9) from the MNIST database and outputs calibrated softmax probability distributions over potential digits. Blank cells are filtered using an intensity threshold.
2. **CSP Engine (AC-3, Forward Checking, Backtracking):** Encodes the 81 cells as variables with binary inequality constraints. The AC-3 algorithm propagates constraints across arcs to detect dead ends or empty domains caused by misread clues.
3. **State-Space Search (A\*, UCS, Greedy):** If AC-3 flags an invalid or unsolvable configuration, state-space search explores candidate digit substitutions for low-confidence or conflicting cells, prioritizing changes that minimize the neural network probability penalty until a solvable puzzle is reached.

---

## 2. Directory Structure

```text
Project/
├── data/
│   ├── generator.py           # 9x9 Sudoku grid generation with uniqueness guarantees
│   ├── make_data.py           # MNIST download, 1-9 filtering, 50k/10k split, mosaic generation
│   ├── data_dictionary.md     # Detailed specification of real vs. generated data attributes
│   └── processed/             # Cached preprocessed MNIST arrays (created on first run)
├── nn/
│   ├── model.py               # PyTorch Convolutional Neural Network architecture
│   ├── baseline.py            # Logistic Regression baseline classifier
│   ├── calibrate.py           # Temperature scaling & Expected Calibration Error (ECE)
│   ├── train.py               # Training loop, validation calibration, and test benchmark
│   └── classifier.py          # High-level inference interface for 28x28 patches
├── csp/
│   ├── sudoku_csp.py          # Sudoku CSP representation (variables, domains, peers, constraints)
│   ├── backtracking.py        # Pure Backtracking search with assignment/backtrack counters
│   ├── forward_checking.py    # Backtracking + Forward Checking (FC)
│   ├── ac3.py                 # AC-3 Arc Consistency propagator and domain pruner
│   └── solver.py              # Unified switchable CSP solver interface ('BT', 'FC', 'AC3')
├── search/
│   ├── correction_search.py   # State-space definition, action branching, and admissible heuristic
│   ├── ucs.py                 # Uniform Cost Search (orders by cumulative probability cost)
│   ├── greedy.py              # Greedy Best-First Search (orders by conflict heuristic)
│   └── astar.py               # A* Search (orders by f(n) = g(n) + h(n))
├── integration/
│   └── pipeline.py            # End-to-end coordinator: NN -> AC-3 Check -> Search -> CSP Solve
├── eval/
│   ├── experiments.py         # Runs the 3 required experiments across 5 distinct seeds
│   ├── plot_results.py        # Generates charts and Markdown summary tables (mean ± std)
│   ├── plots/                 # Saved high-resolution figures (.png)
│   └── results/               # Raw experiment output (.json) and markdown tables (.md)
├── tests/
│   ├── test_csp_and_search.py # Small known-example verification for CSP and Search counters
│   └── test_suite.py          # Full test suite (5 normal, 3 edge, 2 impossible cases)
├── run_demo.py                # 5-minute interactive live demonstration for TA evaluation
├── requirements.txt           # Pinned dependency versions
└── README.md                  # Project documentation and execution instructions
```

---

## 3. Environment Setup

### 3.1 Requirements
* Python 3.8 or higher
* Recommended virtual environment:

```bash
# Clone or navigate to the repository directory
cd "e:/E-JUST Assignments/IS/Project"

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 4. Execution Commands

### Step 1: Run Unit Tests
To verify all components (CSP counters, search counters, 5 normal cases, 3 edge cases, and 2 impossible cases):
```bash
# Verify CSP and Search counters on small known examples
python -m unittest tests/test_csp_and_search.py

# Run the comprehensive 10-test suite
python -m unittest tests/test_suite.py
```

### Step 2: Download Data & Train Neural Network
To download MNIST, train the CNN (digits 1–9), fit temperature scaling calibration, and benchmark against the Logistic Regression baseline:
```bash
python -m nn.train
```

### Step 3: Run the 3 Experimental Studies
To execute Experiments 1, 2, and 3 across 5 random seeds (`[42, 101, 2024, 777, 999]`) and generate mean ± std tables:
```bash
# Run experiments and write eval/results/experiment_results.json
python -m eval.experiments

# Generate summary Markdown tables and PNG charts
python -m eval.plot_results
```

### Step 4: 5-Minute Live Demo (For TA Evaluation)
To launch the interactive live demonstration:
```bash
python run_demo.py
```

The demo menu allows triggering:
* **Option 1:** Full end-to-end pipeline (shows initial misread board, AC-3 conflict detection, A* correction search, and solved board).
* **Option 2:** Live CSP pruning comparison (`BT` vs `FC` vs `AC-3`), highlighting backtracks reduced by multiple orders of magnitude.
* **Option 3:** Neural Network guidance impact (`NN ON` vs `NN OFF / Random`), demonstrating why probability-guided search prevents combinatorial explosion.
* **Option 4:** Complete continuous walkthrough of all three parts.

---

## 5. Experimental Summary

* **Experiment 1 (Search Comparison):** Evaluates UCS vs. Greedy vs. A* over 0 to 4 injected misreads. A* and UCS achieve 100% recovery rates, while A* minimizes node expansions through the conflict-reduction heuristic.
* **Experiment 2 (CSP Pruning):** On 25-given puzzles, pure Backtracking requires an average of **27,160 backtracks**, whereas Forward Checking reduces backtracks to **95.4** (a 280x reduction).
* **Experiment 3 (Guidance Impact):** On puzzles with 2 misreads, NN-guided A* achieves **100% solution recovery** expanding an average of **3.8 nodes**, whereas unguided/random search achieves only **20% recovery** expanding over **276 nodes**.
