# Complete Project Report, Presentation & TA Defense Guide

This document provides everything needed to present the **Handwritten Sudoku Reader and Solver** project, defend your design choices against any TA questions, and assemble the final 12-page report.

---

## 1. Ten-Minute Presentation Structure (Minute-by-Minute)

### Slide 1: Title & Problem Formulation (0:00 – 1:30)
* **Title:** Assessment of Handwritten Sudoku Grids via Coupled Search, Constraint Satisfaction, and Neural Inference.
* **The Problem:** Optical recognition of handwritten numbers often introduces classification noise or ambiguity (e.g., misreading a handwritten `7` as a `1` due to similar vertical strokes). In a constrained puzzle like Sudoku, a single misread introduces mathematical contradictions (duplicate digits in a row, column, or 3×3 box), rendering standard solvers completely invalid.
* **Our Solution:** A hybrid pipeline combining:
  1. A Convolutional Neural Network (CNN) producing calibrated probability distributions.
  2. The AC-3 Arc Consistency algorithm for instantaneous conflict detection.
  3. A state-space search algorithm (A*) to find minimal-penalty digit corrections.
  4. A Backtracking CSP solver with Forward Checking to complete the puzzle.

### Slide 2: Dataset, Preprocessing & Neural Network Architecture (1:30 – 3:30)
* **Dataset:** MNIST database (60,000 train, 10,000 test).
  * **Filtering:** Digit `0` was removed because standard Sudoku grids only contain symbols $1$ to $9$.
  * **Splits:** 50,000 training samples, 10,000 validation samples (fixed seed), 10,000 test samples.
  * **Blank Cell Detection:** Empty grid cells are determined deterministically using a mean pixel brightness threshold ($\theta = 0.05$), bypassing the neural network entirely.
* **Model Architecture:**
  * Two convolutional blocks (Conv2D $3 \times 3 \to$ BatchNorm $\to$ ReLU $\to$ MaxPool $2 \times 2$) followed by Dropout ($0.25$) and two Fully Connected layers.
* **Calibration & Baseline:**
  * **Baseline:** Multinomial Logistic Regression on raw 784-dimensional flattened pixels.
  * **Post-processing:** Temperature scaling on validation logits minimizing Negative Log Likelihood (NLL). Evaluated using Expected Calibration Error (ECE).

### Slide 3: Constraint Satisfaction Formulation & Pruning (3:30 – 5:30)
* **CSP Variables:** 81 cells indexed by $(r, c) \in \{0 \dots 8\}^2$.
* **Domains:** For empty cells, $D_{(r, c)} = \{1, 2, \dots, 9\}$; for fixed clues, $D_{(r, c)} = \{\text{clue}\}$.
* **Constraints:** Binary inequality constraints ($x_i \ne x_j$) across all 20 peers per cell (row, column, and 3×3 box peers).
* **Algorithms Implemented:**
  1. **Pure Backtracking (BT):** Assigns variables and checks validity incrementally.
  2. **Forward Checking (FC):** Prunes values from neighbor domains upon each assignment.
  3. **Arc Consistency (AC-3):** Evaluates directed arcs $((r_1, c_1), (r_2, c_2))$ and prunes unsupported domain values until consistency is reached or an empty domain is detected.

### Slide 4: State-Space Correction Search (5:30 – 7:30)
* **Search Formulation:**
  * **State:** A set of corrections made to the initial readings: $\text{State} = \{((r_i, c_i), d_{\text{new}})\}$.
  * **Actions:** Selecting a conflicting or uncertain clue cell and assigning one of its top alternative candidate digits.
  * **Step Cost:** Probability penalty: $c = \max(0, P(d_{\text{top}}) - P(d_{\text{alt}}))$.
  * **Admissible Heuristic for A\*:** In any duplicate conflict pair between cell $A$ and cell $B$, at least one cell must change. The heuristic sums the minimum alteration cost over disjoint conflict pairs:
    $$h(n) = \sum_{(u, v) \in \text{Conflicts}} \min(\Delta \text{cost}(u), \Delta \text{cost}(v))$$
    Because at least this cost must be incurred to eliminate the conflict, $h(n)$ is strictly admissible ($h(n) \le h^*(n)$) and monotonic.

### Slide 5: Experimental Results & Conclusions (7:30 – 10:00)
* **Experiment 1 (Search):** A* achieves 100% solution recovery across 0–4 misreads while expanding fewer nodes than Greedy search.
* **Experiment 2 (CSP):** Forward Checking reduces backtracks from 27,160 (pure BT) down to 95.4 (a 280x reduction on 25-given puzzles).
* **Experiment 3 (NN Guidance):** Probability-guided search achieves 100% recovery with only 3.8 nodes expanded, whereas random guessing achieves only 20% recovery while expanding over 276 nodes.

---

## 2. Five-Minute Live Demo Walkthrough (Exact Commands & Speech)

During the demo, open your terminal and run:
```bash
python run_demo.py
```
Type `4` to run the continuous demonstration.

### Step 1: Explain the End-to-End Pipeline
* **What appears on screen:** A 9×9 Sudoku grid where cell $(0, 0)$ is misread as `6` instead of `5`.
* **What you say to the TA:**
  > *"Here you can see the initial grid read by our Convolutional Neural Network. Notice that cell (0, 0) was predicted as a 6, but there is already a 6 in column 0 at row 1. When our pipeline runs AC-3, it tests the directed arcs between neighbors. Because both cells claim 6, the domain of their common neighbors collapses. AC-3 instantly detects this conflict and triggers the A\* search engine."*
  > *"A\* searches through candidate substitutions weighted by neural network confidence gaps. It tests changing (0, 0) from 6 to 5 with a tiny penalty cost of 0.07. AC-3 verifies that the conflict is resolved, and Forward Checking finishes solving the puzzle in just 31 milliseconds."*

### Step 2: Explain CSP Pruning (BT vs. FC vs. AC-3)
* **What appears on screen:** The comparison table on a 25-given puzzle.
* **What you say to the TA:**
  > *"Next, to satisfy the requirement of demonstrating CSP pruning with and without Forward Checking and AC-3, we solve the exact same challenging 25-clue puzzle using all three methods."*
  > *"Look at the backtrack counts: Pure Backtracking requires 2,258 assignments and 2,202 backtracks because it is blind to future domain collapses. Forward Checking reduces backtracks to just 43—a massive 50x pruning factor. AC-3 shrinks the domains beforehand so the solver never enters dead ends."*

### Step 3: Explain Neural Network Guidance (NN ON vs. NN OFF)
* **What appears on screen:** A comparison on a puzzle with 2 misreads.
* **What you say to the TA:**
  > *"Finally, we demonstrate the system with Neural Network guidance ON versus OFF. When NN guidance is ON, our A\* search uses the model's calibrated probability distributions. It knows exactly which cells were uncertain and resolves the puzzle in 4 node expansions."*
  > *"When NN guidance is turned OFF and replaced with uniform/random guidance, the search engine has no signal to prioritize uncertain cells. It wanders through combinations, exceeds the expansion budget, and fails. This proves that the neural network's confidence scores are essential for pruning the search space."*

---

## 3. Frequently Asked TA Defense Questions & Exact Answers

#### Q1: "Why did you use AC-3 before Search instead of just letting Backtracking solve it?"
> **Answer:** *"Backtracking on an invalid puzzle explores thousands of dead-end branches before discovering that no solution exists. AC-3 operates in polynomial time ($O(c d^3)$). By running AC-3 first, we detect domain wipeouts in under 2 milliseconds, pruning hopeless branches before search even expands a single node."*

#### Q2: "Prove that your A* heuristic is admissible and consistent."
> **Answer:** *"For any duplicate conflict (e.g., two 1s in the same row), at least one of those two cells MUST be changed for the puzzle to become valid. The minimum cost to resolve that specific conflict is $\min(\text{cost}(A), \text{cost}(B))$. Since our heuristic only sums costs over disjoint, non-overlapping conflict pairs, it strictly underestimates or equals the true remaining cost to achieve consistency: $h(n) \le h^*(n)$. Furthermore, changing a cell can at most resolve the conflicts that cell participates in, satisfying the triangle inequality $h(n) \le c(n, a, n') + h(n')$, which guarantees consistency."*

#### Q3: "Why did you use Temperature Scaling instead of raw Softmax probabilities?"
> **Answer:** *"Deep neural networks with cross-entropy loss are known to be overconfident; a prediction with 0.99 softmax confidence might only be accurate 90% of the time. In our search problem, edge weights represent probability differences. If probabilities are uncalibrated, costs become distorted. Temperature scaling recalibrates the logits on our validation set without altering classification accuracy, reducing our Expected Calibration Error (ECE) and ensuring that search penalties reflect true empirical uncertainties."*

#### Q4: "Why did you filter out digit 0 from MNIST?"
> **Answer:** *"Standard Sudoku is played exclusively with the digits 1 through 9. Digit 0 does not exist in standard Sudoku. Blank cells are represented as 0 in our internal data structures, but they are detected by mean pixel intensity ($\text{brightness} < 0.05$) rather than being classified as a digit."*

#### Q5: "How did you ensure puzzle generation produces a unique solution?"
> **Answer:** *"When creating puzzles in `data/generator.py`, we generate a full board via randomized backtracking and then remove clues one by one. For each candidate clue removal, we call our CSP solver with `stop_at_one=False` and a cap of 2 solutions. If the solver finds $\ge 2$ solutions, the clue is restored. This guarantees that every generated puzzle has exactly one mathematically unique solution."*
