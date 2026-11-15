# Data Dictionary: Handwritten Sudoku Reader and Solver

## 1. Overview
This project combines real handwritten digit images from the MNIST database with synthetically generated Sudoku puzzle structures to simulate reading handwritten Sudoku grids. The data pipeline handles downloading, filtering (digits 1–9), validation splitting, Sudoku puzzle generation, and 9x9 image mosaic creation.

---

## 2. Dataset Components

### 2.1 Real Components (MNIST Database)
* **Source:** Modified National Institute of Standards and Technology (MNIST) dataset (`http://yann.lecun.com/exdb/mnist/`).
* **License & Access:** Public domain / CC BY-SA 3.0.
* **Original Dimensions:** Grayscale images of size $28 \times 28$ pixels.
* **Pixel Values:** Scaled from $[0, 255]$ integer values to $[0.0, 1.0]$ floating-point values via normalization: $x_{\text{norm}} = \frac{x}{255.0}$.
* **Classes Retained:** Digits $1$ through $9$ (digit $0$ is filtered out because standard Sudoku grids only contain symbols $1$ to $9$).
* **Splits:**
  * **Training Set:** 50,000 samples randomly partitioned from the official 60,000 training set using a fixed random seed (`seed = 42`).
  * **Validation Set:** 10,000 samples partitioned from the official training set.
  * **Test Set:** Official 10,000 test set left untouched for final model evaluation and test mosaic sampling.

| Field / Attribute | Type | Source | Value Range / Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image` | `float32[1, 28, 28]` | Real (MNIST) | $[0.0, 1.0]$ | Normalized single-channel grayscale handwritten digit image. |
| `label` | `int64` | Real (MNIST) | $\{1, 2, 3, 4, 5, 6, 7, 8, 9\}$ | Ground truth digit class label. |

---

### 2.2 Generated Components

#### A. Sudoku Puzzle Grids
* **Generation Procedure:**
  1. A complete, valid $9 \times 9$ Sudoku solution board is generated using randomized backtracking with constraint propagation.
  2. Numbers are removed one by one while verifying with a CSP solver that the remaining board maintains a unique valid solution.
  3. The number of givens (pre-filled clues) is parameterized (e.g., $25, 30, 35$ clues) to control puzzle difficulty.
* **Storage:** $9 \times 9$ 2D integer array where $0$ denotes an empty cell and $1 \le d \le 9$ represents a given clue.

#### B. Blank Cell Detection
* Blank cells (empty cells in the Sudoku grid) are identified using a deterministic pixel brightness / mean intensity threshold:
  $$\text{IsBlank}(I) = \begin{cases} \text{True} & \text{if } \frac{1}{28 \times 28} \sum_{i,j} I_{i,j} < \theta_{\text{blank}} \\ \text{False} & \text{otherwise} \end{cases}$$
  where $\theta_{\text{blank}} = 0.05$. Blank cells bypass the neural network entirely.

#### C. 9×9 Image Mosaics
* **Dimensions:** A $252 \times 252$ single-channel composite image consisting of an array of $9 \times 9$ patches, each of size $28 \times 28$.
* **Construction:**
  * For each cell $(r, c)$ in the Sudoku grid:
    * If cell is empty ($0$): Rendered as an empty/black patch ($28 \times 28$ zeros).
    * If cell has a clue $d \in \{1 \dots 9\}$: An image of digit $d$ is randomly sampled from the test partition of MNIST.

#### D. Injected Hard / Misread Digits
* To test the robustness of the AC-3 constraint checker and state-space correction search, misreads are intentionally injected into test puzzles:
  * **Low-Confidence Injection Rule:** For a chosen given cell $(r, c)$ containing digit $A$, substitute the image with an ambiguous handwritten image or swap the image with another digit $B$ that causes a row/column/box collision.
  * **Recorded Metadata:**
    * `cell`: Tuple `(row, col)`.
    * `true_digit`: Ground truth value from the valid puzzle solution.
    * `injected_digit`: Substituted digit creating a constraint conflict.
    * `confidence_gap`: Difference in softmax probability between the top prediction and the ground truth class.

---

## 3. Data Flow Diagram

```
[Raw MNIST Digits 0-9]
         │
         ▼ Filter out 0 & Normalize [0, 1]
[Filtered MNIST Digits 1-9] ────────► Split: Train (50k) / Val (10k) / Test (10k)
                                                 │
                                                 ▼
[Randomized Backtracking] ──► [Valid 9x9 Grids]  │
                                      │          │
                                      ▼          ▼
                           [Sample Clue Images from Test Set]
                                      │
                                      ▼
                           [Inject Ambiguous Digits (Misreads)]
                                      │
                                      ▼
                           [Final 9x9 Image Mosaics]
```
