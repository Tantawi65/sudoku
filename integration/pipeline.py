"""
Integration Pipeline: Sudoku Reader and Solver.
Glues together:
1. Neural Network reading with brightness-based blank detection
2. AC-3 conflict detection
3. State-space correction search (A*, UCS, Greedy)
4. CSP solver (Backtracking with Forward Checking and AC-3)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from csp.sudoku_csp import SudokuCSP
from csp.ac3 import run_ac3
from csp.solver import UnifiedCSPSolver
from search.correction_search import CorrectionSearchProblem, SearchNode
from search.ucs import UCSSolver
from search.greedy import GreedySolver
from search.astar import AStarSolver


class SudokuPipeline:
    """End-to-end integrated solver coordinating NN, Search, and CSP modules."""

    def __init__(self, nn_model=None, blank_threshold: float = 0.05):
        self.nn_model = nn_model
        self.blank_threshold = blank_threshold

    def is_blank_patch(self, patch: np.ndarray) -> bool:
        """Determines if a 28x28 patch is empty via mean pixel intensity threshold."""
        return float(np.mean(patch)) < self.blank_threshold

    def parse_mosaic(
        self,
        mosaic: np.ndarray
    ) -> Tuple[List[List[int]], Dict[Tuple[int, int], Dict[int, float]], List[Tuple[int, int]]]:
        """
        Parses a 9x9 composite mosaic of 28x28 digit images.
        mosaic: (252, 252) or (9, 9, 28, 28) numpy array with values in [0, 1].
        Returns:
            initial_grid: 9x9 int matrix
            probabilities: dict of (r, c) -> {digit: prob}
            blank_cells: list of (r, c)
        """
        initial_grid = [[0 for _ in range(9)] for _ in range(9)]
        probabilities: Dict[Tuple[int, int], Dict[int, float]] = {}
        blank_cells: List[Tuple[int, int]] = []

        # Convert 252x252 image into 9x9 grid of 28x28 patches if necessary
        if mosaic.shape == (252, 252):
            patches = np.zeros((9, 9, 28, 28), dtype=np.float32)
            for r in range(9):
                for c in range(9):
                    patches[r, c] = mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28]
        else:
            patches = mosaic

        for r in range(9):
            for c in range(9):
                patch = patches[r, c]
                if self.is_blank_patch(patch):
                    initial_grid[r][c] = 0
                    blank_cells.append((r, c))
                else:
                    if self.nn_model is not None:
                        # Feed to neural network
                        probs_dict = self.nn_model.predict_patch(patch)
                        top_digit = max(probs_dict.keys(), key=lambda d: probs_dict[d])
                        initial_grid[r][c] = top_digit
                        probabilities[(r, c)] = probs_dict
                    else:
                        raise RuntimeError("Neural network model is required for non-blank digit inference.")

        return initial_grid, probabilities, blank_cells

    def process_and_solve(
        self,
        initial_grid: List[List[int]],
        probabilities: Dict[Tuple[int, int], Dict[int, float]],
        search_algo: str = "astar",
        csp_mode: str = "AC3",
        max_corrections: int = 3,
        require_unique: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline:
        1. Checks initial grid with AC-3
        2. If invalid/conflicted, invokes Search (A*, UCS, or Greedy)
        3. Invokes CSP to complete the puzzle
        Supports minimum version (solvable goal) and optional extension (uniqueness goal).
        """
        result: Dict[str, Any] = {
            "initial_grid": [row[:] for row in initial_grid],
            "is_initially_valid": False,
            "corrections_needed": False,
            "corrections_made": [],
            "solved_grid": None,
            "status": "unsolved",
            "search_stats": {},
            "csp_stats": {},
            "least_confident_cells": []
        }

        # Check initial grid consistency with AC-3
        csp_check = SudokuCSP(initial_grid)
        is_consistent, _, _ = run_ac3(csp_check)

        if is_consistent and not csp_check.has_immediate_conflict():
            # Try direct CSP solve
            solver = UnifiedCSPSolver(mode=csp_mode)
            if require_unique:
                if solver.count_solutions(SudokuCSP(initial_grid), max_count=2) == 1:
                    solved = solver.solve(SudokuCSP(initial_grid))
                else:
                    solved = None
            else:
                solved = solver.solve(SudokuCSP(initial_grid))

            if solved is not None:
                result["is_initially_valid"] = True
                result["solved_grid"] = solved
                result["status"] = "solved_without_corrections"
                result["csp_stats"] = {
                    "assignments": solver.assignments,
                    "backtracks": solver.backtracks,
                    "time": solver.elapsed_time
                }
                return result

        # Inconsistency detected: Trigger correction search
        result["corrections_needed"] = True
        problem = CorrectionSearchProblem(
            initial_grid=initial_grid,
            probabilities=probabilities,
            max_corrections=max_corrections,
            require_unique=require_unique
        )

        if search_algo.lower() == "ucs":
            searcher = UCSSolver()
        elif search_algo.lower() == "greedy":
            searcher = GreedySolver()
        else:
            searcher = AStarSolver()

        success, best_node, solved_grid = searcher.search(problem)

        result["search_stats"] = {
            "algorithm": search_algo,
            "nodes_expanded": searcher.nodes_expanded,
            "nodes_generated": searcher.nodes_generated,
            "elapsed_time": searcher.elapsed_time
        }

        if success and best_node is not None and solved_grid is not None:
            result["status"] = "solved_with_corrections"
            result["solved_grid"] = solved_grid
            result["corrections_made"] = [
                {
                    "cell": cell,
                    "original_digit": initial_grid[cell[0]][cell[1]],
                    "corrected_digit": new_val,
                    "step_cost": max(0.0, max(probabilities[cell].values()) - probabilities[cell].get(new_val, 0.0))
                }
                for cell, new_val in best_node.corrections
            ]
        else:
            result["status"] = "unreadable"
            # Identify the least confident cells to report to the user
            sorted_uncertain = sorted(
                probabilities.keys(),
                key=lambda cell: max(probabilities[cell].values())
            )
            result["least_confident_cells"] = [
                {
                    "cell": cell,
                    "read_digit": initial_grid[cell[0]][cell[1]],
                    "confidence": max(probabilities[cell].values())
                }
                for cell in sorted_uncertain[:5]
            ]

        return result
