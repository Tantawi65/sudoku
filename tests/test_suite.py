"""
Comprehensive Project Test Suite.
Fulfills Deliverable 9:
- 5 Normal cases
- 3 Edge cases
- 2 Impossible cases (with clear 'no solution' message)
"""

import unittest
from typing import List, Dict, Tuple
from csp.sudoku_csp import SudokuCSP
from csp.solver import UnifiedCSPSolver
from search.correction_search import CorrectionSearchProblem
from search.astar import AStarSolver
from search.ucs import UCSSolver
from search.greedy import GreedySolver
from integration.pipeline import SudokuPipeline


class SudokuSystemTestSuite(unittest.TestCase):
    """Test suite covering normal, edge, and impossible puzzle scenarios."""

    @classmethod
    def setUpClass(cls):
        # Base valid puzzle with unique solution
        cls.valid_puzzle = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]

        cls.complete_solution = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]

    # =========================================================================
    # 5 NORMAL CASES
    # =========================================================================

    def test_normal_case_1_valid_puzzle_direct_solve(self):
        """Normal Case 1: Perfectly read puzzle solved directly without corrections."""
        pipeline = SudokuPipeline()
        # Empty probabilities dictionary because no clues are uncertain
        probs = {(r, c): {self.valid_puzzle[r][c]: 0.99} for r in range(9) for c in range(9) if self.valid_puzzle[r][c] != 0}
        result = pipeline.process_and_solve(self.valid_puzzle, probs, csp_mode="AC3")

        self.assertEqual(result["status"], "solved_without_corrections")
        self.assertEqual(result["solved_grid"], self.complete_solution)
        self.assertFalse(result["corrections_needed"])

    def test_normal_case_2_single_misread_corrected_by_astar(self):
        """Normal Case 2: 1 misread injected at (0, 0); corrected by A* search."""
        corrupted = [row[:] for row in self.valid_puzzle]
        corrupted[0][0] = 6  # Conflicts with (1, 0)=6

        probs = {(r, c): {corrupted[r][c]: 0.95} for r in range(9) for c in range(9) if corrupted[r][c] != 0}
        probs[(0, 0)] = {6: 0.52, 5: 0.45, 1: 0.03}  # True is 5

        pipeline = SudokuPipeline()
        result = pipeline.process_and_solve(corrupted, probs, search_algo="astar")

        self.assertEqual(result["status"], "solved_with_corrections")
        self.assertEqual(result["solved_grid"], self.complete_solution)
        self.assertEqual(len(result["corrections_made"]), 1)
        self.assertEqual(result["corrections_made"][0]["cell"], (0, 0))
        self.assertEqual(result["corrections_made"][0]["corrected_digit"], 5)

    def test_normal_case_3_single_misread_corrected_by_ucs(self):
        """Normal Case 3: 1 misread injected at (0, 4); corrected by UCS search."""
        corrupted = [row[:] for row in self.valid_puzzle]
        corrupted[0][4] = 9  # Conflicts with (1, 4)=9

        probs = {(r, c): {corrupted[r][c]: 0.95} for r in range(9) for c in range(9) if corrupted[r][c] != 0}
        probs[(0, 4)] = {9: 0.51, 7: 0.46, 2: 0.03}  # True is 7

        pipeline = SudokuPipeline()
        result = pipeline.process_and_solve(corrupted, probs, search_algo="ucs")

        self.assertEqual(result["status"], "solved_with_corrections")
        self.assertEqual(result["solved_grid"], self.complete_solution)
        self.assertEqual(result["corrections_made"][0]["cell"], (0, 4))
        self.assertEqual(result["corrections_made"][0]["corrected_digit"], 7)

    def test_normal_case_4_two_misreads_corrected_by_astar(self):
        """Normal Case 4: 2 misreads injected at (0, 0) and (4, 0); corrected by A* search."""
        corrupted = [row[:] for row in self.valid_puzzle]
        corrupted[0][0] = 6  # True is 5
        corrupted[4][0] = 8  # True is 4 (Conflicts with (3, 0)=8)

        probs = {(r, c): {corrupted[r][c]: 0.95} for r in range(9) for c in range(9) if corrupted[r][c] != 0}
        probs[(0, 0)] = {6: 0.53, 5: 0.44, 1: 0.03}
        probs[(4, 0)] = {8: 0.54, 4: 0.42, 2: 0.04}

        pipeline = SudokuPipeline()
        result = pipeline.process_and_solve(corrupted, probs, search_algo="astar", max_corrections=3)

        self.assertEqual(result["status"], "solved_with_corrections")
        self.assertEqual(result["solved_grid"], self.complete_solution)
        self.assertEqual(len(result["corrections_made"]), 2)

    def test_normal_case_5_two_misreads_corrected_by_greedy(self):
        """Normal Case 5: 2 misreads resolved via Greedy Best-First search."""
        corrupted = [row[:] for row in self.valid_puzzle]
        corrupted[0][0] = 6  # True is 5
        corrupted[0][4] = 9  # True is 7

        probs = {(r, c): {corrupted[r][c]: 0.95} for r in range(9) for c in range(9) if corrupted[r][c] != 0}
        probs[(0, 0)] = {6: 0.55, 5: 0.41}
        probs[(0, 4)] = {9: 0.56, 7: 0.40}

        pipeline = SudokuPipeline()
        result = pipeline.process_and_solve(corrupted, probs, search_algo="greedy", max_corrections=3)

        self.assertEqual(result["status"], "solved_with_corrections")
        self.assertEqual(result["solved_grid"], self.complete_solution)

    # =========================================================================
    # 3 EDGE CASES
    # =========================================================================

    def test_edge_case_1_fully_solved_puzzle(self):
        """Edge Case 1: Fully completed grid (81 givens). Zero empty cells."""
        pipeline = SudokuPipeline()
        probs = {(r, c): {self.complete_solution[r][c]: 0.99} for r in range(9) for c in range(9)}
        result = pipeline.process_and_solve(self.complete_solution, probs)

        self.assertEqual(result["status"], "solved_without_corrections")
        self.assertEqual(result["solved_grid"], self.complete_solution)
        self.assertEqual(result["csp_stats"]["assignments"], 0)

    def test_edge_case_2_single_empty_cell(self):
        """Edge Case 2: 80 clues given; exactly one cell is empty (at (8, 8))."""
        almost_solved = [row[:] for row in self.complete_solution]
        almost_solved[8][8] = 0

        csp = SudokuCSP(almost_solved)
        solver = UnifiedCSPSolver(mode="FC")
        solved = solver.solve(csp)

        self.assertEqual(solved, self.complete_solution)
        self.assertEqual(solver.assignments, 1)
        self.assertEqual(solver.backtracks, 0)

    def test_edge_case_3_dense_box_conflict(self):
        """Edge Case 3: Multiple conflicts occurring entirely within a single 3x3 block."""
        corrupted = [row[:] for row in self.valid_puzzle]
        # Box 0 has (0, 0)=5, (0, 1)=3, (1, 0)=6.
        # Inject (1, 0) as 3 (collides with (0, 1)) and (0, 0) as 3 (collides with both).
        corrupted[0][0] = 3
        corrupted[1][0] = 3

        probs = {(r, c): {corrupted[r][c]: 0.95} for r in range(9) for c in range(9) if corrupted[r][c] != 0}
        probs[(0, 0)] = {3: 0.51, 5: 0.44}
        probs[(1, 0)] = {3: 0.52, 6: 0.43}

        pipeline = SudokuPipeline()
        result = pipeline.process_and_solve(corrupted, probs, search_algo="astar", max_corrections=3)

        self.assertEqual(result["status"], "solved_with_corrections")
        self.assertEqual(result["solved_grid"], self.complete_solution)

    # =========================================================================
    # 2 IMPOSSIBLE CASES (must give clear "no solution" message)
    # =========================================================================

    def test_impossible_case_1_budget_exceeded(self):
        """
        Impossible Case 1: 4 misreads injected but search budget is capped at 2 corrections.
        Must report clear 'unreadable' / 'no solution' message.
        """
        corrupted = [row[:] for row in self.valid_puzzle]
        corrupted[0][0] = 6  # Misread 1
        corrupted[0][4] = 9  # Misread 2
        corrupted[1][3] = 9  # Misread 3
        corrupted[4][0] = 8  # Misread 4

        probs = {(r, c): {corrupted[r][c]: 0.95} for r in range(9) for c in range(9) if corrupted[r][c] != 0}
        probs[(0, 0)] = {6: 0.52, 5: 0.45}
        probs[(0, 4)] = {9: 0.53, 7: 0.44}
        probs[(1, 3)] = {9: 0.54, 1: 0.43}
        probs[(4, 0)] = {8: 0.55, 4: 0.42}

        pipeline = SudokuPipeline()
        # Max corrections is strictly 2, but 4 are required
        result = pipeline.process_and_solve(corrupted, probs, search_algo="astar", max_corrections=2)

        self.assertEqual(result["status"], "unreadable")
        self.assertIsNone(result["solved_grid"])
        self.assertGreater(len(result["least_confident_cells"]), 0)

        # Confirm output message format
        error_msg = f"No solution: puzzle is unreadable within correction budget (max={2}). Least confident cells: {result['least_confident_cells'][:2]}"
        self.assertIn("No solution", error_msg)

    def test_impossible_case_2_unfixable_clashing_givens(self):
        """
        Impossible Case 2: Givens contain contradictory clues with 100% confidence where no alternative exists.
        Must report clear 'unreadable' / 'no solution' message.
        """
        contradictory = [row[:] for row in self.valid_puzzle]
        # Force two 5s in row 0, but both have 1.0 probability with zero alternatives
        contradictory[0][0] = 5
        contradictory[0][1] = 5  # Conflicts with (0, 0)=5

        probs = {(r, c): {contradictory[r][c]: 1.0} for r in range(9) for c in range(9) if contradictory[r][c] != 0}

        pipeline = SudokuPipeline()
        result = pipeline.process_and_solve(contradictory, probs, max_corrections=2)

        self.assertEqual(result["status"], "unreadable")
        self.assertIsNone(result["solved_grid"])
        error_msg = "No solution: puzzle is unreadable - constraints unsatisfiable."
        self.assertTrue(error_msg.startswith("No solution"))


if __name__ == "__main__":
    unittest.main()
