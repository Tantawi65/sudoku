"""
Small Example Validation for Search and CSP.
Tests the search and CSP implementations on small known examples with verified solutions.
"""

import unittest
from csp.sudoku_csp import SudokuCSP
from csp.solver import UnifiedCSPSolver
from csp.ac3 import run_ac3
from search.correction_search import CorrectionSearchProblem
from search.ucs import UCSSolver
from search.greedy import GreedySolver
from search.astar import AStarSolver


class TestCSPAndSearchSmallExamples(unittest.TestCase):
    """Verifies CSP and Search counters and correctness on small known instances."""

    def setUp(self):
        # Known valid 9x9 puzzle (0 denotes empty cell)
        self.known_puzzle = [
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

        self.expected_solution = [
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

    def test_csp_solvers_and_counters(self):
        """Test BT, FC, and AC-3 on known Sudoku puzzle."""
        # 1. Pure Backtracking
        csp_bt = SudokuCSP(self.known_puzzle)
        solver_bt = UnifiedCSPSolver(mode="BT")
        sol_bt = solver_bt.solve(csp_bt)
        self.assertEqual(sol_bt, self.expected_solution)
        self.assertGreater(solver_bt.assignments, 0)

        # 2. Forward Checking
        csp_fc = SudokuCSP(self.known_puzzle)
        solver_fc = UnifiedCSPSolver(mode="FC")
        sol_fc = solver_fc.solve(csp_fc)
        self.assertEqual(sol_fc, self.expected_solution)
        self.assertGreater(solver_fc.assignments, 0)
        # FC should be significantly more efficient than pure BT in backtracks
        self.assertLessEqual(solver_fc.backtracks, solver_bt.backtracks)

        # 3. AC-3 + FC
        csp_ac3 = SudokuCSP(self.known_puzzle)
        solver_ac3 = UnifiedCSPSolver(mode="AC3")
        sol_ac3 = solver_ac3.solve(csp_ac3)
        self.assertEqual(sol_ac3, self.expected_solution)

    def test_ac3_immediate_conflict_detection(self):
        """Test that AC-3 immediately detects conflicting givens."""
        conflicting_puzzle = [row[:] for row in self.known_puzzle]
        # Introduce duplicate 5 in row 0 at column 1 (row 0 now has 5 at col 0 and 5 at col 1)
        conflicting_puzzle[0][1] = 5
        csp = SudokuCSP(conflicting_puzzle)
        is_consistent, _, _ = run_ac3(csp)
        self.assertFalse(is_consistent)

    def test_search_on_known_misread(self):
        """
        Test UCS, Greedy, and A* on a known puzzle where cell (0, 0)
        was misread as 6 (causing conflict with (1, 0)=6),
        with true digit 5 as the second alternative.
        """
        corrupted_puzzle = [row[:] for row in self.known_puzzle]
        corrupted_puzzle[0][0] = 6  # Conflict with (1, 0)=6 in column 0

        # Simulate NN probabilities for cell (0, 0): top is 6 (0.55), alt is 5 (0.40)
        probabilities = {
            (0, 0): {6: 0.55, 5: 0.40, 3: 0.05}
        }

        problem = CorrectionSearchProblem(
            initial_grid=corrupted_puzzle,
            probabilities=probabilities,
            max_corrections=2
        )

        # 1. UCS
        ucs = UCSSolver()
        success_ucs, node_ucs, sol_ucs = ucs.search(problem)
        self.assertTrue(success_ucs)
        self.assertEqual(node_ucs.corrections, (((0, 0), 5),))
        self.assertEqual(sol_ucs, self.expected_solution)
        self.assertGreater(ucs.nodes_expanded, 0)
        self.assertGreater(ucs.nodes_generated, 0)

        # 2. Greedy
        greedy = GreedySolver()
        success_greedy, node_greedy, sol_greedy = greedy.search(problem)
        self.assertTrue(success_greedy)
        self.assertEqual(node_greedy.corrections, (((0, 0), 5),))
        self.assertEqual(sol_greedy, self.expected_solution)
        self.assertGreater(greedy.nodes_expanded, 0)
        self.assertGreater(greedy.nodes_generated, 0)

        # 3. A*
        astar = AStarSolver()
        success_astar, node_astar, sol_astar = astar.search(problem)
        self.assertTrue(success_astar)
        self.assertEqual(node_astar.corrections, (((0, 0), 5),))
        self.assertEqual(sol_astar, self.expected_solution)
        self.assertGreater(astar.nodes_expanded, 0)
        self.assertGreater(astar.nodes_generated, 0)


if __name__ == "__main__":
    unittest.main()
