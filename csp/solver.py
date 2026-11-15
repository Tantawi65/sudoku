"""
Unified CSP Interface with switchable solvers:
- 'BT'    : Pure Backtracking
- 'FC'    : Backtracking + Forward Checking
- 'AC3'   : AC-3 Preprocessing + Backtracking + Forward Checking
Tracks assignments, backtracks, and solving success.
"""

import time
from typing import Dict, List, Optional, Tuple
from csp.sudoku_csp import SudokuCSP
from csp.backtracking import BacktrackingSolver
from csp.forward_checking import ForwardCheckingSolver
from csp.ac3 import AC3Solver


class UnifiedCSPSolver:
    """Unified runner allowing fair comparison across CSP propagation techniques."""

    def __init__(self, mode: str = "AC3"):
        if mode not in ("BT", "FC", "AC3"):
            raise ValueError(f"Unknown mode: {mode}. Choose from 'BT', 'FC', 'AC3'.")
        self.mode = mode
        self.assignments: int = 0
        self.backtracks: int = 0
        self.elapsed_time: float = 0.0

    def solve(self, csp: SudokuCSP, stop_at_one: bool = True) -> Optional[List[List[int]]]:
        """
        Solves the given SudokuCSP.
        Returns a completed 9x9 grid, or None if unsolvable.
        """
        start_time = time.perf_counter()
        working_csp = csp.copy()

        if self.mode == "BT":
            bt_solver = BacktrackingSolver()
            solutions = bt_solver.solve(working_csp, stop_at_one=stop_at_one)
            self.assignments = bt_solver.assignments
            self.backtracks = bt_solver.backtracks

        elif self.mode == "FC":
            fc_solver = ForwardCheckingSolver()
            solutions = fc_solver.solve(working_csp, use_mrv=True, stop_at_one=stop_at_one)
            self.assignments = fc_solver.assignments
            self.backtracks = fc_solver.backtracks

        elif self.mode == "AC3":
            ac3 = AC3Solver()
            is_consistent = ac3.enforce_arc_consistency(working_csp)
            if not is_consistent:
                self.assignments = 0
                self.backtracks = 0
                self.elapsed_time = time.perf_counter() - start_time
                return None

            fc_solver = ForwardCheckingSolver()
            solutions = fc_solver.solve(working_csp, use_mrv=True, stop_at_one=stop_at_one)
            self.assignments = fc_solver.assignments
            self.backtracks = fc_solver.backtracks

        self.elapsed_time = time.perf_counter() - start_time

        if not solutions:
            return None

        sol = solutions[0]
        grid = [[sol.get((r, c), 0) for c in range(9)] for r in range(9)]
        return grid

    def count_solutions(self, csp: SudokuCSP, max_count: int = 2) -> int:
        """Counts up to max_count solutions to verify puzzle uniqueness."""
        working_csp = csp.copy()
        fc_solver = ForwardCheckingSolver()
        solutions = fc_solver.solve(working_csp, use_mrv=True, max_solutions=max_count)
        return len(solutions)
