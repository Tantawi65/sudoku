"""
Pure Backtracking Search for Sudoku CSP.
Does not perform forward domain pruning; checks consistency at each assignment.
"""

from typing import Dict, List, Optional, Tuple
from csp.sudoku_csp import SudokuCSP


class BacktrackingSolver:
    """Solves Sudoku using standard Backtracking Search with assignment & backtrack counters."""

    def __init__(self):
        self.assignments: int = 0
        self.backtracks: int = 0

    def reset_counters(self) -> None:
        self.assignments = 0
        self.backtracks = 0

    def solve(self, csp: SudokuCSP, stop_at_one: bool = True) -> List[Dict[Tuple[int, int], int]]:
        """
        Runs pure backtracking search.
        Returns a list of valid complete assignments.
        """
        self.reset_counters()
        assignment: Dict[Tuple[int, int], int] = {}

        # Fixed clues are part of the initial assignment
        for var in csp.variables:
            if len(csp.domains[var]) == 1:
                val = next(iter(csp.domains[var]))
                assignment[var] = val

        # Check if initial clues are already conflicting
        for var, val in assignment.items():
            for peer in csp.peers[var]:
                if peer in assignment and assignment[peer] == val:
                    return []

        solutions: List[Dict[Tuple[int, int], int]] = []
        self._backtrack(csp, assignment, solutions, stop_at_one)
        return solutions

    def _select_unassigned_variable(
        self,
        csp: SudokuCSP,
        assignment: Dict[Tuple[int, int], int]
    ) -> Optional[Tuple[int, int]]:
        """Return the first unassigned variable."""
        for var in csp.variables:
            if var not in assignment:
                return var
        return None

    def _backtrack(
        self,
        csp: SudokuCSP,
        assignment: Dict[Tuple[int, int], int],
        solutions: List[Dict[Tuple[int, int], int]],
        stop_at_one: bool
    ) -> bool:
        if len(assignment) == len(csp.variables):
            solutions.append(dict(assignment))
            return True

        var = self._select_unassigned_variable(csp, assignment)
        if var is None:
            return True

        # Pure backtracking considers all values 1-9 for unassigned cells
        for val in range(1, 10):
            if csp.is_consistent_assignment(var, val, assignment):
                assignment[var] = val
                self.assignments += 1

                found = self._backtrack(csp, assignment, solutions, stop_at_one)
                if found and stop_at_one:
                    return True

                del assignment[var]
                self.backtracks += 1

        return False
