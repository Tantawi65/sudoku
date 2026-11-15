"""
Arc Consistency Algorithm (AC-3) for Sudoku CSP.
Enforces arc consistency across all binary constraints and detects dead-ends early.
"""

from collections import deque
from typing import Dict, Set, Tuple, Optional
from csp.sudoku_csp import SudokuCSP


class AC3Solver:
    """Implements the AC-3 algorithm to enforce arc consistency on a Sudoku CSP."""

    def __init__(self):
        self.arcs_tested: int = 0
        self.values_pruned: int = 0

    def reset_counters(self) -> None:
        self.arcs_tested = 0
        self.values_pruned = 0

    def revise(self, csp: SudokuCSP, xi: Tuple[int, int], xj: Tuple[int, int]) -> bool:
        """
        Revise the domain of xi given xj.
        Deletes a value x in domain(xi) if there is no y in domain(xj) such that x != y.
        Returns True if domain(xi) was modified.
        """
        revised = False
        domain_xi = set(csp.domains[xi])
        domain_xj = csp.domains[xj]

        for x in domain_xi:
            # Check if there exists some y in domain(xj) satisfying the binary constraint (x != y)
            has_support = any(x != y for y in domain_xj)
            if not has_support:
                csp.domains[xi].remove(x)
                self.values_pruned += 1
                revised = True

        return revised

    def enforce_arc_consistency(
        self,
        csp: SudokuCSP,
        queue: Optional[deque] = None
    ) -> bool:
        """
        Runs AC-3 on the CSP.
        If queue is None, initializes with all arcs in the problem.
        Returns False if any domain is wiped out (unsolvable), True otherwise.
        """
        if queue is None:
            queue = deque()
            for xi in csp.variables:
                for xj in csp.peers[xi]:
                    queue.append((xi, xj))

        while queue:
            xi, xj = queue.popleft()
            self.arcs_tested += 1

            if self.revise(csp, xi, xj):
                if len(csp.domains[xi]) == 0:
                    return False  # Inconsistency detected
                # Domain of xi was reduced: add all arcs (xk, xi) where xk is a peer of xi (except xj)
                for xk in csp.peers[xi]:
                    if xk != xj:
                        queue.append((xk, xi))

        return True


def run_ac3(csp: SudokuCSP) -> Tuple[bool, int, int]:
    """
    Convenience wrapper to run AC-3 on a copy of CSP.
    Returns (is_consistent, arcs_tested, values_pruned).
    """
    solver = AC3Solver()
    success = solver.enforce_arc_consistency(csp)
    return success, solver.arcs_tested, solver.values_pruned
