"""
Backtracking with Forward Checking (FC) for Sudoku CSP.
Prunes neighbor domains upon assignment and backtracks immediately if any domain is wiped out.
"""

from typing import Dict, List, Optional, Set, Tuple
from csp.sudoku_csp import SudokuCSP


class ForwardCheckingSolver:
    """Solves Sudoku using Backtracking with Forward Checking and counters."""

    def __init__(self):
        self.assignments: int = 0
        self.backtracks: int = 0

    def reset_counters(self) -> None:
        self.assignments = 0
        self.backtracks = 0

    def solve(
        self,
        csp: SudokuCSP,
        use_mrv: bool = True,
        stop_at_one: bool = True
    ) -> List[Dict[Tuple[int, int], int]]:
        """
        Solves the CSP using Forward Checking.
        Optional use_mrv enables Minimum Remaining Values variable ordering.
        """
        self.reset_counters()
        assignment: Dict[Tuple[int, int], int] = {}
        domains = {var: set(csp.domains[var]) for var in csp.variables}

        # Check initial consistency
        for var in csp.variables:
            if len(domains[var]) == 1:
                val = next(iter(domains[var]))
                assignment[var] = val

        # Forward check initial clues against each other
        for var, val in list(assignment.items()):
            for peer in csp.peers[var]:
                if peer in assignment and assignment[peer] == val:
                    return []  # Immediate conflict
                if val in domains[peer] and peer not in assignment:
                    domains[peer].remove(val)
                    if len(domains[peer]) == 0:
                        return []

        solutions: List[Dict[Tuple[int, int], int]] = []
        self._backtrack_fc(csp, assignment, domains, solutions, use_mrv, stop_at_one)
        return solutions

    def _select_unassigned_variable(
        self,
        csp: SudokuCSP,
        assignment: Dict[Tuple[int, int], int],
        domains: Dict[Tuple[int, int], Set[int]],
        use_mrv: bool
    ) -> Optional[Tuple[int, int]]:
        unassigned = [v for v in csp.variables if v not in assignment]
        if not unassigned:
            return None
        if use_mrv:
            # Pick variable with smallest domain size
            return min(unassigned, key=lambda v: len(domains[v]))
        return unassigned[0]

    def _backtrack_fc(
        self,
        csp: SudokuCSP,
        assignment: Dict[Tuple[int, int], int],
        domains: Dict[Tuple[int, int], Set[int]],
        solutions: List[Dict[Tuple[int, int], int]],
        use_mrv: bool,
        stop_at_one: bool
    ) -> bool:
        if len(assignment) == len(csp.variables):
            solutions.append(dict(assignment))
            return True

        var = self._select_unassigned_variable(csp, assignment, domains, use_mrv)
        if var is None:
            return True

        for val in sorted(list(domains[var])):
            # Forward check: verify assigning val does not wipe out any neighbor domain
            removals: List[Tuple[Tuple[int, int], int]] = []
            consistent = True

            for peer in csp.peers[var]:
                if peer not in assignment and val in domains[peer]:
                    if len(domains[peer]) == 1:
                        consistent = False
                        break
                    domains[peer].remove(val)
                    removals.append((peer, val))

            if consistent:
                assignment[var] = val
                self.assignments += 1

                found = self._backtrack_fc(csp, assignment, domains, solutions, use_mrv, stop_at_one)
                if found and stop_at_one:
                    return True

                del assignment[var]
                self.backtracks += 1

            # Restore domains pruned by this branch
            for peer, removed_val in removals:
                domains[peer].add(removed_val)

        return False
