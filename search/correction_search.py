"""
Correction Search State Space Definition.
Models the search problem of correcting misread Sudoku digits using NN probability penalties.
"""

from typing import Dict, List, Optional, Set, Tuple
import math
from csp.sudoku_csp import SudokuCSP
from csp.ac3 import AC3Solver
from csp.solver import UnifiedCSPSolver


class SearchNode:
    """Represents a node in the state-space correction search tree."""

    def __init__(
        self,
        corrections: Tuple[Tuple[Tuple[int, int], int], ...],
        cost: float,
        heuristic: float = 0.0,
        parent: Optional['SearchNode'] = None
    ):
        self.corrections = corrections  # Tuple of ((r, c), new_val)
        self.cost = cost                # Path cost g(n)
        self.heuristic = heuristic      # Heuristic h(n)
        self.total_cost = cost + heuristic  # f(n) = g(n) + h(n)
        self.parent = parent

    def __lt__(self, other: 'SearchNode') -> bool:
        if math.isclose(self.total_cost, other.total_cost, rel_tol=1e-7):
            return self.cost < other.cost
        return self.total_cost < other.total_cost


class CorrectionSearchProblem:
    """
    Defines the search problem:
    - Initial state: No corrections ()
    - Actions: Change an uncertain clue cell to one of its alternative digits
    - Step cost: Probability gap between top prediction and alternative digit
    - Goal test: Corrected grid has zero direct conflicts and is solvable by CSP
    """

    def __init__(
        self,
        initial_grid: List[List[int]],
        probabilities: Dict[Tuple[int, int], Dict[int, float]],
        max_corrections: int = 3,
        top_k_alternatives: int = 3
    ):
        self.initial_grid = [row[:] for row in initial_grid]
        self.probabilities = probabilities
        self.max_corrections = max_corrections
        self.top_k_alternatives = top_k_alternatives
        self.clue_cells = list(probabilities.keys())

    def get_corrected_grid(self, corrections: Tuple[Tuple[Tuple[int, int], int], ...]) -> List[List[int]]:
        """Applies a set of corrections to the initial grid."""
        grid = [row[:] for row in self.initial_grid]
        for (r, c), new_val in corrections:
            grid[r][c] = new_val
        return grid

    def get_conflicts(self, grid: List[List[int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detects all pairs of given cells that have identical values in the same row, col, or box.
        Returns a list of conflicting coordinate pairs ((r1, c1), (r2, c2)).
        """
        conflicts = []
        # Check rows
        for r in range(9):
            seen: Dict[int, int] = {}
            for c in range(9):
                val = grid[r][c]
                if val != 0:
                    if val in seen:
                        conflicts.append(((r, seen[val]), (r, c)))
                    else:
                        seen[val] = c

        # Check cols
        for c in range(9):
            seen = {}
            for r in range(9):
                val = grid[r][c]
                if val != 0:
                    if val in seen:
                        conflicts.append(((seen[val], c), (r, c)))
                    else:
                        seen[val] = r

        # Check 3x3 boxes
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                seen = {}
                for r in range(br, br + 3):
                    for c in range(bc, bc + 3):
                        val = grid[r][c]
                        if val != 0:
                            if val in seen:
                                conflicts.append((seen[val], (r, c)))
                            else:
                                seen[val] = (r, c)
        return conflicts

    def compute_heuristic(self, grid: List[List[int]]) -> float:
        """
        Admissible heuristic:
        In any duplicate-digit conflict between cell A and cell B, at least one cell must change.
        We find independent conflict pairs and sum the minimum cost to alter one of the conflicting cells.
        """
        conflicts = self.get_conflicts(grid)
        if not conflicts:
            return 0.0

        used_cells: Set[Tuple[int, int]] = set()
        h_cost = 0.0

        for cell1, cell2 in conflicts:
            if cell1 not in used_cells and cell2 not in used_cells:
                used_cells.add(cell1)
                used_cells.add(cell2)

                # Cost to alter cell1
                cost1 = self._min_alteration_cost(cell1, grid[cell1[0]][cell1[1]])
                # Cost to alter cell2
                cost2 = self._min_alteration_cost(cell2, grid[cell2[0]][cell2[1]])
                h_cost += min(cost1, cost2)

        return h_cost

    def _min_alteration_cost(self, cell: Tuple[int, int], current_val: int) -> float:
        """Finds the lowest cost to change `cell` away from `current_val`."""
        if cell not in self.probabilities:
            return 0.1
        probs = self.probabilities[cell]
        p_current = probs.get(current_val, 0.5)

        min_cost = float('inf')
        for alt_val, p_alt in probs.items():
            if alt_val != current_val:
                cost = max(0.0, p_current - p_alt)
                if cost < min_cost:
                    min_cost = cost
        return min_cost if min_cost != float('inf') else 0.1

    def is_goal(self, corrections: Tuple[Tuple[Tuple[int, int], int], ...]) -> Tuple[bool, Optional[List[List[int]]]]:
        """
        Goal test:
        1. Fast AC-3 test to ensure no domain wiped out.
        2. CSP solver to verify existence of a complete valid solution.
        """
        grid = self.get_corrected_grid(corrections)
        if self.get_conflicts(grid):
            return False, None

        csp = SudokuCSP(grid)
        ac3 = AC3Solver()
        if not ac3.enforce_arc_consistency(csp):
            return False, None

        solver = UnifiedCSPSolver(mode="FC")
        solution = solver.solve(csp, stop_at_one=True)
        if solution is not None:
            return True, solution
        return False, None

    def get_successors(self, node: SearchNode) -> List[SearchNode]:
        """
        Generates successor states by altering a conflicting or low-confidence clue.
        Prunes paths that exceed max_corrections.
        """
        if len(node.corrections) >= self.max_corrections:
            return []

        corrected_cells = {cell for cell, _ in node.corrections}
        current_grid = self.get_corrected_grid(node.corrections)

        # Prioritize cells actively causing duplicate conflicts
        conflicts = self.get_conflicts(current_grid)
        candidate_cells = set()
        for c1, c2 in conflicts:
            if c1 not in corrected_cells and c1 in self.probabilities:
                candidate_cells.add(c1)
            if c2 not in corrected_cells and c2 in self.probabilities:
                candidate_cells.add(c2)

        # If no active conflicts or candidates exhausted, inspect lowest confidence clues
        if not candidate_cells:
            sorted_by_uncertainty = sorted(
                [c for c in self.clue_cells if c not in corrected_cells],
                key=lambda cell: max(self.probabilities[cell].values())
            )
            candidate_cells = set(sorted_by_uncertainty[:2])

        successors = []
        for cell in candidate_cells:
            r, c = cell
            current_val = current_grid[r][c]
            probs = self.probabilities[cell]
            p_top = max(probs.values())

            # Sort alternatives by probability descending
            alt_digits = sorted(
                [d for d in probs.keys() if d != current_val],
                key=lambda d: probs[d],
                reverse=True
            )[:self.top_k_alternatives]

            for new_val in alt_digits:
                p_alt = probs[new_val]
                step_cost = max(0.0, p_top - p_alt)
                new_cost = node.cost + step_cost

                # New corrections tuple
                new_corrections = tuple(sorted(list(node.corrections) + [(cell, new_val)]))
                new_grid = self.get_corrected_grid(new_corrections)
                h = self.compute_heuristic(new_grid)

                child = SearchNode(
                    corrections=new_corrections,
                    cost=new_cost,
                    heuristic=h,
                    parent=node
                )
                successors.append(child)

        return successors
