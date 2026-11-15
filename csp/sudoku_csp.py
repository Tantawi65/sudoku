"""
Sudoku CSP Representation
Defines variables, domains, unit peers, and binary constraints for 9x9 Sudoku.
"""

from typing import Dict, List, Set, Tuple, Optional
import copy


def get_peers(row: int, col: int) -> Set[Tuple[int, int]]:
    """Return all neighbor cells (same row, column, and 3x3 box) excluding (row, col)."""
    peers = set()
    # Row and Column peers
    for i in range(9):
        if i != col:
            peers.add((row, i))
        if i != row:
            peers.add((i, col))

    # 3x3 Box peers
    box_r, box_c = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            if (r, c) != (row, col):
                peers.add((r, c))

    return peers


class SudokuCSP:
    """Represents the 9x9 Sudoku puzzle as a Constraint Satisfaction Problem."""

    def __init__(self, grid: Optional[List[List[int]]] = None):
        self.variables: List[Tuple[int, int]] = [(r, c) for r in range(9) for c in range(9)]
        self.peers: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
            var: get_peers(var[0], var[1]) for var in self.variables
        }
        self.domains: Dict[Tuple[int, int], Set[int]] = {}

        if grid is not None:
            self.init_from_grid(grid)
        else:
            for var in self.variables:
                self.domains[var] = set(range(1, 10))

    def init_from_grid(self, grid: List[List[int]]) -> None:
        """Initialize variable domains based on a 9x9 grid where 0 indicates an empty cell."""
        for r in range(9):
            for c in range(9):
                val = grid[r][c]
                if val in range(1, 10):
                    self.domains[(r, c)] = {val}
                else:
                    self.domains[(r, c)] = set(range(1, 10))

    def copy(self) -> 'SudokuCSP':
        """Deep copy of the CSP structure."""
        new_csp = SudokuCSP()
        new_csp.variables = self.variables
        new_csp.peers = self.peers
        new_csp.domains = {k: set(v) for k, v in self.domains.items()}
        return new_csp

    def to_grid(self) -> List[List[int]]:
        """Convert current single-valued domains to a 9x9 grid. Empty cells remain 0."""
        grid = [[0 for _ in range(9)] for _ in range(9)]
        for (r, c), dom in self.domains.items():
            if len(dom) == 1:
                grid[r][c] = next(iter(dom))
            else:
                grid[r][c] = 0
        return grid

    def is_consistent_assignment(self, var: Tuple[int, int], val: int, assignment: Dict[Tuple[int, int], int]) -> bool:
        """Check if assigning val to var satisfies binary inequality with assigned peers."""
        for peer in self.peers[var]:
            if peer in assignment and assignment[peer] == val:
                return False
        return True

    def has_immediate_conflict(self) -> bool:
        """Check if any fixed clues directly violate Sudoku constraints."""
        for var in self.variables:
            if len(self.domains[var]) == 1:
                val = next(iter(self.domains[var]))
                for peer in self.peers[var]:
                    if len(self.domains[peer]) == 1 and next(iter(self.domains[peer])) == val:
                        return True
            elif len(self.domains[var]) == 0:
                return True
        return False
