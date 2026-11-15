"""
Sudoku Puzzle Generator.
Generates complete valid 9x9 boards using randomized backtracking,
and creates unique-solution puzzles with a parameterized number of clues.
"""

import random
from typing import List, Optional, Tuple
from csp.sudoku_csp import SudokuCSP
from csp.solver import UnifiedCSPSolver


class SudokuGenerator:
    """Generates valid 9x9 Sudoku puzzles with guaranteed unique solutions."""

    def __init__(self, seed: Optional[int] = 42):
        if seed is not None:
            random.seed(seed)
        self.solver = UnifiedCSPSolver(mode="FC")

    def generate_solution(self) -> List[List[int]]:
        """Generates a complete, randomly filled valid 9x9 Sudoku board."""
        board = [[0 for _ in range(9)] for _ in range(9)]
        self._fill_board(board)
        return board

    def _fill_board(self, board: List[List[int]]) -> bool:
        """Helper to fill the board with random digits via backtracking."""
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    digits = list(range(1, 10))
                    random.shuffle(digits)
                    for d in digits:
                        if self._is_valid_move(board, r, c, d):
                            board[r][c] = d
                            if self._fill_board(board):
                                return True
                            board[r][c] = 0
                    return False
        return True

    def _is_valid_move(self, board: List[List[int]], row: int, col: int, val: int) -> bool:
        # Check row and col
        for i in range(9):
            if board[row][i] == val or board[i][col] == val:
                return False
        # Check 3x3 box
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if board[r][c] == val:
                    return False
        return True

    def generate_puzzle(self, num_givens: int = 30) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Creates a puzzle with exactly `num_givens` clues having a unique solution.
        Returns (puzzle_grid, solution_grid).
        """
        solution = self.generate_solution()
        puzzle = [row[:] for row in solution]

        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)

        current_givens = 81
        for r, c in cells:
            if current_givens <= num_givens:
                break

            removed_val = puzzle[r][c]
            puzzle[r][c] = 0

            # Check if puzzle still has a unique solution
            csp = SudokuCSP(puzzle)
            if self.solver.count_solutions(csp, max_count=2) != 1:
                # Removing this clue caused ambiguity or unsolvability; restore it
                puzzle[r][c] = removed_val
            else:
                current_givens -= 1

        return puzzle, solution
