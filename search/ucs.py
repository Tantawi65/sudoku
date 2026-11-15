"""
Uniform Cost Search (UCS) for Misread Digit Correction.
Expands nodes in order of minimum path cost g(n).
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple
from search.correction_search import CorrectionSearchProblem, SearchNode


class UCSSolver:
    """Solves the correction search problem using Uniform Cost Search."""

    def __init__(self):
        self.nodes_expanded: int = 0
        self.nodes_generated: int = 0
        self.elapsed_time: float = 0.0

    def reset_counters(self) -> None:
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.elapsed_time = 0.0

    def search(
        self,
        problem: CorrectionSearchProblem
    ) -> Tuple[bool, Optional[SearchNode], Optional[List[List[int]]]]:
        """
        Runs UCS to find the lowest-cost set of corrections.
        Returns (success, best_node, solved_grid).
        """
        self.reset_counters()
        start_time = time.perf_counter()

        initial_node = SearchNode(corrections=(), cost=0.0, heuristic=0.0)
        self.nodes_generated += 1

        # Check if initial state is already a valid goal
        is_goal, solved_grid = problem.is_goal(initial_node.corrections)
        if is_goal:
            self.elapsed_time = time.perf_counter() - start_time
            return True, initial_node, solved_grid

        # Priority queue ordered by g(n)
        frontier: List[Tuple[float, int, SearchNode]] = []
        counter = 0
        heapq.heappush(frontier, (initial_node.cost, counter, initial_node))

        # Track visited corrections sets to avoid redundant expansions
        explored: Set[Tuple[Tuple[Tuple[int, int], int], ...]] = set()

        while frontier:
            cost, _, node = heapq.heappop(frontier)

            if node.corrections in explored:
                continue
            explored.add(node.corrections)
            self.nodes_expanded += 1

            is_goal, solved_grid = problem.is_goal(node.corrections)
            if is_goal:
                self.elapsed_time = time.perf_counter() - start_time
                return True, node, solved_grid

            for child in problem.get_successors(node):
                if child.corrections not in explored:
                    self.nodes_generated += 1
                    counter += 1
                    heapq.heappush(frontier, (child.cost, counter, child))

        self.elapsed_time = time.perf_counter() - start_time
        return False, None, None
