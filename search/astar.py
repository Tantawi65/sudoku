"""
A* Search for Misread Digit Correction.
Expands nodes in order of evaluation function f(n) = g(n) + h(n).
Uses the conflict-reduction admissible heuristic.
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple
from search.correction_search import CorrectionSearchProblem, SearchNode


class AStarSolver:
    """Solves the correction search problem using A* Search."""

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
        Runs A* search ordered by f(n) = g(n) + h(n).
        Returns (success, best_node, solved_grid).
        """
        self.reset_counters()
        start_time = time.perf_counter()

        initial_grid = problem.get_corrected_grid(())
        h_init = problem.compute_heuristic(initial_grid)
        initial_node = SearchNode(corrections=(), cost=0.0, heuristic=h_init)
        self.nodes_generated += 1

        is_goal, solved_grid = problem.is_goal(initial_node.corrections)
        if is_goal:
            self.elapsed_time = time.perf_counter() - start_time
            return True, initial_node, solved_grid

        # Priority queue ordered by f(n)
        frontier: List[Tuple[float, int, SearchNode]] = []
        counter = 0
        heapq.heappush(frontier, (initial_node.total_cost, counter, initial_node))

        # Best cost seen for a given corrections set
        best_g: Dict[Tuple[Tuple[Tuple[int, int], int], ...], float] = {(): 0.0}
        explored: Set[Tuple[Tuple[Tuple[int, int], int], ...]] = set()

        while frontier:
            f, _, node = heapq.heappop(frontier)

            if node.corrections in explored:
                continue
            explored.add(node.corrections)
            self.nodes_expanded += 1

            is_goal, solved_grid = problem.is_goal(node.corrections)
            if is_goal:
                self.elapsed_time = time.perf_counter() - start_time
                return True, node, solved_grid

            for child in problem.get_successors(node):
                if child.corrections in explored:
                    continue

                if child.corrections not in best_g or child.cost < best_g[child.corrections]:
                    best_g[child.corrections] = child.cost
                    self.nodes_generated += 1
                    counter += 1
                    heapq.heappush(frontier, (child.total_cost, counter, child))

        self.elapsed_time = time.perf_counter() - start_time
        return False, None, None
