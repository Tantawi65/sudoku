"""
Evaluation Experiments Framework.
Executes the three mandatory experiments defined on Page 15 across 5 distinct random seeds,
computing mean and standard deviation for all evaluation metrics.

Experiments:
1. UCS vs Greedy vs A*: nodes expanded, nodes generated, runtime, and recovery rate for 0-4 misreads.
2. BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, 35 givens.
3. NN-guided correction vs no correction vs random correction.
"""

import os
import json
import random
import time
from typing import Dict, List, Tuple
import numpy as np

from csp.sudoku_csp import SudokuCSP
from csp.solver import UnifiedCSPSolver
from data.generator import SudokuGenerator
from search.correction_search import CorrectionSearchProblem, SearchNode
from search.ucs import UCSSolver
from search.greedy import GreedySolver
from search.astar import AStarSolver


SEEDS = [42, 101, 2024, 777, 999]


def inject_simulated_misreads(
    puzzle: List[List[int]],
    solution: List[List[int]],
    num_misreads: int,
    seed: int
) -> Tuple[List[List[int]], Dict[Tuple[int, int], Dict[int, float]]]:
    """
    Creates controlled simulated misreads and calibrated NN probability outputs.
    Returns corrupted_puzzle and probabilities dict.
    """
    random.seed(seed)
    np.random.seed(seed)

    corrupted = [row[:] for row in puzzle]
    probs: Dict[Tuple[int, int], Dict[int, float]] = {}

    clues = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] != 0]
    random.shuffle(clues)

    # Base high confidence for uncorrupted clues
    for r, c in clues:
        true_d = puzzle[r][c]
        d_probs = {d: 0.01 for d in range(1, 10)}
        d_probs[true_d] = 0.92
        probs[(r, c)] = d_probs

    misread_count = 0
    for r, c in clues:
        if misread_count >= num_misreads:
            break

        true_val = puzzle[r][c]
        # Choose a colliding digit in same row/col/box
        candidates = []
        for i in range(9):
            if i != c and puzzle[r][i] != 0:
                candidates.append(puzzle[r][i])
            if i != r and puzzle[i][c] != 0:
                candidates.append(puzzle[i][c])

        colliding_digit = candidates[0] if candidates else ((true_val % 9) + 1)
        corrupted[r][c] = colliding_digit

        # Construct ambiguous NN probabilities (e.g. 52% vs 45%)
        p_top = random.uniform(0.50, 0.55)
        p_true = random.uniform(0.40, p_top - 0.02)
        rem = 1.0 - (p_top + p_true)

        d_probs = {d: rem / 7.0 for d in range(1, 10)}
        d_probs[colliding_digit] = p_top
        d_probs[true_val] = p_true
        probs[(r, c)] = d_probs

        misread_count += 1

    return corrupted, probs


def run_experiment_1() -> Dict[str, object]:
    """
    Experiment 1: UCS vs Greedy vs A* across 0-4 misreads.
    Metrics: recovery rate, nodes expanded, nodes generated, time.
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 1: UCS vs Greedy vs A* across 0-4 Misreads")
    print("=" * 60)

    misread_levels = [0, 1, 2, 3, 4]
    algorithms = ["UCS", "Greedy", "AStar"]
    raw_results = {algo: {m: [] for m in misread_levels} for algo in algorithms}

    for seed in SEEDS:
        print(f"Executing Seed {seed}...")
        gen = SudokuGenerator(seed=seed)
        puzzle, solution = gen.generate_puzzle(num_givens=30)

        for m in misread_levels:
            corrupted, probs = inject_simulated_misreads(puzzle, solution, num_misreads=m, seed=seed)
            max_corr = max(m, 3)

            for algo in algorithms:
                problem = CorrectionSearchProblem(
                    initial_grid=corrupted,
                    probabilities=probs,
                    max_corrections=max_corr
                )

                if algo == "UCS":
                    solver = UCSSolver()
                elif algo == "Greedy":
                    solver = GreedySolver()
                else:
                    solver = AStarSolver()

                success, node, solved = solver.search(problem)
                recovered = (success and solved == solution)

                raw_results[algo][m].append({
                    "recovered": 1 if recovered else 0,
                    "nodes_expanded": solver.nodes_expanded,
                    "nodes_generated": solver.nodes_generated,
                    "time": solver.elapsed_time
                })

    summary = {algo: {} for algo in algorithms}
    for algo in algorithms:
        for m in misread_levels:
            data = raw_results[algo][m]
            summary[algo][m] = {
                "recovery_rate_mean": float(np.mean([d["recovered"] for d in data])),
                "nodes_expanded_mean": float(np.mean([d["nodes_expanded"] for d in data])),
                "nodes_expanded_std": float(np.std([d["nodes_expanded"] for d in data])),
                "nodes_generated_mean": float(np.mean([d["nodes_generated"] for d in data])),
                "nodes_generated_std": float(np.std([d["nodes_generated"] for d in data])),
                "time_mean": float(np.mean([d["time"] for d in data])),
                "time_std": float(np.std([d["time"] for d in data]))
            }

    return {"summary": summary, "raw": raw_results}


def run_experiment_2() -> Dict[str, object]:
    """
    Experiment 2: BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, 35 givens.
    Metrics: assignments, backtracks, elapsed time.
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 2: CSP Pruning (BT vs FC vs AC3)")
    print("=" * 60)

    given_levels = [25, 30, 35]
    solvers = ["BT", "FC", "AC3"]
    raw_results = {mode: {g: [] for g in given_levels} for mode in solvers}

    for seed in SEEDS:
        print(f"Executing Seed {seed}...")
        gen = SudokuGenerator(seed=seed)

        for g in given_levels:
            puzzle, solution = gen.generate_puzzle(num_givens=g)

            for mode in solvers:
                csp = SudokuCSP(puzzle)
                solver = UnifiedCSPSolver(mode=mode)
                solved = solver.solve(csp)

                raw_results[mode][g].append({
                    "assignments": solver.assignments,
                    "backtracks": solver.backtracks,
                    "time": solver.elapsed_time,
                    "solved": 1 if solved == solution else 0
                })

    summary = {mode: {} for mode in solvers}
    for mode in solvers:
        for g in given_levels:
            data = raw_results[mode][g]
            summary[mode][g] = {
                "assignments_mean": float(np.mean([d["assignments"] for d in data])),
                "assignments_std": float(np.std([d["assignments"] for d in data])),
                "backtracks_mean": float(np.mean([d["backtracks"] for d in data])),
                "backtracks_std": float(np.std([d["backtracks"] for d in data])),
                "time_mean": float(np.mean([d["time"] for d in data])),
                "time_std": float(np.std([d["time"] for d in data]))
            }

    return {"summary": summary, "raw": raw_results}


def run_experiment_3() -> Dict[str, object]:
    """
    Experiment 3: NN-guided correction vs No correction vs Random correction.
    Tested on puzzles with 2 injected misreads across 5 seeds.
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 3: Guidance Impact (NN-Guided vs No-Correction vs Random)")
    print("=" * 60)

    strategies = ["NN_Guided", "No_Correction", "Random_Correction"]
    raw_results = {strat: [] for strat in strategies}

    for seed in SEEDS:
        print(f"Executing Seed {seed}...")
        gen = SudokuGenerator(seed=seed)
        puzzle, solution = gen.generate_puzzle(num_givens=30)
        corrupted, probs = inject_simulated_misreads(puzzle, solution, num_misreads=2, seed=seed)

        # 1. NN-guided (A* using probabilities)
        prob_nn = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
        solver_nn = AStarSolver()
        succ_nn, node_nn, sol_nn = solver_nn.search(prob_nn)
        raw_results["NN_Guided"].append({
            "recovered": 1 if (succ_nn and sol_nn == solution) else 0,
            "nodes_expanded": solver_nn.nodes_expanded,
            "time": solver_nn.elapsed_time
        })

        # 2. No correction (direct CSP solve without search)
        csp_direct = SudokuCSP(corrupted)
        csp_solver = UnifiedCSPSolver(mode="FC")
        sol_direct = csp_solver.solve(csp_direct)
        raw_results["No_Correction"].append({
            "recovered": 1 if (sol_direct == solution) else 0,
            "nodes_expanded": 0,
            "time": csp_solver.elapsed_time
        })

        # 3. Random correction (uninformative flat probabilities)
        flat_probs = {}
        for (r, c), pdict in probs.items():
            top_d = corrupted[r][c]
            # Uniform probabilities without NN confidence
            flat_probs[(r, c)] = {d: 1.0 / 9.0 for d in range(1, 10)}

        prob_rnd = CorrectionSearchProblem(corrupted, flat_probs, max_corrections=3)
        solver_rnd = AStarSolver()
        succ_rnd, node_rnd, sol_rnd = solver_rnd.search(prob_rnd)
        raw_results["Random_Correction"].append({
            "recovered": 1 if (succ_rnd and sol_rnd == solution) else 0,
            "nodes_expanded": solver_rnd.nodes_expanded,
            "time": solver_rnd.elapsed_time
        })

    summary = {}
    for strat in strategies:
        data = raw_results[strat]
        summary[strat] = {
            "recovery_rate_mean": float(np.mean([d["recovered"] for d in data])),
            "nodes_expanded_mean": float(np.mean([d["nodes_expanded"] for d in data])),
            "nodes_expanded_std": float(np.std([d["nodes_expanded"] for d in data])),
            "time_mean": float(np.mean([d["time"] for d in data])),
            "time_std": float(np.std([d["time"] for d in data]))
        }

    return {"summary": summary, "raw": raw_results}


def run_all_experiments() -> None:
    """Executes all three experiments and writes aggregated output to JSON."""
    os.makedirs("eval/results", exist_ok=True)
    all_results = {
        "experiment_1": run_experiment_1(),
        "experiment_2": run_experiment_2(),
        "experiment_3": run_experiment_3()
    }

    out_file = "eval/results/experiment_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll experiments successfully completed. Saved to: {out_file}")


if __name__ == "__main__":
    run_all_experiments()
