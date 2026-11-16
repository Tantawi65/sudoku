"""
Runs the 3 required experiments for the handwritten Sudoku reader and solver project.
Generates results to be plotted and documented in the final report.
"""
import os
import json
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any

from csp.sudoku_csp import SudokuCSP
from csp.solver import UnifiedCSPSolver
from search.correction_search import CorrectionSearchProblem
from search.astar import AStarSolver
from search.ucs import UCSSolver
from search.greedy import GreedySolver
from data.generator import SudokuGenerator

def run_experiment_1(seeds: List[int]) -> Dict[str, Any]:
    """
    Experiment 1 (Search Comparison):
    UCS vs Greedy vs A*: nodes and recovery rate for 0-4 misreads.
    """
    print("\n--- Running Experiment 1: Search Comparison ---")
    algorithms = {
        "AStar": AStarSolver,
        "UCS": UCSSolver,
        "Greedy": GreedySolver
    }
    misread_levels = [0, 1, 2, 3, 4]
    
    results = defaultdict(lambda: defaultdict(list))
    
    for seed in seeds:
        gen = SudokuGenerator(seed=seed)
        base_puzzle, _ = gen.generate_puzzle(num_givens=30)
        
        valid_coords = [(r, c) for r in range(9) for c in range(9) if base_puzzle[r][c] != 0]
        
        for num_misreads in misread_levels:
            corrupted = [row[:] for row in base_puzzle]
            probs = {(r, c): {base_puzzle[r][c]: 0.95} for r in range(9) for c in range(9) if base_puzzle[r][c] != 0}
            
            # Inject misreads
            np.random.seed(seed + num_misreads)
            if num_misreads > 0:
                coords_to_corrupt = [valid_coords[i] for i in np.random.choice(len(valid_coords), num_misreads, replace=False)]
                for r, c in coords_to_corrupt:
                    true_val = base_puzzle[r][c]
                    # Ensure the misread value isn't the true value
                    misread_val = (true_val % 9) + 1
                    corrupted[r][c] = misread_val
                    probs[(r, c)] = {misread_val: 0.55, true_val: 0.40, (misread_val % 9) + 1: 0.05}
            
            for algo_name, solver_cls in algorithms.items():
                prob_model = CorrectionSearchProblem(corrupted, probs, max_corrections=5)
                solver = solver_cls()
                succ, _, _ = solver.search(prob_model)
                
                results[algo_name][num_misreads].append({
                    "success": 1 if succ else 0,
                    "nodes_expanded": solver.nodes_expanded,
                    "nodes_generated": solver.nodes_generated,
                    "time": solver.elapsed_time
                })
                
    # Aggregate results (Mean and Std Dev)
    summary = defaultdict(dict)
    for algo, levels in results.items():
        for level, runs in levels.items():
            summary[algo][level] = {
                "recovery_rate_mean": np.mean([r["success"] for r in runs]),
                "nodes_expanded_mean": np.mean([r["nodes_expanded"] for r in runs]),
                "nodes_expanded_std": np.std([r["nodes_expanded"] for r in runs]),
                "nodes_generated_mean": np.mean([r["nodes_generated"] for r in runs]),
                "nodes_generated_std": np.std([r["nodes_generated"] for r in runs]),
                "time_mean": np.mean([r["time"] for r in runs]),
                "time_std": np.std([r["time"] for r in runs])
            }
            
    print("Experiment 1 complete.")
    return summary

def run_experiment_2(seeds: List[int]) -> Dict[str, Any]:
    """
    Experiment 2 (CSP Pruning Comparison):
    BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, 35 givens.
    """
    print("\n--- Running Experiment 2: CSP Pruning ---")
    modes = ["BT", "FC", "AC3"]
    givens_levels = [25, 30, 35]
    
    results = defaultdict(lambda: defaultdict(list))
    
    for seed in seeds:
        gen = SudokuGenerator(seed=seed)
        for givens in givens_levels:
            puzzle, _ = gen.generate_puzzle(num_givens=givens)
            
            for mode in modes:
                solver = UnifiedCSPSolver(mode=mode)
                csp = SudokuCSP(puzzle)
                sol = solver.solve(csp)
                
                results[mode][givens].append({
                    "solved": 1 if sol else 0,
                    "assignments": solver.assignments,
                    "backtracks": solver.backtracks,
                    "time": solver.elapsed_time
                })
                
    # Aggregate results
    summary = defaultdict(dict)
    for mode, levels in results.items():
        for givens, runs in levels.items():
            summary[mode][givens] = {
                "assignments_mean": np.mean([r["assignments"] for r in runs]),
                "assignments_std": np.std([r["assignments"] for r in runs]),
                "backtracks_mean": np.mean([r["backtracks"] for r in runs]),
                "backtracks_std": np.std([r["backtracks"] for r in runs]),
                "time_mean": np.mean([r["time"] for r in runs]),
                "time_std": np.std([r["time"] for r in runs])
            }
            
    print("Experiment 2 complete.")
    return summary

def run_experiment_3(seeds: List[int]) -> Dict[str, Any]:
    """
    Experiment 3 (NN Guidance Impact):
    NN-guided correction vs no correction vs random correction.
    Uses A* search for correction attempts.
    """
    print("\n--- Running Experiment 3: Guidance Impact ---")
    results = defaultdict(list)
    
    for seed in seeds:
        gen = SudokuGenerator(seed=seed)
        base_puzzle, _ = gen.generate_puzzle(num_givens=30)
        valid_coords = [(r, c) for r in range(9) for c in range(9) if base_puzzle[r][c] != 0]
        
        # Inject 2 misreads for this experiment
        corrupted = [row[:] for row in base_puzzle]
        probs = {(r, c): {base_puzzle[r][c]: 0.95} for r in range(9) for c in range(9) if base_puzzle[r][c] != 0}
        
        np.random.seed(seed + 100)
        coords_to_corrupt = [valid_coords[i] for i in np.random.choice(len(valid_coords), 2, replace=False)]
        for r, c in coords_to_corrupt:
            true_val = base_puzzle[r][c]
            misread_val = (true_val % 9) + 1
            corrupted[r][c] = misread_val
            probs[(r, c)] = {misread_val: 0.52, true_val: 0.44, (misread_val % 9) + 1: 0.04}
            
        # Strategy A: NN Guided
        prob_nn = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
        solver_nn = AStarSolver()
        succ_nn, _, _ = solver_nn.search(prob_nn)
        results["NN_Guided"].append({
            "success": 1 if succ_nn else 0,
            "nodes_expanded": solver_nn.nodes_expanded,
            "time": solver_nn.elapsed_time
        })
        
        # Strategy B: Random Correction (Flat Probabilities)
        flat_probs = {c: {d: 1.0 / 9.0 for d in range(1, 10)} for c in probs.keys()}
        prob_rand = CorrectionSearchProblem(corrupted, flat_probs, max_corrections=3)
        solver_rand = AStarSolver()
        succ_rand, _, _ = solver_rand.search(prob_rand)
        results["Random_Correction"].append({
            "success": 1 if succ_rand else 0,
            "nodes_expanded": solver_rand.nodes_expanded,
            "time": solver_rand.elapsed_time
        })
        
        # Strategy C: No Correction (Direct CSP)
        solver_none = UnifiedCSPSolver(mode="FC")
        csp_none = SudokuCSP(corrupted)
        sol_none = solver_none.solve(csp_none)
        results["No_Correction"].append({
            "success": 1 if sol_none else 0,
            "nodes_expanded": 0,
            "time": solver_none.elapsed_time
        })

    # Aggregate results
    summary = {}
    for strategy, runs in results.items():
        summary[strategy] = {
            "recovery_rate_mean": np.mean([r["success"] for r in runs]),
            "nodes_expanded_mean": np.mean([r["nodes_expanded"] for r in runs]),
            "nodes_expanded_std": np.std([r["nodes_expanded"] for r in runs]),
            "time_mean": np.mean([r["time"] for r in runs]),
            "time_std": np.std([r["time"] for r in runs])
        }
        
    print("Experiment 3 complete.")
    return summary

def run_all_experiments():
    seeds = [42, 101, 2024, 777, 999]
    print(f"Running experiments across {len(seeds)} random seeds: {seeds}")
    
    exp1_data = run_experiment_1(seeds)
    exp2_data = run_experiment_2(seeds)
    exp3_data = run_experiment_3(seeds)
    
    final_results = {
        "metadata": {
            "seeds_used": seeds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "experiment_1": {
            "description": "Search Algorithm Comparison (UCS vs Greedy vs A*) across 0-4 misreads",
            "summary": exp1_data
        },
        "experiment_2": {
            "description": "CSP Pruning Mechanisms (BT vs FC vs AC-3) across 25, 30, 35 givens",
            "summary": exp2_data
        },
        "experiment_3": {
            "description": "Neural Network Guidance Impact (NN Guided vs Random vs No Correction)",
            "summary": exp3_data
        }
    }
    
    os.makedirs("eval/results", exist_ok=True)
    out_path = "eval/results/experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nAll experiments finished. Results saved to {out_path}")

    # Generate plots and markdown
    try:
        from eval.plot_results import generate_markdown_tables, generate_plots
        generate_markdown_tables(final_results)
        generate_plots(final_results)
    except Exception as e:
        print(f"Plot generation notice: {e}")

    # Print clean summary tables directly to CMD
    print("\n" + "=" * 80)
    print("                    COURSE EXPERIMENT RESULTS (CMD VIEW)")
    print("=" * 80)

    print("\n[TABLE 1: Search Algorithm Comparison across 0-4 Misreads (5 Seeds)]")
    print(f"{'Algorithm':<10} | {'Misreads':<8} | {'Recovery Rate':<14} | {'Nodes Expanded (Mean ± Std)':<28} | {'Time (s)':<10}")
    print("-" * 80)
    for algo in ["AStar", "UCS", "Greedy"]:
        for m in [0, 1, 2, 3, 4]:
            st = exp1_data[algo][m]
            print(f"{algo:<10} | {m:<8} | {st['recovery_rate_mean']*100:>12.1f}% | {st['nodes_expanded_mean']:>10.2f} ± {st['nodes_expanded_std']:<14.2f} | {st['time_mean']:>8.4f}")

    print("\n[TABLE 2: CSP Pruning Mechanisms across 25, 30, 35 Givens (5 Seeds)]")
    print(f"{'Givens':<8} | {'Algorithm':<10} | {'Assignments (Mean ± Std)':<26} | {'Backtracks (Mean ± Std)':<26} | {'Time (s)':<10}")
    print("-" * 88)
    for g in [25, 30, 35]:
        for mode in ["BT", "FC", "AC3"]:
            st = exp2_data[mode][g]
            print(f"{g:<8} | {mode:<10} | {st['assignments_mean']:>10.1f} ± {st['assignments_std']:<12.1f} | {st['backtracks_mean']:>10.1f} ± {st['backtracks_std']:<12.1f} | {st['time_mean']:>8.4f}")

    print("\n[TABLE 3: Neural Network Guidance Impact (2 Misreads, 5 Seeds)]")
    print(f"{'Strategy':<20} | {'Recovery Rate':<14} | {'Nodes Expanded (Mean ± Std)':<28} | {'Time (s)':<10}")
    print("-" * 80)
    for strat in ["NN_Guided", "Random_Correction", "No_Correction"]:
        st = exp3_data[strat]
        print(f"{strat:<20} | {st['recovery_rate_mean']*100:>12.1f}% | {st['nodes_expanded_mean']:>10.2f} ± {st['nodes_expanded_std']:<14.2f} | {st['time_mean']:>8.4f}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_all_experiments()
