"""
Plotting and Summary Table Generation.
Reads experiment_results.json and produces publication-quality charts and Markdown tables (mean +/- std).
"""

import os
import json
from typing import Dict
import numpy as np


def generate_markdown_tables(results_path: str = "eval/results/experiment_results.json") -> None:
    """Generates Markdown tables from experimental results for the 12-page report."""
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs("eval/results", exist_ok=True)
    table_file = "eval/results/experiment_tables.md"

    exp1 = data["experiment_1"]["summary"]
    misreads = [0, 1, 2, 3, 4]
    algos = ["UCS", "Greedy", "AStar"]

    exp2 = data["experiment_2"]["summary"]
    givens = [25, 30, 35]
    modes = ["BT", "FC", "AC3"]

    exp3 = data["experiment_3"]["summary"]
    strats = ["NN_Guided", "Random_Correction", "No_Correction"]
    labels = ["NN Guided (A*)", "Random Guidance", "No Correction"]

    with open(table_file, "w") as f:
        f.write("# Experimental Evaluation Tables (Mean ± Standard Deviation)\n\n")

        # Table 1
        f.write("## Table 1: Experiment 1 - Search Algorithm Comparison across 0-4 Misreads (5 Seeds)\n\n")
        f.write("| Algorithm | Misreads | Recovery Rate (%) | Nodes Expanded (Mean ± Std) | Nodes Generated (Mean ± Std) | Runtime (ms) (Mean ± Std) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for algo in algos:
            for m in misreads:
                st = exp1[algo][str(m)]
                f.write(
                    f"| {algo} | {m} | {st['recovery_rate_mean'] * 100:.1f}% | "
                    f"{st['nodes_expanded_mean']:.2f} ± {st['nodes_expanded_std']:.2f} | "
                    f"{st['nodes_generated_mean']:.2f} ± {st['nodes_generated_std']:.2f} | "
                    f"{st['time_mean'] * 1000:.2f} ± {st['time_std'] * 1000:.2f} |\n"
                )

        # Table 2
        f.write("\n## Table 2: Experiment 2 - CSP Pruning Comparison (5 Seeds)\n\n")
        f.write("| Solver Mode | Clues (Givens) | Assignments (Mean ± Std) | Backtracks (Mean ± Std) | Runtime (s) (Mean ± Std) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for mode in modes:
            for g in givens:
                st = exp2[mode][str(g)]
                f.write(
                    f"| {mode} | {g} | {st['assignments_mean']:.1f} ± {st['assignments_std']:.1f} | "
                    f"{st['backtracks_mean']:.1f} ± {st['backtracks_std']:.1f} | "
                    f"{st['time_mean']:.4f} ± {st['time_std']:.4f} |\n"
                )

        # Table 3
        f.write("\n## Table 3: Experiment 3 - NN Guidance Impact (5 Seeds, 2 Misreads)\n\n")
        f.write("| Strategy | Recovery Rate (%) | Nodes Expanded (Mean ± Std) | Runtime (ms) (Mean ± Std) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for s, l in zip(strats, labels):
            st = exp3[s]
            f.write(
                f"| {l} | {st['recovery_rate_mean'] * 100:.1f}% | "
                f"{st['nodes_expanded_mean']:.2f} ± {st['nodes_expanded_std']:.2f} | "
                f"{st['time_mean'] * 1000:.2f} ± {st['time_std'] * 1000:.2f} |\n"
            )

    print(f"Generated Markdown tables at: {table_file}")


def generate_experiment_plots(results_path: str = "eval/results/experiment_results.json") -> None:
    """Generates charts for Experiments 1, 2, and 3 using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed yet. Skipping plot generation.")
        return

    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs("eval/plots", exist_ok=True)

    # Plot 1
    exp1 = data["experiment_1"]["summary"]
    misreads = [0, 1, 2, 3, 4]
    algos = ["UCS", "Greedy", "AStar"]
    colors = {"UCS": "#2b5c8f", "Greedy": "#e67e22", "AStar": "#27ae60"}
    markers = {"UCS": "o", "Greedy": "s", "AStar": "^"}

    plt.figure(figsize=(8, 5))
    for algo in algos:
        means = [exp1[algo][str(m)]["nodes_expanded_mean"] for m in misreads]
        stds = [exp1[algo][str(m)]["nodes_expanded_std"] for m in misreads]
        plt.errorbar(
            misreads, means, yerr=stds, label=f"{algo}",
            fmt=f"-{markers[algo]}", color=colors[algo], capsize=4, linewidth=2
        )

    plt.title("Experiment 1: Search Nodes Expanded vs Number of Misreads (5 Seeds)", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Injected Misreads", fontsize=11)
    plt.ylabel("Nodes Expanded (Mean ± Std)", fontsize=11)
    plt.xticks(misreads)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("eval/plots/exp1_search_comparison.png", dpi=300)
    plt.close()

    # Plot 2
    exp2 = data["experiment_2"]["summary"]
    givens = [25, 30, 35]
    modes = ["BT", "FC", "AC3"]
    x = np.arange(len(givens))
    width = 0.25

    plt.figure(figsize=(8, 5))
    for i, mode in enumerate(modes):
        b_means = [exp2[mode][str(g)]["backtracks_mean"] for g in givens]
        plt.bar(x + (i - 1) * width, [max(1, m) for m in b_means], width, label=mode, alpha=0.85)

    plt.yscale("log")
    plt.title("Experiment 2: CSP Pruning Efficiency (Backtracks vs Clues)", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Given Clues", fontsize=11)
    plt.ylabel("Backtracks (Log Scale)", fontsize=11)
    plt.xticks(x, [f"{g} Clues" for g in givens])
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("eval/plots/exp2_csp_comparison.png", dpi=300)
    plt.close()

    # Plot 3
    exp3 = data["experiment_3"]["summary"]
    strats = ["NN_Guided", "Random_Correction", "No_Correction"]
    labels = ["NN Guided (A*)", "Random Guidance", "No Correction"]
    rec_rates = [exp3[s]["recovery_rate_mean"] * 100 for s in strats]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Search Guidance Strategy", fontsize=11)
    ax1.set_ylabel("Solution Recovery Rate (%)", color=color, fontsize=11)
    bars = ax1.bar(labels, rec_rates, color=color, alpha=0.7, width=0.4)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 115)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.title("Experiment 3: Impact of NN Probability Guidance (2 Misreads)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig("eval/plots/exp3_guidance_comparison.png", dpi=300)
    plt.close()
    print("All plots saved in eval/plots/")


if __name__ == "__main__":
    generate_markdown_tables()
    generate_experiment_plots()
