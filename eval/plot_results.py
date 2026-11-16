"""
Reads the experiment results and generates summary tables and plots.
Fulfills the requirement for formatting results (mean +/- std dev).
"""
import os
import json
import matplotlib.pyplot as plt

def generate_markdown_tables(data: dict = None) -> None:
    if data is None:
        res_path = "eval/results/experiment_results.json"
        if not os.path.exists(res_path):
            print(f"Error: Could not find {res_path}. Run experiments.py first.")
            return
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    print("Generating Markdown tables...")
    os.makedirs("eval/results", exist_ok=True)
    out_path = "eval/results/experiment_tables.md"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Results\n\n")
        
        # Exp 1
        f.write("## Experiment 1: Search Comparison (0-4 Misreads)\n")
        f.write("| Algorithm | Misreads | Recovery Rate | Nodes Expanded | Nodes Generated | Time (s) |\n")
        f.write("|-----------|----------|---------------|----------------|-----------------|----------|\n")
        exp1 = data["experiment_1"]["summary"]
        for algo in ["AStar", "UCS", "Greedy"]:
            for m in [0, 1, 2, 3, 4]:
                st = exp1[algo].get(m) if m in exp1[algo] else exp1[algo].get(str(m))
                f.write(f"| {algo} | {m} | {st['recovery_rate_mean']*100:.1f}% | {st['nodes_expanded_mean']:.1f} +/- {st['nodes_expanded_std']:.1f} | {st['nodes_generated_mean']:.1f} +/- {st['nodes_generated_std']:.1f} | {st['time_mean']:.4f} |\n")
        
        # Exp 2
        f.write("\n## Experiment 2: CSP Pruning Comparison\n")
        f.write("| Givens | Algorithm | Assignments | Backtracks | Time (s) |\n")
        f.write("|--------|-----------|-------------|------------|----------|\n")
        exp2 = data["experiment_2"]["summary"]
        mode_labels = {
            "BT": "BT alone",
            "FC": "BT + FC",
            "AC3": "BT + FC + AC-3"
        }
        for givens in ["25", "30", "35"]:
            for mode in ["BT", "FC", "AC3"]:
                st = exp2[mode].get(givens) or exp2[mode].get(int(givens))
                lbl = mode_labels[mode]
                f.write(f"| {givens} | {lbl} | {st['assignments_mean']:.1f} +/- {st['assignments_std']:.1f} | {st['backtracks_mean']:.1f} +/- {st['backtracks_std']:.1f} | {st['time_mean']:.4f} |\n")
                
        # Exp 3
        f.write("\n## Experiment 3: Guidance Impact (2 Misreads)\n")
        f.write("| Strategy | Recovery Rate | Nodes Expanded | Time (s) |\n")
        f.write("|----------|---------------|----------------|----------|\n")
        exp3 = data["experiment_3"]["summary"]
        for strat in ["NN_Guided", "Random_Correction", "No_Correction"]:
            st = exp3[strat]
            f.write(f"| {strat} | {st['recovery_rate_mean']*100:.1f}% | {st['nodes_expanded_mean']:.1f} +/- {st['nodes_expanded_std']:.1f} | {st['time_mean']:.4f} |\n")
            
    print(f"Tables written to {out_path}")

def generate_plots(data: dict = None) -> None:
    if data is None:
        res_path = "eval/results/experiment_results.json"
        if not os.path.exists(res_path):
            print(f"Error: Could not find {res_path}. Run experiments.py first.")
            return
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    print("Generating plots...")
    os.makedirs("eval/plots", exist_ok=True)
    
    # -------------------------------------------------------------
    # Plot 1: Exp 1 Search Comparison across 0-4 Misreads
    # -------------------------------------------------------------
    exp1 = data["experiment_1"]["summary"]
    misreads = [0, 1, 2, 3, 4]
    algos = ["AStar", "UCS", "Greedy"]
    algo_labels = {"AStar": "A* Search", "UCS": "UCS (Uniform Cost)", "Greedy": "Greedy Best-First"}
    colors = {"AStar": "#2563eb", "UCS": "#059669", "Greedy": "#d97706"}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for algo in algos:
        rec_rates = [
            (exp1[algo].get(m) or exp1[algo].get(str(m)))["recovery_rate_mean"] * 100
            for m in misreads
        ]
        nodes_exp = [
            (exp1[algo].get(m) or exp1[algo].get(str(m)))["nodes_expanded_mean"]
            for m in misreads
        ]
        lbl = algo_labels[algo]
        c = colors[algo]
        
        ax1.plot(misreads, rec_rates, marker='o', linewidth=2, label=lbl, color=c)
        ax2.plot(misreads, nodes_exp, marker='s', linewidth=2, label=lbl, color=c)
        
    ax1.axvline(x=3, color='#dc2626', linestyle='--', alpha=0.7, label='Max 3 Corrections Budget')
    ax1.set_xlabel('Number of Injected Misreads')
    ax1.set_ylabel('Exact Recovery Rate (%)')
    ax1.set_title('Recovery Rate vs. Misreads (Exp 1)')
    ax1.set_xticks(misreads)
    ax1.set_ylim(-5, 110)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left')
    
    ax2.set_xlabel('Number of Injected Misreads')
    ax2.set_ylabel('Nodes Expanded (Mean)')
    ax2.set_title('Search Nodes Expanded vs. Misreads (Exp 1)')
    ax2.set_xticks(misreads)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("eval/plots/exp1_search_comparison.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: Exp 2 CSP Pruning Mechanisms (Backtracks on Log Scale)
    # -------------------------------------------------------------
    exp2 = data["experiment_2"]["summary"]
    givens = ["25", "30", "35"]
    
    bt_means = [(exp2["BT"].get(g) or exp2["BT"].get(int(g)))["backtracks_mean"] for g in givens]
    fc_means = [(exp2["FC"].get(g) or exp2["FC"].get(int(g)))["backtracks_mean"] for g in givens]
    ac3_means = [(exp2["AC3"].get(g) or exp2["AC3"].get(int(g)))["backtracks_mean"] for g in givens]
    
    plt.figure(figsize=(8, 5))
    x = range(len(givens))
    width = 0.25
    
    plt.bar([pos - width for pos in x], bt_means, width, label='BT alone', color='#dc2626')
    plt.bar(x, fc_means, width, label='BT + FC', color='#2563eb')
    plt.bar([pos + width for pos in x], ac3_means, width, label='BT + FC + AC-3', color='#059669')
    
    plt.yscale('log')
    plt.xlabel('Number of Givens (Clues)')
    plt.ylabel('Backtracks (Log Scale)')
    plt.title('CSP Solver Efficiency by Pruning Method (Exp 2)')
    plt.xticks(x, givens)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig("eval/plots/exp2_backtracks.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 3: Exp 3 Neural Network Guidance Impact
    # -------------------------------------------------------------
    exp3 = data["experiment_3"]["summary"]
    strats = ["NN_Guided", "Random_Correction", "No_Correction"]
    strat_labels = ["NN-Guided (A*)", "Random Guidance", "No Correction"]
    
    recs = [exp3[s]["recovery_rate_mean"] * 100 for s in strats]
    nodes = [exp3[s]["nodes_expanded_mean"] for s in strats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_colors = ["#059669", "#d97706", "#dc2626"]
    
    ax1.bar(strat_labels, recs, color=bar_colors, width=0.55)
    ax1.set_ylabel('Recovery Rate (%)')
    ax1.set_title('Recovery Rate by Guidance Strategy (Exp 3)')
    ax1.set_ylim(0, 115)
    for i, v in enumerate(recs):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    ax2.bar(strat_labels, nodes, color=bar_colors, width=0.55)
    ax2.set_ylabel('Nodes Expanded (Mean)')
    ax2.set_title('Search Nodes Expanded by Strategy (Exp 3)')
    for i, v in enumerate(nodes):
        ax2.text(i, v + 10, f"{v:.1f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("eval/plots/exp3_guidance_impact.png", dpi=150)
    plt.close()

    print("Plots saved in eval/plots/ (exp1_search_comparison.png, exp2_backtracks.png, exp3_guidance_impact.png)")

def main():
    res_path = "eval/results/experiment_results.json"
    if not os.path.exists(res_path):
        print(f"Error: Could not find {res_path}. Run experiments.py first.")
        return
        
    with open(res_path, 'r') as f:
        data = json.load(f)
        
    generate_markdown_tables(data)
    generate_plots(data)

# Backward compatibility alias
generate_experiment_plots = generate_plots

if __name__ == "__main__":
    main()
