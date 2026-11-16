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
    
    # Plot 1: Exp 2 Backtracks (Log Scale)
    exp2 = data["experiment_2"]["summary"]
    givens = ["25", "30", "35"]
    
    bt_means = [(exp2["BT"].get(g) or exp2["BT"].get(int(g)))["backtracks_mean"] for g in givens]
    fc_means = [(exp2["FC"].get(g) or exp2["FC"].get(int(g)))["backtracks_mean"] for g in givens]
    ac3_means = [(exp2["AC3"].get(g) or exp2["AC3"].get(int(g)))["backtracks_mean"] for g in givens]
    
    plt.figure(figsize=(8, 5))
    x = range(len(givens))
    width = 0.25
    
    plt.bar([pos - width for pos in x], bt_means, width, label='BT alone')
    plt.bar(x, fc_means, width, label='BT + FC')
    plt.bar([pos + width for pos in x], ac3_means, width, label='BT + FC + AC-3')
    
    plt.yscale('log')
    plt.xlabel('Number of Givens')
    plt.ylabel('Backtracks (Log Scale)')
    plt.title('CSP Solver Efficiency by Pruning Method')
    plt.xticks(x, givens)
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval/plots/exp2_backtracks.png")
    plt.close()
    
    print("Plots saved in eval/plots/")

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
