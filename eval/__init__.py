"""
Evaluation module initialization
"""
from eval.experiments import run_all_experiments
from eval.plot_results import generate_markdown_tables, generate_experiment_plots

__all__ = ["run_all_experiments", "generate_markdown_tables", "generate_experiment_plots"]
