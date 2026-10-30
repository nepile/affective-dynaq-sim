"""Evaluation package for Dyna-Q reinforcement learning agents.

Provides comprehensive metric calculation, statistical analysis, and publication-ready
visualizations for both the initial learning phase (80%) and the stable convergence phase (20%).
"""

from src.evaluation.comparison import (
    generate_group_comparison,
    merge_and_compare_groups,
    plot_combined_comparison_dashboard,
)
from src.evaluation.exporter import export_evaluation_suite
from src.evaluation.metrics import (
    compute_cer_ci95,
    compute_cer_cv,
    compute_cer_rme,
    compute_cer_std,
    compute_convergence_metrics,
    compute_gaussian_pdf,
    compute_stable_phase_statistics,
    split_learning_phases,
)
from src.evaluation.visualizer import (
    plot_collision_rate,
    plot_coefficient_of_variation,
    plot_convergence_speed,
    plot_current_cumulative_reward,
    plot_current_episodic_reward,
    plot_normal_distribution,
    plot_relative_margin_error,
    plot_steps_per_episode,
)

__all__ = [
    "split_learning_phases",
    "compute_cer_std",
    "compute_cer_ci95",
    "compute_cer_cv",
    "compute_cer_rme",
    "compute_gaussian_pdf",
    "compute_stable_phase_statistics",
    "compute_convergence_metrics",
    "export_evaluation_suite",
    "generate_group_comparison",
    "merge_and_compare_groups",
    "plot_combined_comparison_dashboard",
    "plot_steps_per_episode",
    "plot_current_cumulative_reward",
    "plot_current_episodic_reward",
    "plot_convergence_speed",
    "plot_collision_rate",
    "plot_coefficient_of_variation",
    "plot_relative_margin_error",
    "plot_normal_distribution",
]

