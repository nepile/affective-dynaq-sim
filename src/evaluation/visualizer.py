"""Visualization generator for learning evaluation metrics across learning phases.

Produces publication-grade plots (300 DPI) for each of the 8 metrics in any phase
(Initial Phase 80% and Stable Phase 20%), plus comprehensive multi-panel dashboards.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLOR_PRIMARY = "#2980b9"   # Strong blue
COLOR_SECONDARY = "#e67e22" # Orange
COLOR_ACCENT = "#27ae60"    # Green
COLOR_RED = "#c0392b"       # Crimson red
COLOR_DARK = "#2c3e50"      # Navy dark
COLOR_GRID = "#e0e0e0"      # Subtle grid line
SEED_COLORS = ["#2980b9", "#e67e22", "#27ae60", "#8e44ad", "#d35400", "#16a085"]


def _apply_plot_style(ax: plt.Axes) -> None:
    """Apply consistent styling to Matplotlib axes."""
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.6, color=COLOR_GRID, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7f8c8d")
    ax.spines["bottom"].set_color("#7f8c8d")
    ax.tick_params(colors="#34495e", labelsize=9)


_apply_style = _apply_plot_style


def plot_steps_per_episode(
    seed_episodes: Dict[int, List[Dict[str, Any]]],
    output_path: str,
    phase_name: str = "Initial Phase",
    dpi: int = 300,
) -> None:
    """Plot steps per episode across the specified learning phase.

    Args:
        seed_episodes (Dict[int, List[Dict[str, Any]]]): Episode logs per seed for this phase.
        output_path (str): Destination PNG path.
        phase_name (str, optional): Label for the phase. Defaults to "Initial Phase".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
    _apply_plot_style(ax)

    steps_matrix: List[List[float]] = []
    episodes_x: List[int] = []

    for i, (seed, eps) in enumerate(seed_episodes.items()):
        sub_eps = [e["steps"] for e in eps]
        steps_matrix.append(sub_eps)
        if not episodes_x:
            episodes_x = [e["episode"] for e in eps]
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(
            episodes_x[: len(sub_eps)],
            sub_eps,
            label=f"Seed {seed}",
            color=color,
            alpha=0.45,
            linewidth=1.2,
        )

    if steps_matrix and episodes_x:
        min_len = min(len(s) for s in steps_matrix)
        mean_steps = [np.mean([s[k] for s in steps_matrix]) for k in range(min_len)]
        std_steps = [np.std([s[k] for s in steps_matrix]) for k in range(min_len)]
        mean_x = episodes_x[:min_len]

        ax.plot(mean_x, mean_steps, label="Mean Steps", color=COLOR_DARK, linewidth=2.4)
        ax.fill_between(
            mean_x,
            [max(0, m - s) for m, s in zip(mean_steps, std_steps)],
            [m + s for m, s in zip(mean_steps, std_steps)],
            color=COLOR_DARK,
            alpha=0.15,
            label="±1 Std Dev",
        )
        ax.set_xlim(episodes_x[0], episodes_x[-1])

    ax.set_title(f"Steps per Episode ({phase_name})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax.set_ylabel("Steps to Goal", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_current_cumulative_reward(
    seed_episodes: Dict[int, List[Dict[str, Any]]],
    output_path: str,
    phase_name: str = "Initial Phase",
    dpi: int = 300,
) -> None:
    """Plot Current Cumulative Reward (CCR) across the specified learning phase.

    Args:
        seed_episodes (Dict[int, List[Dict[str, Any]]]): Episode logs per seed for this phase.
        output_path (str): Destination PNG path.
        phase_name (str, optional): Label for the phase. Defaults to "Initial Phase".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
    _apply_plot_style(ax)

    ccr_matrix: List[List[float]] = []
    episodes_x: List[int] = []

    for i, (seed, eps) in enumerate(seed_episodes.items()):
        sub_rewards = [e["reward"] for e in eps]
        ccr_vals = list(np.cumsum(sub_rewards))
        ccr_matrix.append(ccr_vals)
        if not episodes_x:
            episodes_x = [e["episode"] for e in eps]
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(
            episodes_x[: len(ccr_vals)],
            ccr_vals,
            label=f"Seed {seed}",
            color=color,
            alpha=0.45,
            linewidth=1.2,
        )

    if ccr_matrix and episodes_x:
        min_len = min(len(c) for c in ccr_matrix)
        mean_ccr = [np.mean([c[k] for c in ccr_matrix]) for k in range(min_len)]
        std_ccr = [np.std([c[k] for c in ccr_matrix]) for k in range(min_len)]
        mean_x = episodes_x[:min_len]

        ax.plot(mean_x, mean_ccr, label="Mean CCR", color=COLOR_PRIMARY, linewidth=2.4)
        ax.fill_between(
            mean_x,
            [m - s for m, s in zip(mean_ccr, std_ccr)],
            [m + s for m, s in zip(mean_ccr, std_ccr)],
            color=COLOR_PRIMARY,
            alpha=0.15,
            label="±1 Std Dev",
        )
        ax.set_xlim(episodes_x[0], episodes_x[-1])

    ax.set_title(f"Current Cumulative Reward (CCR) ({phase_name})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative Total Reward", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_current_episodic_reward(
    seed_episodes: Dict[int, List[Dict[str, Any]]],
    output_path: str,
    phase_name: str = "Initial Phase",
    dpi: int = 300,
) -> None:
    """Plot Current Episodic Reward (CER) across the specified learning phase.

    Args:
        seed_episodes (Dict[int, List[Dict[str, Any]]]): Episode logs per seed for this phase.
        output_path (str): Destination PNG path.
        phase_name (str, optional): Label for the phase. Defaults to "Initial Phase".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
    _apply_plot_style(ax)

    cer_matrix: List[List[float]] = []
    episodes_x: List[int] = []

    for i, (seed, eps) in enumerate(seed_episodes.items()):
        sub_cer = [e["reward"] for e in eps]
        cer_matrix.append(sub_cer)
        if not episodes_x:
            episodes_x = [e["episode"] for e in eps]
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(
            episodes_x[: len(sub_cer)],
            sub_cer,
            label=f"Seed {seed}",
            color=color,
            alpha=0.35,
            linewidth=1.0,
        )

    if cer_matrix and episodes_x:
        min_len = min(len(c) for c in cer_matrix)
        mean_cer = [np.mean([c[k] for c in cer_matrix]) for k in range(min_len)]
        mean_x = episodes_x[:min_len]

        window = max(2, min(10, min_len // 8))
        ma_cer = [
            np.mean(mean_cer[max(0, k - window + 1) : k + 1])
            for k in range(min_len)
        ]

        ax.plot(mean_x, mean_cer, label="Raw Mean CER", color="#95a5a6", linewidth=1.2, alpha=0.6)
        ax.plot(mean_x, ma_cer, label=f"Mean CER Trend (MA-{window})", color=COLOR_ACCENT, linewidth=2.5)
        ax.set_xlim(episodes_x[0], episodes_x[-1])

    ax.set_title(f"Current Episodic Reward (CER) ({phase_name})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax.set_ylabel("Episodic Reward (CER)", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_convergence_speed(
    seed_episodes: Dict[int, List[Dict[str, Any]]],
    conv_metrics: Dict[int, Dict[str, Any]],
    output_path: str,
    phase_name: str = "Initial Phase",
    dpi: int = 300,
) -> None:
    """Plot convergence speed trajectory with convergence episode markers.

    Args:
        seed_episodes (Dict[int, List[Dict[str, Any]]]): Episode logs per seed for this phase.
        conv_metrics (Dict[int, Dict[str, Any]]): Convergence stats per seed.
        output_path (str): Destination PNG path.
        phase_name (str, optional): Label for the phase. Defaults to "Initial Phase".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
    _apply_plot_style(ax)

    summary_text = []
    first_eps = next(iter(seed_episodes.values()), [])
    episodes_x = [e["episode"] for e in first_eps] if first_eps else []

    for i, (seed, conv) in enumerate(conv_metrics.items()):
        ma_steps = conv["moving_avg_steps"]
        color = SEED_COLORS[i % len(SEED_COLORS)]
        plot_x = episodes_x[: len(ma_steps)] if episodes_x else list(range(1, len(ma_steps) + 1))
        ax.plot(
            plot_x,
            ma_steps,
            label=f"Seed {seed} (Moving Avg)",
            color=color,
            linewidth=1.8,
        )

        conv_ep = conv["convergence_episode"]
        if conv_ep:
            ax.axvline(x=conv_ep, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
            idx = conv_ep - (episodes_x[0] if episodes_x else 1)
            if 0 <= idx < len(ma_steps):
                ax.scatter([conv_ep], [ma_steps[idx]], color=color, s=50, zorder=5)
            summary_text.append(f"Seed {seed}: Conv at Ep {conv_ep}")
        else:
            summary_text.append(f"Seed {seed}: In Progress")

    first_conv = next(iter(conv_metrics.values()), None)
    if first_conv and first_conv["optimal_steps"] > 0:
        opt = first_conv["optimal_steps"]
        ax.axhline(y=opt, color=COLOR_RED, linestyle=":", linewidth=1.5, label=f"Optimal Geodesic ({opt} steps)")

    ax.set_title(f"Convergence Speed ({phase_name})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax.set_ylabel("Moving Average Steps to Goal", fontsize=10, fontweight="bold")
    if episodes_x:
        ax.set_xlim(episodes_x[0], episodes_x[-1])

    if summary_text:
        info_str = "\n".join(summary_text)
        ax.text(
            0.98, 0.95, info_str,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.9),
        )

    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_collision_rate(
    seed_episodes: Dict[int, List[Dict[str, Any]]],
    output_path: str,
    phase_name: str = "Stable Phase",
    dpi: int = 300,
) -> None:
    """Plot collision rate across episodes in the specified learning phase.

    Args:
        seed_episodes (Dict[int, List[Dict[str, Any]]]): Episode logs per seed for this phase.
        output_path (str): Destination PNG path.
        phase_name (str, optional): Label for the phase. Defaults to "Stable Phase".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
    _apply_plot_style(ax)

    first_eps = next(iter(seed_episodes.values()), [])
    episodes_x = [e["episode"] for e in first_eps] if first_eps else []
    collision_rates_matrix: List[List[float]] = []

    for i, (seed, eps) in enumerate(seed_episodes.items()):
        rates = [
            (e.get("collisions", 0) / max(1, e.get("steps", 1))) * 100.0
            for e in eps
        ]
        collision_rates_matrix.append(rates)
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(
            episodes_x[: len(rates)],
            rates,
            marker="o",
            markersize=3.5,
            linestyle="-",
            label=f"Seed {seed}",
            color=color,
            alpha=0.6,
            linewidth=1.2,
        )

    if collision_rates_matrix and episodes_x:
        min_len = min(len(r) for r in collision_rates_matrix)
        mean_rates = [np.mean([r[k] for r in collision_rates_matrix]) for k in range(min_len)]
        ax.plot(
            episodes_x[:min_len],
            mean_rates,
            label="Mean Collision Rate (%)",
            color=COLOR_RED,
            linewidth=2.2,
        )
        overall_mean = float(np.mean(mean_rates))
        ax.axhline(
            y=overall_mean,
            color=COLOR_RED,
            linestyle="--",
            linewidth=1.0,
            label=f"Overall Avg ({overall_mean:.2f}%)",
        )
        ax.set_xlim(episodes_x[0], episodes_x[-1])

    ax.set_title(f"Collision Rate ({phase_name})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax.set_ylabel("Collision Rate (% of Steps)", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_coefficient_of_variation(
    stats_per_seed: Dict[int, Dict[str, Any]],
    combined_stats: Dict[str, Any],
    output_path: str,
    phase_name: str = "Stable Phase",
    dpi: int = 300,
) -> None:
    """Plot Coefficient of Variation (CV) of CER during the specified learning phase.

    Args:
        stats_per_seed (Dict[int, Dict[str, Any]]): Statistics per seed.
        combined_stats (Dict[str, Any]): Aggregated statistics.
        output_path (str): Destination PNG path.
        phase_name (str, optional): Label for the phase. Defaults to "Stable Phase".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=dpi)
    _apply_plot_style(ax1)
    _apply_plot_style(ax2)

    seeds = list(stats_per_seed.keys())
    labels = [f"Seed {s}" for s in seeds] + ["Overall"]
    cv_values = [stats_per_seed[s]["cer_cv"] for s in seeds] + [combined_stats["cer_cv"]]
    cer_avgs = [stats_per_seed[s]["cer_avg"] for s in seeds] + [combined_stats["cer_avg"]]
    cer_stds = [stats_per_seed[s]["cer_std"] for s in seeds] + [combined_stats["cer_std"]]

    bar_colors = [SEED_COLORS[i % len(SEED_COLORS)] for i in range(len(seeds))] + [COLOR_DARK]

    bars = ax1.bar(labels, cv_values, color=bar_colors, width=0.55, edgecolor="#2c3e50", linewidth=0.8)
    for bar, val in zip(bars, cv_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(0.5, height * 0.03),
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.set_title("Coefficient of Variation (CER_cv)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("CV (%) = (CER_std / |CER_avg|) * 100", fontsize=9, fontweight="bold")

    x_indices = list(range(len(labels)))
    ax2.errorbar(
        x_indices,
        cer_avgs,
        yerr=cer_stds,
        fmt="o",
        color=COLOR_PRIMARY,
        ecolor=COLOR_SECONDARY,
        elinewidth=2.0,
        capsize=6,
        capthick=1.5,
        markersize=6,
    )
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels)
    ax2.set_title("Mean CER ± CER_std Dispersion", fontsize=11, fontweight="bold")
    ax2.set_ylabel("CER Value", fontsize=9, fontweight="bold")

    fig.suptitle(f"Coefficient of Variation (CV) Analysis — {phase_name}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_relative_margin_error(
    stats_per_seed: Dict[int, Dict[str, Any]],
    combined_stats: Dict[str, Any],
    output_path: str,
    phase_name: str = "Stable Phase",
    dpi: int = 300,
) -> None:
    """Plot Relative Margin Error (RME) and 95% Confidence Interval for CER.

    Args:
        stats_per_seed (Dict[int, Dict[str, Any]]): Statistics per seed.
        combined_stats (Dict[str, Any]): Aggregated statistics.
        output_path (str): Destination PNG path.
        phase_name (str, optional): Label for the phase. Defaults to "Stable Phase".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=dpi)
    _apply_plot_style(ax1)
    _apply_plot_style(ax2)

    seeds = list(stats_per_seed.keys())
    labels = [f"Seed {s}" for s in seeds] + ["Overall"]
    rme_values = [stats_per_seed[s]["cer_rme"] for s in seeds] + [combined_stats["cer_rme"]]
    cer_avgs = [stats_per_seed[s]["cer_avg"] for s in seeds] + [combined_stats["cer_avg"]]
    ci95_values = [stats_per_seed[s]["cer_ci95"] for s in seeds] + [combined_stats["cer_ci95"]]

    bar_colors = [SEED_COLORS[i % len(SEED_COLORS)] for i in range(len(seeds))] + [COLOR_DARK]

    bars = ax1.bar(labels, rme_values, color=bar_colors, width=0.55, edgecolor="#2c3e50", linewidth=0.8)
    for bar, val in zip(bars, rme_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(0.2, height * 0.03),
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.set_title("Relative Margin Error (CER_rme)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RME (%) = (CER_ci95 / |CER_avg|) * 100", fontsize=9, fontweight="bold")

    x_indices = list(range(len(labels)))
    ax2.errorbar(
        x_indices,
        cer_avgs,
        yerr=ci95_values,
        fmt="s",
        color=COLOR_PRIMARY,
        ecolor=COLOR_RED,
        elinewidth=2.2,
        capsize=6,
        capthick=1.8,
        markersize=6,
    )
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels)
    ax2.set_title("Mean CER with 95% Confidence Interval", fontsize=11, fontweight="bold")
    ax2.set_ylabel("CER (95% CI Range)", fontsize=9, fontweight="bold")

    fig.suptitle(f"Relative Margin Error (RME) & 95% CI — {phase_name}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_normal_distribution(
    combined_stats: Dict[str, Any],
    raw_cer_values: List[float],
    output_path: str,
    phase_name: str = "Stable Phase",
    dpi: int = 300,
) -> None:
    """Plot Gaussian Normal Distribution curve of CER in the specified phase.

    Args:
        combined_stats (Dict[str, Any]): Combined statistics for this phase.
        raw_cer_values (List[float]): Actual empirical CER values in this phase.
        output_path (str): Destination PNG path.
        phase_name (str, optional): Label for the phase. Defaults to "Stable Phase".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
    _apply_plot_style(ax)

    mu = combined_stats["cer_avg"]
    sigma = max(1e-4, combined_stats["cer_std"])
    ci_lower = combined_stats["ci_lower"]
    ci_upper = combined_stats["ci_upper"]
    variance = combined_stats["cer_var"]
    rme = combined_stats["cer_rme"]

    x_min = min(min(raw_cer_values) - 0.5 * sigma, mu - 3.5 * sigma) if raw_cer_values else mu - 3.5 * sigma
    x_max = max(max(raw_cer_values) + 0.5 * sigma, mu + 3.5 * sigma) if raw_cer_values else mu + 3.5 * sigma
    x_vals = np.linspace(x_min, x_max, 300)
    pdf_vals = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (((x_vals - mu) / sigma) ** 2))

    if raw_cer_values:
        ax.hist(
            raw_cer_values,
            bins=max(5, min(15, len(raw_cer_values) // 3)),
            density=True,
            alpha=0.3,
            color="#95a5a6",
            edgecolor="#7f8c8d",
            label="Empirical CER Distribution",
        )

    ax.plot(x_vals, pdf_vals, color=COLOR_PRIMARY, linewidth=2.4, label="Gaussian PDF f(x)")

    ci_mask = (x_vals >= ci_lower) & (x_vals <= ci_upper)
    ax.fill_between(
        x_vals[ci_mask],
        pdf_vals[ci_mask],
        color=COLOR_ACCENT,
        alpha=0.25,
        label=f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]",
    )

    ax.axvline(x=mu, color=COLOR_RED, linestyle="--", linewidth=1.5, label=f"Mean μ = {mu:.2f}")

    stat_card = (
        f"Mean (μ): {mu:.2f}\n"
        f"Std Dev (σ): {sigma:.2f}\n"
        f"Variance (σ²): {variance:.2f}\n"
        f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]\n"
        f"RME: {rme:.2f}%"
    )
    ax.text(
        0.97, 0.95, stat_card,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.set_title(f"Gaussian Normal Distribution of CER ({phase_name})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Cumulative Episodic Reward (CER)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Probability Density f(x)", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_combined_phase_dashboard(
    phase_name: str,
    seed_episodes: Dict[int, List[Dict[str, Any]]],
    conv_metrics: Dict[int, Dict[str, Any]],
    stats_per_seed: Dict[int, Dict[str, Any]],
    combined_stats: Dict[str, Any],
    raw_cer_values: List[float],
    output_path: str,
    dpi: int = 300,
) -> None:
    """Generate a single 8-panel dashboard combining all 8 metrics for this learning phase.

    Args:
        phase_name (str): Label of phase (e.g. 'Initial Phase (80%)', 'Stable Phase (20%)').
        seed_episodes (Dict[int, List[Dict[str, Any]]]): Episode logs per seed for this phase.
        conv_metrics (Dict[int, Dict[str, Any]]): Convergence stats per seed.
        stats_per_seed (Dict[int, Dict[str, Any]]): Statistical parameters per seed.
        combined_stats (Dict[str, Any]): Aggregated statistics.
        raw_cer_values (List[float]): Empirical CER values.
        output_path (str): File path for combined image.
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), dpi=dpi)

    first_eps = next(iter(seed_episodes.values()), [])
    episodes_x = [e["episode"] for e in first_eps] if first_eps else []

    # 1. Steps per Episode (Row 0, Col 0)
    ax1 = axes[0, 0]
    _apply_plot_style(ax1)
    steps_matrix = []
    for i, (seed, eps) in enumerate(seed_episodes.items()):
        sub_s = [e["steps"] for e in eps]
        steps_matrix.append(sub_s)
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax1.plot(episodes_x[:len(sub_s)], sub_s, label=f"Seed {seed}", color=color, alpha=0.45, linewidth=1.2)
    if steps_matrix and episodes_x:
        min_len = min(len(s) for s in steps_matrix)
        mean_steps = [np.mean([s[k] for s in steps_matrix]) for k in range(min_len)]
        ax1.plot(episodes_x[:min_len], mean_steps, label="Mean Steps", color=COLOR_DARK, linewidth=2.2)
        ax1.set_xlim(episodes_x[0], episodes_x[-1])
    ax1.set_title("1. Steps per Episode", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Steps to Goal", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

    # 2. Current Cumulative Reward (Row 0, Col 1)
    ax2 = axes[0, 1]
    _apply_plot_style(ax2)
    ccr_matrix = []
    for i, (seed, eps) in enumerate(seed_episodes.items()):
        sub_r = [e["reward"] for e in eps]
        ccr_vals = list(np.cumsum(sub_r))
        ccr_matrix.append(ccr_vals)
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax2.plot(episodes_x[:len(ccr_vals)], ccr_vals, label=f"Seed {seed}", color=color, alpha=0.45, linewidth=1.2)
    if ccr_matrix and episodes_x:
        min_len = min(len(c) for c in ccr_matrix)
        mean_ccr = [np.mean([c[k] for c in ccr_matrix]) for k in range(min_len)]
        ax2.plot(episodes_x[:min_len], mean_ccr, label="Mean CCR", color=COLOR_PRIMARY, linewidth=2.2)
        ax2.set_xlim(episodes_x[0], episodes_x[-1])
    ax2.set_title("2. Current Cumulative Reward (CCR)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Reward", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

    # 3. Current Episodic Reward (Row 1, Col 0)
    ax3 = axes[1, 0]
    _apply_plot_style(ax3)
    cer_matrix = []
    for i, (seed, eps) in enumerate(seed_episodes.items()):
        sub_cer = [e["reward"] for e in eps]
        cer_matrix.append(sub_cer)
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax3.plot(episodes_x[:len(sub_cer)], sub_cer, color=color, alpha=0.35, linewidth=1.0)
    if cer_matrix and episodes_x:
        min_len = min(len(c) for c in cer_matrix)
        mean_cer = [np.mean([c[k] for c in cer_matrix]) for k in range(min_len)]
        w = max(2, min(5, min_len // 8))
        ma_cer = [np.mean(mean_cer[max(0, k - w + 1):k + 1]) for k in range(min_len)]
        ax3.plot(episodes_x[:min_len], mean_cer, label="Raw Mean CER", color="#95a5a6", linewidth=1.0, alpha=0.6)
        ax3.plot(episodes_x[:min_len], ma_cer, label=f"Mean CER Trend (MA-{w})", color=COLOR_ACCENT, linewidth=2.2)
        ax3.set_xlim(episodes_x[0], episodes_x[-1])
    ax3.set_title("3. Current Episodic Reward (CER)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Episodic Reward", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

    # 4. Convergence Speed (Row 1, Col 1)
    ax4 = axes[1, 1]
    _apply_style(ax4)
    for i, (seed, conv) in enumerate(conv_metrics.items()):
        ma_steps = conv["moving_avg_steps"]
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax4.plot(episodes_x[:len(ma_steps)], ma_steps, label=f"Seed {seed}", color=color, linewidth=1.8)
        c_ep = conv["convergence_episode"]
        if c_ep and c_ep in episodes_x:
            ax4.axvline(x=c_ep, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
    first_conv = next(iter(conv_metrics.values()), None)
    if first_conv and first_conv["optimal_steps"] > 0:
        opt = first_conv["optimal_steps"]
        ax4.axhline(y=opt, color=COLOR_RED, linestyle=":", linewidth=1.5, label=f"Optimal ({opt} steps)")
    ax4.set_title("4. Convergence Speed (Steps Progression)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Moving Avg Steps", fontsize=9, fontweight="bold")
    if episodes_x:
        ax4.set_xlim(episodes_x[0], episodes_x[-1])
    ax4.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

    # 5. Collision Rate (Row 2, Col 0)
    ax5 = axes[2, 0]
    _apply_plot_style(ax5)
    coll_matrix = []
    for i, (seed, eps) in enumerate(seed_episodes.items()):
        rates = [(e.get("collisions", 0) / max(1, e.get("steps", 1))) * 100.0 for e in eps]
        coll_matrix.append(rates)
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax5.plot(episodes_x[:len(rates)], rates, marker="o", markersize=3.0, label=f"Seed {seed}", color=color, alpha=0.5, linewidth=1.2)
    if coll_matrix and episodes_x:
        min_len = min(len(r) for r in coll_matrix)
        mean_coll = [np.mean([r[k] for r in coll_matrix]) for k in range(min_len)]
        ax5.plot(episodes_x[:min_len], mean_coll, label="Mean Collision Rate (%)", color=COLOR_RED, linewidth=2.0)
        ax5.set_xlim(episodes_x[0], episodes_x[-1])
    ax5.set_title("5. Collision Rate (% of Steps)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Collision Rate (%)", fontsize=9, fontweight="bold")
    ax5.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

    # 6. Coefficient of Variation (Row 2, Col 1)
    ax6 = axes[2, 1]
    _apply_style(ax6)
    seeds = list(stats_per_seed.keys())
    labels = [f"Seed {s}" for s in seeds] + ["Overall"]
    cv_values = [stats_per_seed[s]["cer_cv"] for s in seeds] + [combined_stats["cer_cv"]]
    bar_colors = [SEED_COLORS[i % len(SEED_COLORS)] for i in range(len(seeds))] + [COLOR_DARK]
    bars6 = ax6.bar(labels, cv_values, color=bar_colors, width=0.45, edgecolor="#2c3e50")
    for bar, val in zip(bars6, cv_values):
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.2, h * 0.02), f"{val:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax6.set_title("6. Coefficient of Variation (CV %)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("CV (%) = (CER_std / |CER_avg|) * 100", fontsize=9, fontweight="bold")

    # 7. Relative Margin Error (Row 3, Col 0)
    ax7 = axes[3, 0]
    _apply_style(ax7)
    rme_values = [stats_per_seed[s]["cer_rme"] for s in seeds] + [combined_stats["cer_rme"]]
    bars7 = ax7.bar(labels, rme_values, color=bar_colors, width=0.45, edgecolor="#2c3e50")
    for bar, val in zip(bars7, rme_values):
        h = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.2, h * 0.02), f"{val:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax7.set_title("7. Relative Margin Error (RME %)", fontsize=11, fontweight="bold")
    ax7.set_ylabel("RME (%) = (CER_ci95 / |CER_avg|) * 100", fontsize=9, fontweight="bold")

    # 8. Normal Distribution (Row 3, Col 1)
    ax8 = axes[3, 1]
    _apply_style(ax8)
    mu = combined_stats["cer_avg"]
    sigma = max(1e-4, combined_stats["cer_std"])
    x_min = min(raw_cer_values) - 0.5 * sigma if raw_cer_values else mu - 3.5 * sigma
    x_max = max(raw_cer_values) + 0.5 * sigma if raw_cer_values else mu + 3.5 * sigma
    x_vals = np.linspace(x_min, x_max, 300)
    pdf_vals = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (((x_vals - mu) / sigma) ** 2))
    ax8.plot(x_vals, pdf_vals, color=COLOR_PRIMARY, linewidth=2.2, label=f"Gaussian PDF (μ={mu:.1f}, σ={sigma:.2f})")
    ax8.axvline(x=mu, color=COLOR_RED, linestyle="--", linewidth=1.2, label=f"Mean μ = {mu:.1f}")
    ax8.set_title("8. Gaussian Normal Distribution of CER", fontsize=11, fontweight="bold")
    ax8.set_ylabel("Density f(x)", fontsize=9, fontweight="bold")
    ax8.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle(f"Evaluation Metrics Dashboard — {phase_name}", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
