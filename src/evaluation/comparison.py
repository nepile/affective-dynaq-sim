"""Cross-experiment comparative evaluation suite for experimental groups.

When multiple experiments belong to the same group_name, this module generates
comparative analysis and visualization figures across all 8 learning metrics
inside::

    reports/<group_name>/comparison/
    ├── initial_phase/
    │   ├── steps_per_episode/
    │   ├── current_cumulative_reward/
    │   ├── current_episodic_reward/
    │   ├── convergence_speed/
    │   ├── collision_rate/
    │   ├── coefficient_of_variation/
    │   ├── relative_margin_error/
    │   ├── normal_distribution/
    │   └── all_metrics_comparison_initial.png
    ├── stable_phase/
    │   ├── steps_per_episode/
    │   ├── current_cumulative_reward/
    │   ├── current_episodic_reward/
    │   ├── convergence_speed/
    │   ├── collision_rate/
    │   ├── coefficient_of_variation/
    │   ├── relative_margin_error/
    │   ├── normal_distribution/
    │   └── all_metrics_comparison_stable.png
    └── all_metrics_comparison.png (Executive Master Dashboard)
"""

import csv
import os
import shutil
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXP_COLORS = ["#2980b9", "#e67e22", "#27ae60", "#8e44ad", "#c0392b", "#16a085", "#d35400", "#2c3e50"]


def _apply_style(ax: plt.Axes) -> None:
    """Apply standard clean styling to Matplotlib axes."""
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.6, color="#e0e0e0", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7f8c8d")
    ax.spines["bottom"].set_color("#7f8c8d")
    ax.tick_params(colors="#34495e", labelsize=9)


def plot_combined_comparison_dashboard(
    group_name: str,
    exp_steps_data: Dict[str, Dict[int, float]],
    exp_ccr_data: Dict[str, Dict[int, float]],
    exp_cer_data: Dict[str, Dict[int, float]],
    conv_summary: List[Dict[str, Any]],
    exp_coll_data: Dict[str, Dict[int, float]],
    cv_summary: List[Dict[str, Any]],
    rme_summary: List[Dict[str, Any]],
    gauss_params: List[Dict[str, Any]],
    output_path: str,
    phase_title: str = "",
    dpi: int = 300,
) -> None:
    """Generate a single unified 8-panel dashboard combining all comparison plots.

    Layout:
    [ (1) Steps per Episode ] [ (2) Current Cumulative Reward ]
    [ (3) Episodic Reward   ] [ (4) Convergence Speed         ]
    [ (5) Collision Rate    ] [ (6) Coefficient of Variation  ]
    [ (7) Rel Margin Error  ] [ (8) Normal Distribution       ]

    Args:
        group_name (str): Experimental group name.
        exp_steps_data (Dict[str, Dict[int, float]]): Steps per episode per experiment.
        exp_ccr_data (Dict[str, Dict[int, float]]): Cumulative reward per experiment.
        exp_cer_data (Dict[str, Dict[int, float]]): Episodic reward per experiment.
        conv_summary (List[Dict[str, Any]]): Convergence stats per experiment.
        exp_coll_data (Dict[str, Dict[int, float]]): Collision rate per experiment.
        cv_summary (List[Dict[str, Any]]): CV % summary per experiment.
        rme_summary (List[Dict[str, Any]]): RME % summary per experiment.
        gauss_params (List[Dict[str, Any]]): Gaussian normal parameters per experiment.
        output_path (str): Destination file path for combined image.
        phase_title (str, optional): Title label for phase. Defaults to "".
        dpi (int, optional): Image resolution. Defaults to 300.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), dpi=dpi)
    tag = f" ({phase_title})" if phase_title else ""

    # 1. Steps per Episode (Row 0, Col 0)
    ax1 = axes[0, 0]
    _apply_style(ax1)
    if exp_steps_data:
        for i, (exp, data) in enumerate(exp_steps_data.items()):
            eps_x = sorted(data.keys())
            vals_y = [data[e] for e in eps_x]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax1.plot(eps_x, vals_y, label=exp, color=color, linewidth=2.0)
        ax1.set_title(f"1. Steps per Episode Comparison{tag}", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Episode", fontsize=9, fontweight="bold")
        ax1.set_ylabel("Mean Steps to Goal", fontsize=9, fontweight="bold")
        ax1.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    # 2. Current Cumulative Reward (CCR) (Row 0, Col 1)
    ax2 = axes[0, 1]
    _apply_style(ax2)
    if exp_ccr_data:
        for i, (exp, data) in enumerate(exp_ccr_data.items()):
            eps_x = sorted(data.keys())
            vals_y = [data[e] for e in eps_x]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax2.plot(eps_x, vals_y, label=exp, color=color, linewidth=2.0)
        ax2.set_title(f"2. Current Cumulative Reward (CCR) Comparison{tag}", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Episode", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Mean Cumulative Reward", fontsize=9, fontweight="bold")
        ax2.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    # 3. Current Episodic Reward (CER) (Row 1, Col 0)
    ax3 = axes[1, 0]
    _apply_style(ax3)
    if exp_cer_data:
        for i, (exp, data) in enumerate(exp_cer_data.items()):
            eps_x = sorted(data.keys())
            vals_y = [data[e] for e in eps_x]
            w = max(2, min(5, len(vals_y) // 8))
            smoothed = [np.mean(vals_y[max(0, k - w + 1):k + 1]) for k in range(len(vals_y))]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax3.plot(eps_x, smoothed, label=f"{exp} (trend)", color=color, linewidth=2.0)
            ax3.plot(eps_x, vals_y, color=color, alpha=0.25, linewidth=0.8)
        ax3.set_title(f"3. Current Episodic Reward (CER) Comparison{tag}", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Episode", fontsize=9, fontweight="bold")
        ax3.set_ylabel("Episodic Reward", fontsize=9, fontweight="bold")
        ax3.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    # 4. Convergence Speed (Row 1, Col 1)
    ax4 = axes[1, 1]
    _apply_style(ax4)
    if conv_summary:
        exp_labels = [c["experiment"] for c in conv_summary]
        y_vals = [c["avg_episodes_to_converge"] for c in conv_summary]
        colors = [EXP_COLORS[i % len(EXP_COLORS)] for i in range(len(exp_labels))]
        bars = ax4.bar(exp_labels, y_vals, color=colors, width=0.45, edgecolor="#2c3e50", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.5, h * 0.02), f"{h:.1f} ep", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax4.set_title(f"4. Convergence Speed Comparison{tag}\n(Fewer episodes = Faster Convergence)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Episodes to Converge", fontsize=9, fontweight="bold")

    # 5. Collision Rate (Row 2, Col 0)
    ax5 = axes[2, 0]
    _apply_style(ax5)
    if exp_coll_data:
        for i, (exp, data) in enumerate(exp_coll_data.items()):
            eps_x = sorted(data.keys())
            vals_y = [data[e] for e in eps_x]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax5.plot(eps_x, vals_y, marker="o", markersize=3.5, label=f"{exp} (Mean: {np.mean(vals_y):.1f}%)", color=color, linewidth=1.5)
        ax5.set_title(f"5. Collision Rate Comparison{tag}", fontsize=11, fontweight="bold")
        ax5.set_xlabel("Episode", fontsize=9, fontweight="bold")
        ax5.set_ylabel("Collision Rate (% of Steps)", fontsize=9, fontweight="bold")
        ax5.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    # 6. Coefficient of Variation (CV) (Row 2, Col 1)
    ax6 = axes[2, 1]
    _apply_style(ax6)
    if cv_summary:
        labels = [c["experiment"] for c in cv_summary]
        y_vals = [c["CER_cv_percent"] for c in cv_summary]
        colors = [EXP_COLORS[i % len(EXP_COLORS)] for i in range(len(labels))]
        bars = ax6.bar(labels, y_vals, color=colors, width=0.45, edgecolor="#2c3e50", linewidth=0.8)
        for bar, val in zip(bars, y_vals):
            h = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.1, h * 0.02), f"{val:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax6.set_title(f"6. Coefficient of Variation (CV %) Comparison{tag}\nCER_cv = (CER_std / |CER_avg|) * 100%", fontsize=11, fontweight="bold")
        ax6.set_ylabel("CV (%)", fontsize=9, fontweight="bold")

    # 7. Relative Margin Error (RME) (Row 3, Col 0)
    ax7 = axes[3, 0]
    _apply_style(ax7)
    if rme_summary:
        labels = [r["experiment"] for r in rme_summary]
        rme_vals = [r["CER_rme_percent"] for r in rme_summary]
        colors = [EXP_COLORS[i % len(EXP_COLORS)] for i in range(len(labels))]
        bars = ax7.bar(labels, rme_vals, color=colors, width=0.45, edgecolor="#2c3e50", linewidth=0.8)
        for bar, val in zip(bars, rme_vals):
            h = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.1, h * 0.02), f"{val:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax7.set_title(f"7. Relative Margin Error (RME %) Comparison{tag}\nCER_rme = (CER_ci95 / |CER_avg|) * 100%", fontsize=11, fontweight="bold")
        ax7.set_ylabel("RME (%)", fontsize=9, fontweight="bold")

    # 8. Normal Distribution Overlaid (Row 3, Col 1)
    ax8 = axes[3, 1]
    _apply_style(ax8)
    if gauss_params:
        min_x = min(g["mu_cer_avg"] - 3.5 * max(0.5, g["sigma_cer_std"]) for g in gauss_params)
        max_x = max(g["mu_cer_avg"] + 3.5 * max(0.5, g["sigma_cer_std"]) for g in gauss_params)
        x_vals = np.linspace(min_x, max_x, 300)

        for i, g in enumerate(gauss_params):
            mu = g["mu_cer_avg"]
            sigma = max(1e-4, g["sigma_cer_std"])
            pdf_vals = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (((x_vals - mu) / sigma) ** 2))
            color = EXP_COLORS[i % len(EXP_COLORS)]
            lbl = f"{g['experiment']} (μ={mu:.1f}, σ={sigma:.2f})"
            ax8.plot(x_vals, pdf_vals, label=lbl, color=color, linewidth=2.0)
            ax8.axvline(x=mu, color=color, linestyle=":", linewidth=1.0, alpha=0.7)

        ax8.set_title(f"8. Gaussian Normal Distribution Overlaid Comparison{tag}", fontsize=11, fontweight="bold")
        ax8.set_xlabel("Cumulative Episodic Reward (CER)", fontsize=9, fontweight="bold")
        ax8.set_ylabel("Probability Density f(x)", fontsize=9, fontweight="bold")
        ax8.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    title_suffix = f" — {phase_title}" if phase_title else ""
    fig.suptitle(f"Cross-Experiment Comparative Dashboard — Group: {group_name}{title_suffix}", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _find_metric_csv(group_path: str, exp: str, phase_slug: str, metric_name: str) -> Optional[str]:
    """Find the path of a metric CSV, supporting both phased and legacy layouts.

    Checks:
    1. group_path/exp/phase_slug/metric_name/metric_name.csv
    2. group_path/exp/metric_name/metric_name.csv
    """
    p1 = os.path.join(group_path, exp, phase_slug, metric_name, f"{metric_name}.csv")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(group_path, exp, metric_name, f"{metric_name}.csv")
    if os.path.exists(p2):
        return p2
    return None


def generate_phase_comparison(
    group_name: str,
    group_path: str,
    exp_names: List[str],
    phase_slug: str,
    phase_title: str,
    target_dir: str,
    dpi: int = 300,
) -> Dict[str, Any]:
    """Generate comparative CSVs and PNGs for all 8 metrics for a single phase.

    Args:
        group_name (str): Experimental group name.
        group_path (str): Path to group reports root.
        exp_names (List[str]): List of sibling experiment names.
        phase_slug (str): Sub-folder slug ("initial_phase" or "stable_phase").
        phase_title (str): Display title ("Initial Phase (80%)" or "Stable Phase (20%)").
        target_dir (str): Destination folder (e.g. comparison/initial_phase).
        dpi (int, optional): Image resolution. Defaults to 300.

    Returns:
        Dict[str, Any]: Extracted data dictionaries for further aggregation.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"\nComparing {phase_title} across experiments -> {target_dir}")

    # 1. METRIC 1: Steps per Episode Comparison
    m1_dir = os.path.join(target_dir, "steps_per_episode")
    os.makedirs(m1_dir, exist_ok=True)
    m1_csv = os.path.join(m1_dir, "steps_per_episode_comparison.csv")
    m1_png = os.path.join(m1_dir, "steps_per_episode_comparison.png")

    exp_steps_data: Dict[str, Dict[int, float]] = {}
    for exp in exp_names:
        csv_p = _find_metric_csv(group_path, exp, phase_slug, "steps_per_episode")
        if csv_p:
            exp_steps_data[exp] = {}
            with open(csv_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep = int(row["episode"])
                    exp_steps_data[exp][ep] = float(row.get("mean_steps", 0))

    if exp_steps_data:
        all_eps_m1 = sorted(list(set(e for d in exp_steps_data.values() for e in d.keys())))
        with open(m1_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode"] + [f"{exp}_mean_steps" for exp in exp_steps_data.keys()])
            writer.writeheader()
            for ep in all_eps_m1:
                row_dict: Dict[str, Any] = {"episode": ep}
                for exp, data in exp_steps_data.items():
                    row_dict[f"{exp}_mean_steps"] = round(data.get(ep, 0.0), 2)
                writer.writerow(row_dict)

        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
        _apply_style(ax)
        for i, (exp, data) in enumerate(exp_steps_data.items()):
            eps_x = sorted(data.keys())
            vals_y = [data[e] for e in eps_x]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax.plot(eps_x, vals_y, label=exp, color=color, linewidth=2.0)
        ax.set_title(f"Steps per Episode Comparison ({group_name} — {phase_title})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Steps to Goal", fontsize=10, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        plt.savefig(m1_png, dpi=dpi)
        plt.close(fig)
        print(f"  [1/8] Steps Comparison: {m1_csv} | {m1_png}")

    # 2. METRIC 2: Current Cumulative Reward (CCR) Comparison
    m2_dir = os.path.join(target_dir, "current_cumulative_reward")
    os.makedirs(m2_dir, exist_ok=True)
    m2_csv = os.path.join(m2_dir, "current_cumulative_reward_comparison.csv")
    m2_png = os.path.join(m2_dir, "current_cumulative_reward_comparison.png")

    exp_ccr_data: Dict[str, Dict[int, float]] = {}
    for exp in exp_names:
        csv_p = _find_metric_csv(group_path, exp, phase_slug, "current_cumulative_reward")
        if csv_p:
            exp_ccr_data[exp] = {}
            with open(csv_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep = int(row["episode"])
                    exp_ccr_data[exp][ep] = float(row.get("mean_ccr", 0))

    if exp_ccr_data:
        all_eps_m2 = sorted(list(set(e for d in exp_ccr_data.values() for e in d.keys())))
        with open(m2_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode"] + [f"{exp}_mean_ccr" for exp in exp_ccr_data.keys()])
            writer.writeheader()
            for ep in all_eps_m2:
                row_dict = {"episode": ep}
                for exp, data in exp_ccr_data.items():
                    row_dict[f"{exp}_mean_ccr"] = round(data.get(ep, 0.0), 2)
                writer.writerow(row_dict)

        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
        _apply_style(ax)
        for i, (exp, data) in enumerate(exp_ccr_data.items()):
            eps_x = sorted(data.keys())
            vals_y = [data[e] for e in eps_x]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax.plot(eps_x, vals_y, label=exp, color=color, linewidth=2.0)
        ax.set_title(f"Current Cumulative Reward (CCR) Comparison ({group_name} — {phase_title})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Cumulative Reward", fontsize=10, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        plt.savefig(m2_png, dpi=dpi)
        plt.close(fig)
        print(f"  [2/8] CCR Comparison: {m2_csv} | {m2_png}")

    # 3. METRIC 3: Current Episodic Reward (CER) Comparison
    m3_dir = os.path.join(target_dir, "current_episodic_reward")
    os.makedirs(m3_dir, exist_ok=True)
    m3_csv = os.path.join(m3_dir, "current_episodic_reward_comparison.csv")
    m3_png = os.path.join(m3_dir, "current_episodic_reward_comparison.png")

    exp_cer_data: Dict[str, Dict[int, float]] = {}
    for exp in exp_names:
        csv_p = _find_metric_csv(group_path, exp, phase_slug, "current_episodic_reward")
        if csv_p:
            exp_cer_data[exp] = {}
            with open(csv_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep = int(row["episode"])
                    exp_cer_data[exp][ep] = float(row.get("mean_cer", 0))

    if exp_cer_data:
        all_eps_m3 = sorted(list(set(e for d in exp_cer_data.values() for e in d.keys())))
        with open(m3_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode"] + [f"{exp}_mean_cer" for exp in exp_cer_data.keys()])
            writer.writeheader()
            for ep in all_eps_m3:
                row_dict = {"episode": ep}
                for exp, data in exp_cer_data.items():
                    row_dict[f"{exp}_mean_cer"] = round(data.get(ep, 0.0), 2)
                writer.writerow(row_dict)

        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
        _apply_style(ax)
        for i, (exp, data) in enumerate(exp_cer_data.items()):
            eps_x = sorted(data.keys())
            vals_y = [data[e] for e in eps_x]
            w = max(2, min(5, len(vals_y) // 8))
            smoothed = [np.mean(vals_y[max(0, k - w + 1):k + 1]) for k in range(len(vals_y))]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax.plot(eps_x, smoothed, label=f"{exp} (trend)", color=color, linewidth=2.0)
            ax.plot(eps_x, vals_y, color=color, alpha=0.25, linewidth=0.8)
        ax.set_title(f"Current Episodic Reward (CER) Comparison ({group_name} — {phase_title})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Episodic Reward", fontsize=10, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        plt.savefig(m3_png, dpi=dpi)
        plt.close(fig)
        print(f"  [3/8] CER Comparison: {m3_csv} | {m3_png}")

    # 4. METRIC 4: Convergence Speed Comparison
    m4_dir = os.path.join(target_dir, "convergence_speed")
    os.makedirs(m4_dir, exist_ok=True)
    m4_csv = os.path.join(m4_dir, "convergence_speed_comparison.csv")
    m4_png = os.path.join(m4_dir, "convergence_speed_comparison.png")

    conv_summary: List[Dict[str, Any]] = []
    for exp in exp_names:
        csv_p = _find_metric_csv(group_path, exp, phase_slug, "convergence_speed")
        if csv_p:
            with open(csv_p, "r", encoding="utf-8") as f:
                lines = f.readlines()
            in_summary = False
            conv_eps = []
            final_steps = []
            for line in lines:
                line = line.strip()
                if line.startswith("# CONVERGENCE SPEED SUMMARY"):
                    in_summary = True
                    continue
                if line.startswith("#") or not line:
                    in_summary = False
                    continue
                if in_summary:
                    parts = line.split(",")
                    if parts[0] == "seed":
                        continue
                    if len(parts) >= 7:
                        c_ep_str = parts[2].strip()
                        c_ep = int(c_ep_str) if c_ep_str not in ("N/A", "None", "") else int(parts[3])
                        conv_eps.append(c_ep)
                        final_steps.append(float(parts[6]))

            avg_c = float(np.mean(conv_eps)) if conv_eps else 0.0
            avg_fs = float(np.mean(final_steps)) if final_steps else 0.0
            conv_summary.append({
                "experiment": exp,
                "avg_episodes_to_converge": round(avg_c, 1),
                "avg_final_steps": round(avg_fs, 1),
            })

    if conv_summary:
        with open(m4_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", "avg_episodes_to_converge", "avg_final_steps"])
            writer.writeheader()
            writer.writerows(conv_summary)

        fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=dpi)
        _apply_style(ax)
        exp_labels = [c["experiment"] for c in conv_summary]
        y_vals = [c["avg_episodes_to_converge"] for c in conv_summary]
        colors = [EXP_COLORS[i % len(EXP_COLORS)] for i in range(len(exp_labels))]
        bars = ax.bar(exp_labels, y_vals, color=colors, width=0.5, edgecolor="#2c3e50", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.5, h * 0.02), f"{h:.1f} ep", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"Convergence Speed Comparison ({group_name} — {phase_title})\n(Fewer episodes = Faster Convergence)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Episodes to Converge", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(m4_png, dpi=dpi)
        plt.close(fig)
        print(f"  [4/8] Convergence Comparison: {m4_csv} | {m4_png}")

    # 5. METRIC 5: Collision Rate Comparison
    m5_dir = os.path.join(target_dir, "collision_rate")
    os.makedirs(m5_dir, exist_ok=True)
    m5_csv = os.path.join(m5_dir, "collision_rate_comparison.csv")
    m5_png = os.path.join(m5_dir, "collision_rate_comparison.png")

    exp_coll_data: Dict[str, Dict[int, float]] = {}
    for exp in exp_names:
        csv_p = _find_metric_csv(group_path, exp, phase_slug, "collision_rate")
        if csv_p:
            exp_coll_data[exp] = {}
            with open(csv_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep = int(row["episode"])
                    exp_coll_data[exp][ep] = float(row.get("mean_collision_rate_pct", 0.0))

    if exp_coll_data:
        all_coll_eps = sorted(list(set(ep for d in exp_coll_data.values() for ep in d.keys())))
        with open(m5_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode"] + [f"{exp}_collision_rate_pct" for exp in exp_coll_data.keys()])
            writer.writeheader()
            for ep in all_coll_eps:
                row_dict = {"episode": ep}
                for exp, data in exp_coll_data.items():
                    row_dict[f"{exp}_collision_rate_pct"] = round(data.get(ep, 0.0), 2)
                writer.writerow(row_dict)

        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=dpi)
        _apply_style(ax)
        for i, (exp, data) in enumerate(exp_coll_data.items()):
            eps_x = sorted(data.keys())
            vals_y = [data[e] for e in eps_x]
            color = EXP_COLORS[i % len(EXP_COLORS)]
            ax.plot(eps_x, vals_y, marker="o", markersize=3.5, label=f"{exp} (Mean: {np.mean(vals_y):.1f}%)", color=color, linewidth=1.5)
        ax.set_title(f"Collision Rate Comparison ({group_name} — {phase_title})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
        ax.set_ylabel("Collision Rate (% of Steps)", fontsize=10, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        plt.savefig(m5_png, dpi=dpi)
        plt.close(fig)
        print(f"  [5/8] Collision Comparison: {m5_csv} | {m5_png}")

    # 6. METRIC 6: Coefficient of Variation (CV) Comparison
    m6_dir = os.path.join(target_dir, "coefficient_of_variation")
    os.makedirs(m6_dir, exist_ok=True)
    m6_csv = os.path.join(m6_dir, "coefficient_of_variation_comparison.csv")
    m6_png = os.path.join(m6_dir, "coefficient_of_variation_comparison.png")

    cv_summary: List[Dict[str, Any]] = []
    for exp in exp_names:
        csv_p = _find_metric_csv(group_path, exp, phase_slug, "coefficient_of_variation")
        if csv_p:
            with open(csv_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "Overall" in row.get("seed", ""):
                        cv_summary.append({
                            "experiment": exp,
                            "N_episodes": int(row.get("N_episodes", 0)),
                            "CER_avg": float(row.get("CER_avg", 0.0)),
                            "CER_std": float(row.get("CER_std", 0.0)),
                            "CER_cv_percent": float(row.get("CER_cv_percent", 0.0)),
                        })

    if cv_summary:
        with open(m6_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", "N_episodes", "CER_avg", "CER_std", "CER_cv_percent"])
            writer.writeheader()
            writer.writerows(cv_summary)

        fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=dpi)
        _apply_style(ax)
        labels = [c["experiment"] for c in cv_summary]
        y_vals = [c["CER_cv_percent"] for c in cv_summary]
        colors = [EXP_COLORS[i % len(EXP_COLORS)] for i in range(len(labels))]
        bars = ax.bar(labels, y_vals, color=colors, width=0.5, edgecolor="#2c3e50", linewidth=0.8)
        for bar, val in zip(bars, y_vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.1, h * 0.02), f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"Coefficient of Variation (CV) Comparison ({group_name} — {phase_title})\nCER_cv = (CER_std / |CER_avg|) * 100%", fontsize=11, fontweight="bold")
        ax.set_ylabel("CV (%)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(m6_png, dpi=dpi)
        plt.close(fig)
        print(f"  [6/8] CV Comparison: {m6_csv} | {m6_png}")

    # 7. METRIC 7: Relative Margin Error (RME) Comparison
    m7_dir = os.path.join(target_dir, "relative_margin_error")
    os.makedirs(m7_dir, exist_ok=True)
    m7_csv = os.path.join(m7_dir, "relative_margin_error_comparison.csv")
    m7_png = os.path.join(m7_dir, "relative_margin_error_comparison.png")

    rme_summary: List[Dict[str, Any]] = []
    for exp in exp_names:
        csv_p = _find_metric_csv(group_path, exp, phase_slug, "relative_margin_error")
        if csv_p:
            with open(csv_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "Overall" in row.get("seed", ""):
                        rme_summary.append({
                            "experiment": exp,
                            "N_episodes": int(row.get("N_episodes", 0)),
                            "CER_avg": float(row.get("CER_avg", 0.0)),
                            "CER_ci95": float(row.get("CER_ci95", 0.0)),
                            "ci_lower_95": float(row.get("ci_lower_95", 0.0)),
                            "ci_upper_95": float(row.get("ci_upper_95", 0.0)),
                            "CER_rme_percent": float(row.get("CER_rme_percent", 0.0)),
                        })

    if rme_summary:
        with open(m7_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "experiment", "N_episodes", "CER_avg", "CER_ci95",
                "ci_lower_95", "ci_upper_95", "CER_rme_percent"
            ])
            writer.writeheader()
            writer.writerows(rme_summary)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=dpi)
        _apply_style(ax1)
        _apply_style(ax2)

        labels = [r["experiment"] for r in rme_summary]
        rme_vals = [r["CER_rme_percent"] for r in rme_summary]
        avgs = [r["CER_avg"] for r in rme_summary]
        ci95s = [r["CER_ci95"] for r in rme_summary]
        colors = [EXP_COLORS[i % len(EXP_COLORS)] for i in range(len(labels))]

        bars = ax1.bar(labels, rme_vals, color=colors, width=0.5, edgecolor="#2c3e50", linewidth=0.8)
        for bar, val in zip(bars, rme_vals):
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.1, h * 0.02), f"{val:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax1.set_title("Relative Margin Error (RME %)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("RME (%) = (CER_ci95 / |CER_avg|) * 100%", fontsize=9, fontweight="bold")

        x_pts = list(range(len(labels)))
        ax2.errorbar(x_pts, avgs, yerr=ci95s, fmt="s", color="#2980b9", ecolor="#c0392b", elinewidth=2.2, capsize=6, capthick=1.8, markersize=6)
        ax2.set_xticks(x_pts)
        ax2.set_xticklabels(labels)
        ax2.set_title("Mean CER with 95% Confidence Interval", fontsize=11, fontweight="bold")
        ax2.set_ylabel("CER (95% CI)", fontsize=9, fontweight="bold")

        fig.suptitle(f"RME & 95% CI Comparison ({group_name} — {phase_title})", fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(m7_png, dpi=dpi)
        plt.close(fig)
        print(f"  [7/8] RME Comparison: {m7_csv} | {m7_png}")

    # 8. METRIC 8: Normal Distribution Comparison
    m8_dir = os.path.join(target_dir, "normal_distribution")
    os.makedirs(m8_dir, exist_ok=True)
    m8_csv = os.path.join(m8_dir, "normal_distribution_comparison.csv")
    m8_png = os.path.join(m8_dir, "normal_distribution_comparison.png")

    gauss_params: List[Dict[str, Any]] = []
    for exp in exp_names:
        csv_p = _find_metric_csv(group_path, exp, phase_slug, "normal_distribution")
        if csv_p:
            with open(csv_p, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("Overall"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 9:
                        gauss_params.append({
                            "experiment": exp,
                            "N_episodes": int(parts[1]),
                            "mu_cer_avg": float(parts[2]),
                            "sigma_cer_std": float(parts[3]),
                            "sigma2_variance": float(parts[4]),
                            "cer_ci95": float(parts[5]),
                            "ci_lower_95": float(parts[6]),
                            "ci_upper_95": float(parts[7]),
                            "cer_rme_percent": float(parts[8]),
                        })

    if gauss_params:
        with open(m8_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "experiment", "N_episodes", "mu_cer_avg", "sigma_cer_std",
                "sigma2_variance", "cer_ci95", "ci_lower_95", "ci_upper_95", "cer_rme_percent"
            ])
            writer.writeheader()
            writer.writerows(gauss_params)

        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
        _apply_style(ax)

        min_x = min(g["mu_cer_avg"] - 3.5 * max(0.5, g["sigma_cer_std"]) for g in gauss_params)
        max_x = max(g["mu_cer_avg"] + 3.5 * max(0.5, g["sigma_cer_std"]) for g in gauss_params)
        x_vals = np.linspace(min_x, max_x, 300)

        for i, g in enumerate(gauss_params):
            mu = g["mu_cer_avg"]
            sigma = max(1e-4, g["sigma_cer_std"])
            pdf_vals = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (((x_vals - mu) / sigma) ** 2))
            color = EXP_COLORS[i % len(EXP_COLORS)]
            lbl = f"{g['experiment']} (μ={mu:.1f}, σ={sigma:.2f})"
            ax.plot(x_vals, pdf_vals, label=lbl, color=color, linewidth=2.0)
            ax.axvline(x=mu, color=color, linestyle=":", linewidth=1.0, alpha=0.7)

        ax.set_title(f"Gaussian Normal Distribution Overlaid Comparison ({group_name} — {phase_title})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Cumulative Episodic Reward (CER)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Probability Density f(x)", fontsize=10, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        plt.savefig(m8_png, dpi=dpi)
        plt.close(fig)
        print(f"  [8/8] Normal Distribution Comparison: {m8_csv} | {m8_png}")

    # 9. COMBINED 8-PANEL DASHBOARD FOR THIS PHASE
    phase_subname = phase_slug.replace("_phase", "")
    phase_dashboard_png = os.path.join(target_dir, f"all_metrics_comparison_{phase_subname}.png")
    plot_combined_comparison_dashboard(
        group_name=group_name,
        exp_steps_data=exp_steps_data,
        exp_ccr_data=exp_ccr_data,
        exp_cer_data=exp_cer_data,
        conv_summary=conv_summary,
        exp_coll_data=exp_coll_data,
        cv_summary=cv_summary,
        rme_summary=rme_summary,
        gauss_params=gauss_params,
        output_path=phase_dashboard_png,
        phase_title=phase_title,
        dpi=dpi,
    )
    print(f"  [★] Phase Comparison Dashboard: {phase_dashboard_png}")

    return {
        "exp_steps_data": exp_steps_data,
        "exp_ccr_data": exp_ccr_data,
        "exp_cer_data": exp_cer_data,
        "conv_summary": conv_summary,
        "exp_coll_data": exp_coll_data,
        "cv_summary": cv_summary,
        "rme_summary": rme_summary,
        "gauss_params": gauss_params,
        "dashboard_path": phase_dashboard_png,
    }


def generate_group_comparison(
    reports_dir: str,
    group_name: str,
    dpi: int = 300,
) -> Optional[Dict[str, Any]]:
    """Generate comparative CSVs and PNG figures across all experiments in a group.

    Creates full comparisons for both Initial Phase and Stable Phase in::

        reports/<group_name>/comparison/
        ├── initial_phase/
        │   ├── steps_per_episode/
        │   ├── current_cumulative_reward/
        │   ├── current_episodic_reward/
        │   ├── convergence_speed/
        │   ├── collision_rate/
        │   ├── coefficient_of_variation/
        │   ├── relative_margin_error/
        │   ├── normal_distribution/
        │   └── all_metrics_comparison_initial.png
        ├── stable_phase/
        │   ├── steps_per_episode/
        │   ├── current_cumulative_reward/
        │   ├── current_episodic_reward/
        │   ├── convergence_speed/
        │   ├── collision_rate/
        │   ├── coefficient_of_variation/
        │   ├── relative_margin_error/
        │   ├── normal_distribution/
        │   └── all_metrics_comparison_stable.png
        └── all_metrics_comparison.png (Executive Master Dashboard)

    Args:
        reports_dir (str): Base reports directory.
        group_name (str): Name of the experiment group.
        dpi (int, optional): Image resolution DPI. Defaults to 300.

    Returns:
        Optional[Dict[str, Any]]: Summary of generated comparison reports, or None if skipped.
    """
    if not group_name or str(group_name).strip().lower() in ("", "none"):
        return None

    group_path = os.path.join(reports_dir, group_name.strip())
    if not os.path.exists(group_path):
        return None

    # Discover sibling experiment directories
    exp_names = sorted([
        d for d in os.listdir(group_path)
        if os.path.isdir(os.path.join(group_path, d)) and d != "comparison"
    ])

    if len(exp_names) < 2:
        print(f"\nGroup '{group_name}' currently has {len(exp_names)} experiment(s). "
              f"Comparison requires at least 2 experiments to generate comparative figures.")
        return None

    comp_dir = os.path.join(group_path, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    print(f"\n======================================================================")
    print(f"GENERATING CROSS-EXPERIMENT COMPARISONS FOR GROUP: {group_name}")
    print(f"Experiments to Compare ({len(exp_names)}): {', '.join(exp_names)}")
    print(f"Comparison Destination: {comp_dir}")
    print(f"======================================================================")

    # 1. Compare INITIAL PHASE
    initial_target_dir = os.path.join(comp_dir, "initial_phase")
    initial_res = generate_phase_comparison(
        group_name=group_name,
        group_path=group_path,
        exp_names=exp_names,
        phase_slug="initial_phase",
        phase_title="Initial Phase",
        target_dir=initial_target_dir,
        dpi=dpi,
    )

    # 2. Compare STABLE PHASE
    stable_target_dir = os.path.join(comp_dir, "stable_phase")
    stable_res = generate_phase_comparison(
        group_name=group_name,
        group_path=group_path,
        exp_names=exp_names,
        phase_slug="stable_phase",
        phase_title="Stable Phase",
        target_dir=stable_target_dir,
        dpi=dpi,
    )

    # 3. EXECUTIVE MASTER DASHBOARD AT ROOT (reports/<group_name>/comparison/all_metrics_comparison.png)
    # Merges Initial Phase progression (learning & speed) with Stable Phase asymptotic stability (safety & stats)
    dashboard_png = os.path.join(comp_dir, "all_metrics_comparison.png")
    dashboard_alt_png = os.path.join(comp_dir, "comparison_dashboard.png")

    # Use initial phase data for early learning dynamics (steps, ccr, cer, conv)
    # and stable phase data for asymptotic stability (collision, cv, rme, normal)
    # If stable phase has its own, or fallback if needed:
    master_steps = initial_res.get("exp_steps_data") or stable_res.get("exp_steps_data") or {}
    master_ccr = initial_res.get("exp_ccr_data") or stable_res.get("exp_ccr_data") or {}
    master_cer = initial_res.get("exp_cer_data") or stable_res.get("exp_cer_data") or {}
    master_conv = initial_res.get("conv_summary") or stable_res.get("conv_summary") or []

    master_coll = stable_res.get("exp_coll_data") or initial_res.get("exp_coll_data") or {}
    master_cv = stable_res.get("cv_summary") or initial_res.get("cv_summary") or []
    master_rme = stable_res.get("rme_summary") or initial_res.get("rme_summary") or []
    master_gauss = stable_res.get("gauss_params") or initial_res.get("gauss_params") or []

    plot_combined_comparison_dashboard(
        group_name=group_name,
        exp_steps_data=master_steps,
        exp_ccr_data=master_ccr,
        exp_cer_data=master_cer,
        conv_summary=master_conv,
        exp_coll_data=master_coll,
        cv_summary=master_cv,
        rme_summary=master_rme,
        gauss_params=master_gauss,
        output_path=dashboard_png,
        phase_title="Full Lifecycle Overview",
        dpi=dpi,
    )

    if os.path.exists(dashboard_png):
        shutil.copy2(dashboard_png, dashboard_alt_png)

    print(f"\n  [★] MASTER COMPARISON DASHBOARD GENERATED:")
    print(f"      - {dashboard_png}")
    print(f"      - {dashboard_alt_png}")
    print(f"Group comparison successfully completed in: {comp_dir}\n")

    return {
        "group_name": group_name,
        "comp_dir": comp_dir,
        "experiments": exp_names,
        "initial_comparison": initial_res,
        "stable_comparison": stable_res,
        "dashboard_path": dashboard_png,
    }


def merge_and_compare_groups(
    source_groups: List[str],
    target_group: str,
    reports_dir: str = "reports",
    experiment_aliases: Optional[Dict[str, str]] = None,
    filter_experiments: Optional[List[str]] = None,
    copy_files: bool = False,
    dpi: int = 300,
) -> Optional[Dict[str, Any]]:
    """Merge multiple experiment groups into a single combined comparison group.

    Creates links (or copies) of individual experiment sub-folders from multiple
    source groups into reports/<target_group>/, and triggers the full cross-experiment
    comparison suite across all 8 metrics for both Initial Phase and Stable Phase.

    Args:
        source_groups (List[str]): List of group folder names in reports/.
        target_group (str): Name for the new merged comparison group folder.
        reports_dir (str, optional): Base reports directory. Defaults to "reports".
        experiment_aliases (Optional[Dict[str, str]], optional): Custom display names
            mapping from original experiment folder names to friendly names.
        filter_experiments (Optional[List[str]], optional): Specific experiment names
            to include (omits any other experiments found in the groups).
        copy_files (bool, optional): If True, physically copies directories instead
            of creating symlinks. Defaults to False (symlinks).
        dpi (int, optional): Image resolution DPI. Defaults to 300.

    Returns:
        Optional[Dict[str, Any]]: Comparison summary dictionary, or None if failed.
    """
    if not source_groups or not target_group:
        print("Error: source_groups and target_group must be specified.")
        return None

    target_path = os.path.join(reports_dir, target_group.strip())
    os.makedirs(target_path, exist_ok=True)

    # Clean up stale experiment links in target_path from previous merges
    for existing_d in os.listdir(target_path):
        if existing_d == "comparison":
            continue
        p = os.path.join(target_path, existing_d)
        if os.path.islink(p):
            os.unlink(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

    linked_experiments = []

    for grp in source_groups:
        grp_clean = grp.strip()
        grp_path = os.path.join(reports_dir, grp_clean)
        if not os.path.exists(grp_path):
            print(f"Warning: Source group path does not exist: {grp_path}")
            continue

        # Find experiments (exclude 'comparison')
        exp_dirs = sorted([
            d for d in os.listdir(grp_path)
            if os.path.isdir(os.path.join(grp_path, d)) and d != "comparison"
        ])

        for exp in exp_dirs:
            if filter_experiments and exp not in filter_experiments:
                continue

            src_exp_path = os.path.join(grp_path, exp)
            dest_name = experiment_aliases.get(exp, exp) if experiment_aliases else exp
            dest_exp_path = os.path.join(target_path, dest_name)

            if os.path.lexists(dest_exp_path):
                if os.path.islink(dest_exp_path):
                    os.unlink(dest_exp_path)
                elif copy_files and os.path.isdir(dest_exp_path):
                    shutil.rmtree(dest_exp_path)

            if not os.path.exists(dest_exp_path):
                if copy_files:
                    shutil.copytree(src_exp_path, dest_exp_path)
                    print(f"Copied experiment: {src_exp_path} -> {dest_exp_path}")
                else:
                    rel_src = os.path.relpath(src_exp_path, target_path)
                    os.symlink(rel_src, dest_exp_path)
                    print(f"Linked experiment: {dest_name} -> {rel_src}")

            linked_experiments.append(dest_name)

    if len(linked_experiments) < 2:
        print(f"Error: Found only {len(linked_experiments)} experiment(s) to merge. Need at least 2.")
        return None

    print(f"\nMerged {len(linked_experiments)} experiments into '{target_group}': {', '.join(linked_experiments)}")
    print("Now generating cross-experiment comparison suite...")

    return generate_group_comparison(reports_dir=reports_dir, group_name=target_group.strip(), dpi=dpi)

