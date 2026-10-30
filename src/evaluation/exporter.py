"""CSV and visual report exporter for agent learning evaluation metrics.

Coordinates the computation of all 8 metrics across both Initial Phase (80%)
and Stable Phase (20%), exporting each metric into its own designated sub-folder::

    reports/[<group_name>/]<experiment_name>/
    ├── initial_phase/
    │   ├── steps_per_episode/
    │   ├── current_cumulative_reward/
    │   ├── current_episodic_reward/
    │   ├── convergence_speed/
    │   ├── collision_rate/
    │   ├── coefficient_of_variation/
    │   ├── relative_margin_error/
    │   ├── normal_distribution/
    │   └── all_metrics_initial.png
    └── stable_phase/
        ├── steps_per_episode/
        ├── current_cumulative_reward/
        ├── current_episodic_reward/
        ├── convergence_speed/
        ├── collision_rate/
        ├── coefficient_of_variation/
        ├── relative_margin_error/
        ├── normal_distribution/
        └── all_metrics_stable.png

Also triggers comparative group evaluations across sibling experiments in:
reports/<group_name>/comparison/
"""

import csv
import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.evaluation.comparison import generate_group_comparison
from src.evaluation.metrics import (
    compute_cer_phase_statistics,
    compute_convergence_metrics,
    compute_gaussian_pdf,
    split_learning_phases,
)
from src.evaluation.visualizer import (
    plot_collision_rate,
    plot_coefficient_of_variation,
    plot_combined_phase_dashboard,
    plot_convergence_speed,
    plot_current_cumulative_reward,
    plot_current_episodic_reward,
    plot_normal_distribution,
    plot_relative_margin_error,
    plot_steps_per_episode,
)


def export_phase_metrics(
    phase_dir: str,
    phase_name: str,
    phase_slug: str,
    phase_episodes: Dict[int, List[Dict[str, Any]]],
    optimal_steps_dict: Dict[int, int],
    dpi: int = 300,
) -> Dict[str, Any]:
    """Export all 8 metrics (CSV and publication PNG) for a single learning phase.

    Each metric is stored in its own sub-folder inside phase_dir:
    - steps_per_episode/
    - current_cumulative_reward/
    - current_episodic_reward/
    - convergence_speed/
    - collision_rate/
    - coefficient_of_variation/
    - relative_margin_error/
    - normal_distribution/
    Plus an aggregated 8-panel dashboard: all_metrics_<phase_slug>.png.

    Args:
        phase_dir (str): Destination folder for this phase.
        phase_name (str): Human-readable title (e.g. "Initial Phase (80%)").
        phase_slug (str): File slug (e.g. "initial" or "stable").
        phase_episodes (Dict[int, List[Dict[str, Any]]]): Sliced episode logs per seed.
        optimal_steps_dict (Dict[int, int]): Optimal geodesic distance per seed.
        dpi (int, optional): Image resolution. Defaults to 300.

    Returns:
        Dict[str, Any]: Phase statistics, convergence metrics, and dashboard file path.
    """
    os.makedirs(phase_dir, exist_ok=True)
    seeds = list(phase_episodes.keys())
    if not seeds or not phase_episodes[seeds[0]]:
        return {}

    first_seed = seeds[0]
    eps_first = phase_episodes[first_seed]
    n_episodes = len(eps_first)
    episode_numbers = [e["episode"] for e in eps_first]

    print(f"\n--- Exporting {phase_name} ({n_episodes} Episodes: Ep {episode_numbers[0]}..{episode_numbers[-1]}) ---")

    # --------------------------------------------------------------------------
    # METRIC 1: Steps per Episode
    # --------------------------------------------------------------------------
    m1_dir = os.path.join(phase_dir, "steps_per_episode")
    os.makedirs(m1_dir, exist_ok=True)
    steps_csv = os.path.join(m1_dir, "steps_per_episode.csv")
    steps_png = os.path.join(m1_dir, "steps_per_episode.png")

    with open(steps_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["episode"] + [f"seed_{s}_steps" for s in seeds] + ["mean_steps", "std_steps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_episodes):
            row: Dict[str, Any] = {"episode": episode_numbers[idx]}
            ep_steps = []
            for s in seeds:
                eps = phase_episodes[s]
                val = eps[idx]["steps"] if idx < len(eps) else 0
                row[f"seed_{s}_steps"] = val
                ep_steps.append(val)
            row["mean_steps"] = round(float(np.mean(ep_steps)), 2)
            row["std_steps"] = round(float(np.std(ep_steps)), 2)
            writer.writerow(row)

    plot_steps_per_episode(phase_episodes, steps_png, phase_name=phase_name, dpi=dpi)
    print(f"  [1/8] Steps per Episode: {steps_csv} | {steps_png}")

    # --------------------------------------------------------------------------
    # METRIC 2: Current Cumulative Reward (CCR)
    # --------------------------------------------------------------------------
    m2_dir = os.path.join(phase_dir, "current_cumulative_reward")
    os.makedirs(m2_dir, exist_ok=True)
    ccr_csv = os.path.join(m2_dir, "current_cumulative_reward.csv")
    ccr_png = os.path.join(m2_dir, "current_cumulative_reward.png")

    seed_ccr: Dict[int, List[float]] = {}
    for s in seeds:
        rewards = [e["reward"] for e in phase_episodes[s]]
        seed_ccr[s] = list(np.cumsum(rewards))

    with open(ccr_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["episode"] + [f"seed_{s}_ccr" for s in seeds] + ["mean_ccr", "std_ccr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_episodes):
            row = {"episode": episode_numbers[idx]}
            ep_ccrs = []
            for s in seeds:
                ccr_list = seed_ccr[s]
                val = round(ccr_list[idx], 2) if idx < len(ccr_list) else 0.0
                row[f"seed_{s}_ccr"] = val
                ep_ccrs.append(val)
            row["mean_ccr"] = round(float(np.mean(ep_ccrs)), 2)
            row["std_ccr"] = round(float(np.std(ep_ccrs)), 2)
            writer.writerow(row)

    plot_current_cumulative_reward(phase_episodes, ccr_png, phase_name=phase_name, dpi=dpi)
    print(f"  [2/8] Current Cumulative Reward: {ccr_csv} | {ccr_png}")

    # --------------------------------------------------------------------------
    # METRIC 3: Current Episodic Reward (CER)
    # --------------------------------------------------------------------------
    m3_dir = os.path.join(phase_dir, "current_episodic_reward")
    os.makedirs(m3_dir, exist_ok=True)
    cer_csv = os.path.join(m3_dir, "current_episodic_reward.csv")
    cer_png = os.path.join(m3_dir, "current_episodic_reward.png")

    with open(cer_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["episode"] + [f"seed_{s}_cer" for s in seeds] + ["mean_cer", "std_cer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_episodes):
            row = {"episode": episode_numbers[idx]}
            ep_cers = []
            for s in seeds:
                eps = phase_episodes[s]
                val = round(float(eps[idx]["reward"]), 2) if idx < len(eps) else 0.0
                row[f"seed_{s}_cer"] = val
                ep_cers.append(val)
            row["mean_cer"] = round(float(np.mean(ep_cers)), 2)
            row["std_cer"] = round(float(np.std(ep_cers)), 2)
            writer.writerow(row)

    plot_current_episodic_reward(phase_episodes, cer_png, phase_name=phase_name, dpi=dpi)
    print(f"  [3/8] Current Episodic Reward: {cer_csv} | {cer_png}")

    # --------------------------------------------------------------------------
    # METRIC 4: Convergence Speed
    # --------------------------------------------------------------------------
    m4_dir = os.path.join(phase_dir, "convergence_speed")
    os.makedirs(m4_dir, exist_ok=True)
    conv_csv = os.path.join(m4_dir, "convergence_speed.csv")
    conv_png = os.path.join(m4_dir, "convergence_speed.png")

    conv_metrics_per_seed: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        steps_list = [e["steps"] for e in phase_episodes[s]]
        rewards_list = [e["reward"] for e in phase_episodes[s]]
        opt_s = optimal_steps_dict.get(s, 0)
        conv_metrics_per_seed[s] = compute_convergence_metrics(
            steps_list=steps_list,
            rewards_list=rewards_list,
            optimal_steps=opt_s,
            episode_numbers=episode_numbers,
        )

    with open(conv_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"# CONVERGENCE SPEED SUMMARY ({phase_name})"])
        writer.writerow([
            "seed", "optimal_steps", "convergence_episode", "episodes_to_converge",
            "converged", "target_threshold", "final_moving_avg_steps"
        ])
        for s in seeds:
            cm = conv_metrics_per_seed[s]
            writer.writerow([
                s,
                cm["optimal_steps"],
                cm["convergence_episode"] if cm["convergence_episode"] is not None else "N/A",
                cm["episodes_to_converge"],
                cm["converged"],
                round(cm["target_threshold"], 2),
                round(cm["final_moving_avg_steps"], 2),
            ])

        writer.writerow([])
        writer.writerow(["# EPISODE MOVING AVERAGE PROGRESSION"])
        ma_fieldnames = ["episode"] + [f"seed_{s}_ma_steps" for s in seeds]
        writer.writerow(ma_fieldnames)
        for idx in range(n_episodes):
            row_ma = [episode_numbers[idx]]
            for s in seeds:
                ma_list = conv_metrics_per_seed[s]["moving_avg_steps"]
                row_ma.append(round(ma_list[idx], 2) if idx < len(ma_list) else 0.0)
            writer.writerow(row_ma)

    plot_convergence_speed(phase_episodes, conv_metrics_per_seed, conv_png, phase_name=phase_name, dpi=dpi)
    print(f"  [4/8] Convergence Speed: {conv_csv} | {conv_png}")

    # --------------------------------------------------------------------------
    # METRIC 5: Collision Rate
    # --------------------------------------------------------------------------
    m5_dir = os.path.join(phase_dir, "collision_rate")
    os.makedirs(m5_dir, exist_ok=True)
    coll_csv = os.path.join(m5_dir, "collision_rate.csv")
    coll_png = os.path.join(m5_dir, "collision_rate.png")

    with open(coll_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["episode"]
        for s in seeds:
            fieldnames.extend([f"seed_{s}_collisions", f"seed_{s}_steps", f"seed_{s}_collision_rate_pct"])
        fieldnames.append("mean_collision_rate_pct")

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(n_episodes):
            row = {"episode": episode_numbers[idx]}
            ep_rates = []
            for s in seeds:
                eps = phase_episodes[s]
                colls = eps[idx].get("collisions", 0) if idx < len(eps) else 0
                st = eps[idx].get("steps", 1) if idx < len(eps) else 1
                rate = (colls / max(1, st)) * 100.0
                row[f"seed_{s}_collisions"] = colls
                row[f"seed_{s}_steps"] = st
                row[f"seed_{s}_collision_rate_pct"] = round(rate, 2)
                ep_rates.append(rate)
            row["mean_collision_rate_pct"] = round(float(np.mean(ep_rates)), 2)
            writer.writerow(row)

    plot_collision_rate(phase_episodes, coll_png, phase_name=phase_name, dpi=dpi)
    print(f"  [5/8] Collision Rate: {coll_csv} | {coll_png}")

    # --------------------------------------------------------------------------
    # CER PHASE STATISTICS (for Metrics 6, 7, and 8)
    # --------------------------------------------------------------------------
    stats_per_seed: Dict[int, Dict[str, Any]] = {}
    all_phase_cers: List[float] = []

    for s in seeds:
        sub_cers = [float(e["reward"]) for e in phase_episodes[s]]
        stats_per_seed[s] = compute_cer_phase_statistics(sub_cers)
        all_phase_cers.extend(sub_cers)

    combined_stats = compute_cer_phase_statistics(all_phase_cers)

    # --------------------------------------------------------------------------
    # METRIC 6: Coefficient of Variation (CV)
    # Formula: CER_cv = (CER_std / |CER_avg|) * 100%
    # --------------------------------------------------------------------------
    m6_dir = os.path.join(phase_dir, "coefficient_of_variation")
    os.makedirs(m6_dir, exist_ok=True)
    cv_csv = os.path.join(m6_dir, "coefficient_of_variation.csv")
    cv_png = os.path.join(m6_dir, "coefficient_of_variation.png")

    with open(cv_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["seed", "N_episodes", "CER_avg", "CER_std", "abs_CER_avg", "CER_cv_percent"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in seeds:
            st = stats_per_seed[s]
            writer.writerow({
                "seed": s,
                "N_episodes": st["n"],
                "CER_avg": round(st["cer_avg"], 4),
                "CER_std": round(st["cer_std"], 4),
                "abs_CER_avg": round(abs(st["cer_avg"]), 4),
                "CER_cv_percent": round(st["cer_cv"], 4),
            })
        writer.writerow({
            "seed": "Overall (Combined)",
            "N_episodes": combined_stats["n"],
            "CER_avg": round(combined_stats["cer_avg"], 4),
            "CER_std": round(combined_stats["cer_std"], 4),
            "abs_CER_avg": round(abs(combined_stats["cer_avg"]), 4),
            "CER_cv_percent": round(combined_stats["cer_cv"], 4),
        })

    plot_coefficient_of_variation(stats_per_seed, combined_stats, cv_png, phase_name=phase_name, dpi=dpi)
    print(f"  [6/8] Coefficient of Variation (CV): {cv_csv} | {cv_png}")

    # --------------------------------------------------------------------------
    # METRIC 7: Relative Margin Error (RME)
    # Formula: CER_ci95 = 1.96 * CER_std / sqrt(N), CER_rme = (CER_ci95 / |CER_avg|) * 100%
    # --------------------------------------------------------------------------
    m7_dir = os.path.join(phase_dir, "relative_margin_error")
    os.makedirs(m7_dir, exist_ok=True)
    rme_csv = os.path.join(m7_dir, "relative_margin_error.csv")
    rme_png = os.path.join(m7_dir, "relative_margin_error.png")

    with open(rme_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seed", "N_episodes", "CER_avg", "CER_std", "CER_ci95",
            "ci_lower_95", "ci_upper_95", "abs_CER_avg", "CER_rme_percent"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in seeds:
            st = stats_per_seed[s]
            writer.writerow({
                "seed": s,
                "N_episodes": st["n"],
                "CER_avg": round(st["cer_avg"], 4),
                "CER_std": round(st["cer_std"], 4),
                "CER_ci95": round(st["cer_ci95"], 4),
                "ci_lower_95": round(st["ci_lower"], 4),
                "ci_upper_95": round(st["ci_upper"], 4),
                "abs_CER_avg": round(abs(st["cer_avg"]), 4),
                "CER_rme_percent": round(st["cer_rme"], 4),
            })
        writer.writerow({
            "seed": "Overall (Combined)",
            "N_episodes": combined_stats["n"],
            "CER_avg": round(combined_stats["cer_avg"], 4),
            "CER_std": round(combined_stats["cer_std"], 4),
            "CER_ci95": round(combined_stats["cer_ci95"], 4),
            "ci_lower_95": round(combined_stats["ci_lower"], 4),
            "ci_upper_95": round(combined_stats["ci_upper"], 4),
            "abs_CER_avg": round(abs(combined_stats["cer_avg"]), 4),
            "CER_rme_percent": round(combined_stats["cer_rme"], 4),
        })

    plot_relative_margin_error(stats_per_seed, combined_stats, rme_png, phase_name=phase_name, dpi=dpi)
    print(f"  [7/8] Relative Margin Error (RME): {rme_csv} | {rme_png}")

    # --------------------------------------------------------------------------
    # METRIC 8: Normal Distribution
    # --------------------------------------------------------------------------
    m8_dir = os.path.join(phase_dir, "normal_distribution")
    os.makedirs(m8_dir, exist_ok=True)
    dist_csv = os.path.join(m8_dir, "normal_distribution.csv")
    dist_png = os.path.join(m8_dir, "normal_distribution.png")

    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"# GAUSSIAN NORMAL DISTRIBUTION PARAMETERS ({phase_name})"])
        writer.writerow([
            "seed", "N_episodes", "CER_avg_mu", "CER_std_sigma", "CER_variance_sigma2",
            "CER_ci95", "ci_lower_95", "ci_upper_95", "CER_rme_percent"
        ])
        for s in seeds:
            st = stats_per_seed[s]
            writer.writerow([
                s,
                st["n"],
                round(st["cer_avg"], 4),
                round(st["cer_std"], 4),
                round(st["cer_var"], 4),
                round(st["cer_ci95"], 4),
                round(st["ci_lower"], 4),
                round(st["ci_upper"], 4),
                round(st["cer_rme"], 4),
            ])
        writer.writerow([
            "Overall (Combined)",
            combined_stats["n"],
            round(combined_stats["cer_avg"], 4),
            round(combined_stats["cer_std"], 4),
            round(combined_stats["cer_var"], 4),
            round(combined_stats["cer_ci95"], 4),
            round(combined_stats["ci_lower"], 4),
            round(combined_stats["ci_upper"], 4),
            round(combined_stats["cer_rme"], 4),
        ])

        # Theoretical Normal PDF points for combined distribution
        writer.writerow([])
        writer.writerow([f"# THEORETICAL GAUSSIAN PDF EVALUATION POINTS (COMBINED - {phase_name})"])
        writer.writerow(["x_cer", "probability_density_f_x"])
        mu = combined_stats["cer_avg"]
        sigma = max(1e-4, combined_stats["cer_std"])
        x_min = min(all_phase_cers) - 0.5 * sigma if all_phase_cers else mu - 3.5 * sigma
        x_max = max(all_phase_cers) + 0.5 * sigma if all_phase_cers else mu + 3.5 * sigma
        x_samples = np.linspace(x_min, x_max, 50)
        for x_val in x_samples:
            pdf_val = compute_gaussian_pdf(float(x_val), mu, sigma)
            writer.writerow([round(float(x_val), 4), round(pdf_val, 6)])

    plot_normal_distribution(combined_stats, all_phase_cers, dist_png, phase_name=phase_name, dpi=dpi)
    print(f"  [8/8] Normal Distribution: {dist_csv} | {dist_png}")

    # --------------------------------------------------------------------------
    # COMBINED 8-PANEL DASHBOARD FOR THIS PHASE
    # --------------------------------------------------------------------------
    dashboard_png = os.path.join(phase_dir, f"all_metrics_{phase_slug}.png")
    plot_combined_phase_dashboard(
        phase_name=phase_name,
        seed_episodes=phase_episodes,
        conv_metrics=conv_metrics_per_seed,
        stats_per_seed=stats_per_seed,
        combined_stats=combined_stats,
        raw_cer_values=all_phase_cers,
        output_path=dashboard_png,
        dpi=dpi,
    )
    print(f"  [★] Combined Phase Dashboard: {dashboard_png}")

    return {
        "phase_name": phase_name,
        "phase_slug": phase_slug,
        "phase_dir": phase_dir,
        "stats_per_seed": stats_per_seed,
        "combined_stats": combined_stats,
        "conv_metrics": conv_metrics_per_seed,
        "dashboard_path": dashboard_png,
    }


def export_evaluation_suite(
    seed_episodes: Dict[int, List[Dict[str, Any]]],
    optimal_steps_dict: Dict[int, int],
    experiment_name: str,
    reports_dir: str = "reports",
    initial_phase_ratio: float = 0.8,
    dpi: int = 300,
    group_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete evaluation suite across both Initial Phase and Stable Phase.

    Creates the designated directory layout::

        reports/[group_name/]<experiment_name>/
        ├── initial_phase/
        │   ├── steps_per_episode/
        │   ├── current_cumulative_reward/
        │   ├── current_episodic_reward/
        │   ├── convergence_speed/
        │   ├── collision_rate/
        │   ├── coefficient_of_variation/
        │   ├── relative_margin_error/
        │   ├── normal_distribution/
        │   └── all_metrics_initial.png
        └── stable_phase/
            ├── steps_per_episode/
            ├── current_cumulative_reward/
            ├── current_episodic_reward/
            ├── convergence_speed/
            ├── collision_rate/
            ├── coefficient_of_variation/
            ├── relative_margin_error/
            ├── normal_distribution/
            └── all_metrics_stable.png

    If group_name is provided, also triggers cross-experiment comparison in::

        reports/<group_name>/comparison/
        ├── initial_phase/<metric_subfolder>/
        ├── stable_phase/<metric_subfolder>/
        └── all_metrics_comparison.png

    Args:
        seed_episodes (Dict[int, List[Dict[str, Any]]]): Mapping from seed to list of episode dicts.
            Each episode dict contains 'episode', 'steps', 'reward', 'collisions', 'status'.
        optimal_steps_dict (Dict[int, int]): Optimal geodesic distance per seed.
        experiment_name (str): Unique experiment name used as destination directory.
        reports_dir (str, optional): Base reports directory. Defaults to "reports".
        initial_phase_ratio (float, optional): Initial phase proportion (default: 0.8).
        dpi (int, optional): Plot resolution DPI. Defaults to 300.
        group_name (Optional[str], optional): Group name grouping sibling experiments.

    Returns:
        Dict[str, Any]: Export summary containing file paths and aggregated statistics.
    """
    clean_group = group_name.strip() if group_name and str(group_name).strip().lower() not in ("", "none") else None

    if clean_group:
        exp_dir = os.path.join(reports_dir, clean_group, experiment_name)
    else:
        exp_dir = os.path.join(reports_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    seeds = list(seed_episodes.keys())
    if not seeds:
        print("Warning: No seed data provided for evaluation.")
        return {}

    total_episodes = max(len(eps) for eps in seed_episodes.values())
    n_initial, n_stable = split_learning_phases(total_episodes, initial_phase_ratio)

    initial_pct = int(round(initial_phase_ratio * 100))
    stable_pct = 100 - initial_pct

    print(f"\n======================================================================")
    print(f"RUNNING LEARNING EVALUATION SUITE: {experiment_name}")
    if clean_group:
        print(f"Experimental Group: {clean_group}")
    print(f"Total Episodes: {total_episodes} | Initial Phase ({initial_pct}%): {n_initial} | Stable Phase ({stable_pct}%): {n_stable}")
    print(f"Destination Folder: {exp_dir}")
    print(f"======================================================================")

    # Slice episodes for each phase
    initial_seed_episodes = {s: eps[:n_initial] for s, eps in seed_episodes.items()}
    stable_seed_episodes = {s: eps[n_initial:total_episodes] for s, eps in seed_episodes.items()}

    # 1. Export all 8 metrics for INITIAL PHASE
    initial_dir = os.path.join(exp_dir, "initial_phase")
    initial_res = export_phase_metrics(
        phase_dir=initial_dir,
        phase_name=f"Initial Phase ({initial_pct}%)",
        phase_slug="initial",
        phase_episodes=initial_seed_episodes,
        optimal_steps_dict=optimal_steps_dict,
        dpi=dpi,
    )

    # 2. Export all 8 metrics for STABLE PHASE
    stable_dir = os.path.join(exp_dir, "stable_phase")
    stable_res = export_phase_metrics(
        phase_dir=stable_dir,
        phase_name=f"Stable Phase ({stable_pct}%)",
        phase_slug="stable",
        phase_episodes=stable_seed_episodes,
        optimal_steps_dict=optimal_steps_dict,
        dpi=dpi,
    )

    print(f"\nAll 8 metrics generated for BOTH phases inside: {exp_dir}")

    # --------------------------------------------------------------------------
    # GROUP COMPARISON (Automatically generated if group_name is configured)
    # --------------------------------------------------------------------------
    if clean_group:
        generate_group_comparison(reports_dir=reports_dir, group_name=clean_group, dpi=dpi)

    return {
        "group_name": clean_group,
        "experiment_name": experiment_name,
        "experiment_dir": exp_dir,
        "n_initial": n_initial,
        "n_stable": n_stable,
        "initial_metrics": initial_res,
        "stable_metrics": stable_res,
        "stats_per_seed": stable_res.get("stats_per_seed", {}),
        "combined_stats": stable_res.get("combined_stats", {}),
        "conv_metrics": initial_res.get("conv_metrics", {}),
    }
