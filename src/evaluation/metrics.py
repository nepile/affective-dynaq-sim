"""Mathematical and statistical metrics for evaluating Dyna-Q agent learning performance.

Implements evaluation across two distinct phases (Initial Phase 80% vs Stable Phase 20%):

- Initial Phase:
    1. Steps per episode
    2. Current Cumulative Reward (CCR)
    3. Current Episodic Reward (CER)
    4. Convergence Speed

- Stable Phase:
    5. Collision Rate
    6. Coefficient of Variation (CV) based on CER
    7. Relative Margin Error (RME) based on CER
    8. Gaussian Normal Distribution parameters based on CER
"""

import math
from typing import Any, Dict, List, Optional, Tuple


def split_learning_phases(total_episodes: int, initial_phase_ratio: float = 0.8) -> Tuple[int, int]:
    """Calculate the number of episodes for the initial phase and stable phase.

    Args:
        total_episodes (int): Total number of learning episodes (N_total).
        initial_phase_ratio (float, optional): Proportion for initial phase (0.0 to 1.0).
            Defaults to 0.8 (80% initial phase, 20% stable phase).

    Returns:
        Tuple[int, int]: (n_initial, n_stable), where n_initial + n_stable == total_episodes.
    """
    initial_phase_ratio = max(0.1, min(0.95, float(initial_phase_ratio)))
    n_initial = max(1, int(round(total_episodes * initial_phase_ratio)))
    if n_initial >= total_episodes:
        n_initial = max(1, total_episodes - 1)
    n_stable = total_episodes - n_initial
    return n_initial, n_stable


def compute_cer_std(cer_values: List[float], cer_avg: float) -> float:
    r"""Compute standard deviation of CER (Cumulative Episodic Reward).

    Formula (from specification document):

    .. math::

        CER_{std} = \sqrt{\frac{1}{N} \sum_{i=1}^N (CER_i - CER_{avg})^2}

    Args:
        cer_values (List[float]): List of episodic rewards in the stable phase.
        cer_avg (float): Mean of CER in the stable phase.

    Returns:
        float: Population standard deviation of CER.
    """
    n = len(cer_values)
    if n == 0:
        return 0.0
    variance = sum((x - cer_avg) ** 2 for x in cer_values) / n
    return math.sqrt(max(0.0, variance))


def compute_cer_ci95(cer_std: float, n: int) -> float:
    r"""Compute 95% Confidence Interval half-width margin of error.

    Formula (from specification document):

    .. math::

        CER_{ci95} = 1.96 \times \frac{CER_{std}}{\sqrt{N}}

    Args:
        cer_std (float): Standard deviation of CER.
        n (int): Sample size (number of episodes N in the stable phase).

    Returns:
        float: Half-width margin of error for 95% confidence interval.
    """
    if n <= 0:
        return 0.0
    return 1.96 * (cer_std / math.sqrt(n))


def compute_cer_cv(cer_std: float, cer_avg: float) -> float:
    r"""Compute Coefficient of Variation (CV) in percent.

    Formula (from specification document):

    .. math::

        CER_{cv} = \frac{CER_{std}}{|CER_{avg}|} \times 100\%

    Args:
        cer_std (float): Standard deviation of CER.
        cer_avg (float): Mean of CER.

    Returns:
        float: Coefficient of variation expressed as a percentage (%).
    """
    abs_avg = abs(cer_avg)
    if abs_avg < 1e-9:
        return 0.0
    return (cer_std / abs_avg) * 100.0


def compute_cer_rme(cer_ci95: float, cer_avg: float) -> float:
    r"""Compute Relative Margin Error (RME) in percent.

    Formula (from specification document):

    .. math::

        CER_{rme} = \frac{CER_{ci95}}{|CER_{avg}|} \times 100\%

    Args:
        cer_ci95 (float): 95% confidence interval margin of error.
        cer_avg (float): Mean of CER.

    Returns:
        float: Relative margin error expressed as a percentage (%).
    """
    abs_avg = abs(cer_avg)
    if abs_avg < 1e-9:
        return 0.0
    return (cer_ci95 / abs_avg) * 100.0


def compute_gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    r"""Evaluate normal (Gaussian) probability density function f(x).

    Formula:

    .. math::

        f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right)

    Args:
        x (float): Evaluation point.
        mu (float): Mean (CER_avg).
        sigma (float): Standard deviation (CER_std).

    Returns:
        float: Probability density at x.
    """
    if sigma <= 1e-9:
        return 1.0 if abs(x - mu) < 1e-6 else 0.0
    factor = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * (((x - mu) / sigma) ** 2)
    return factor * math.exp(exponent)


def compute_cer_phase_statistics(cer_values: List[float]) -> Dict[str, Any]:
    """Compute complete statistical metrics for CER across any learning phase.

    Calculates N, CER_avg, CER_std, CER_var, CER_ci95, CI boundaries, CER_cv, and CER_rme.

    Args:
        cer_values (List[float]): List of episodic rewards from a learning phase.

    Returns:
        Dict[str, Any]: Dictionary containing N, CER_avg, CER_std, CER_var,
        CER_ci95, CI boundaries, CER_cv, and CER_rme.
    """
    n = len(cer_values)
    if n == 0:
        return {
            "n": 0,
            "cer_avg": 0.0,
            "cer_std": 0.0,
            "cer_var": 0.0,
            "cer_ci95": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "cer_cv": 0.0,
            "cer_rme": 0.0,
        }

    cer_avg = sum(cer_values) / n
    cer_std = compute_cer_std(cer_values, cer_avg)
    cer_var = cer_std ** 2
    cer_ci95 = compute_cer_ci95(cer_std, n)
    cer_cv = compute_cer_cv(cer_std, cer_avg)
    cer_rme = compute_cer_rme(cer_ci95, cer_avg)

    return {
        "n": n,
        "cer_avg": cer_avg,
        "cer_std": cer_std,
        "cer_var": cer_var,
        "cer_ci95": cer_ci95,
        "ci_lower": cer_avg - cer_ci95,
        "ci_upper": cer_avg + cer_ci95,
        "cer_cv": cer_cv,
        "cer_rme": cer_rme,
    }


# Backward compatibility alias
compute_stable_phase_statistics = compute_cer_phase_statistics


def compute_convergence_metrics(
    steps_list: List[int],
    rewards_list: List[float],
    optimal_steps: Optional[int] = None,
    window_size: int = 5,
    tolerance_ratio: float = 0.25,
    episode_numbers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Calculate convergence episode and speed metrics during a learning phase.

    Identifies when the agent achieves consistent task performance. An agent is
    considered converged when moving average steps approach optimal geodesic distance
    within tolerance_ratio (or stabilizes with low relative variance) for at least
    window_size consecutive episodes.

    Args:
        steps_list (List[int]): Steps per episode during the phase.
        rewards_list (List[float]): Rewards per episode during the phase.
        optimal_steps (Optional[int], optional): Geodesic shortest path length.
        window_size (int, optional): Sliding window for stability check. Defaults to 5.
        tolerance_ratio (float, optional): Permissible margin above optimal. Defaults to 0.25.
        episode_numbers (Optional[List[int]], optional): Actual episode numbers in this phase.

    Returns:
        Dict[str, Any]: Convergence details including convergence_episode,
        episodes_to_converge, moving averages, and convergence status.
    """
    total_initial = len(steps_list)
    if total_initial == 0:
        return {
            "convergence_episode": None,
            "episodes_to_converge": None,
            "converged": False,
            "optimal_steps": optimal_steps or 0,
            "final_moving_avg_steps": 0.0,
            "moving_avg_steps": [],
            "moving_avg_rewards": [],
        }

    window = max(2, min(window_size, total_initial // 2 if total_initial >= 4 else total_initial))

    moving_avg_steps: List[float] = []
    moving_avg_rewards: List[float] = []
    for i in range(total_initial):
        start_idx = max(0, i - window + 1)
        sub_steps = steps_list[start_idx : i + 1]
        sub_rewards = rewards_list[start_idx : i + 1]
        moving_avg_steps.append(sum(sub_steps) / len(sub_steps))
        moving_avg_rewards.append(sum(sub_rewards) / len(sub_rewards))

    # Determine baseline target threshold
    target_threshold: float
    if optimal_steps and optimal_steps > 0:
        target_threshold = optimal_steps * (1.0 + tolerance_ratio)
    else:
        # Fallback to 1.15 * min of moving average steps in latter half
        half = total_initial // 2
        latter_min = min(moving_avg_steps[half:]) if half < total_initial else min(moving_avg_steps)
        target_threshold = latter_min * 1.25

    convergence_ep: Optional[int] = None
    episodes_to_converge: Optional[int] = None
    consecutive_stable = 0
    required_consecutive = max(2, window)

    for i in range(window - 1, total_initial):
        if moving_avg_steps[i] <= target_threshold:
            consecutive_stable += 1
            if consecutive_stable >= required_consecutive:
                # First episode index of this stable sequence
                rel_idx = i - required_consecutive + 2
                if episode_numbers and rel_idx < len(episode_numbers):
                    convergence_ep = episode_numbers[rel_idx]
                else:
                    convergence_ep = rel_idx + 1
                episodes_to_converge = rel_idx + 1
                break
        else:
            consecutive_stable = 0

    converged = convergence_ep is not None
    if not converged:
        episodes_to_converge = total_initial

    return {
        "convergence_episode": convergence_ep,
        "episodes_to_converge": episodes_to_converge,
        "converged": converged,
        "optimal_steps": optimal_steps or 0,
        "target_threshold": target_threshold,
        "final_moving_avg_steps": moving_avg_steps[-1] if moving_avg_steps else 0.0,
        "moving_avg_steps": moving_avg_steps,
        "moving_avg_rewards": moving_avg_rewards,
    }