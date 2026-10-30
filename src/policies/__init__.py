"""Exploration policies package.

Provides action-selection strategies including Epsilon-Greedy, UCB,
Boltzmann Softmax, and Gaussian Thompson Sampling, along with policy instantiation
factories and mode validation.
"""

from typing import Any, Dict, Optional
from src.policies.base import BasePolicy
from src.policies.epsilon_greedy import EpsilonGreedyPolicy


def create_policy(
    exploration_cfg: Dict[str, Any],
    env_type: str,
    seed: Optional[int] = None,
) -> BasePolicy:
    """Instantiate and validate an exploration policy based on configuration.

    Args:
        exploration_cfg (Dict[str, Any]): Exploration configuration dictionary.
        env_type (str): Environment type ('classic_va', 'standard', 'ann').
        seed (Optional[int], optional): Random seed. Defaults to None.

    Returns:
        BasePolicy: Configured exploration policy instance.

    Raises:
        ValueError: If adaptive_va mode is requested for non-classic_va environments,
            or if the requested strategy is unknown.
    """
    strategy = exploration_cfg.get("strategy", "epsilon_greedy").strip().lower()
    mode = exploration_cfg.get("mode", "static").strip().lower()

    # Enforce constraint: adaptive_va mode is only permissible in classic_va
    if mode == "adaptive_va" and env_type.strip().lower() not in ("classic_va", "affective", "va_classic"):
        raise ValueError(
            f"Policy mode 'adaptive_va' is strictly restricted to 'classic_va' environment, "
            f"but current environment is '{env_type}'. Use 'static' or 'exponential_decay'."
        )

    if strategy in ("epsilon_greedy", "eps_greedy", "epsilon"):
        epsilon = float(exploration_cfg.get("epsilon", 0.1))
        epsilon_init = float(exploration_cfg.get("epsilon_init", 1.0))
        epsilon_min = float(exploration_cfg.get("epsilon_min", 0.01))
        decay_rate = float(
            exploration_cfg.get("epsilon_decay", exploration_cfg.get("decay_rate", 0.95))
        )

        return EpsilonGreedyPolicy(
            mode=mode,
            epsilon=epsilon,
            epsilon_init=epsilon_init,
            epsilon_min=epsilon_min,
            decay_rate=decay_rate,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown exploration strategy '{strategy}'. Supported strategies: 'epsilon_greedy'."
        )


__all__ = ["BasePolicy", "EpsilonGreedyPolicy", "create_policy"]

