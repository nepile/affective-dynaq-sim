"""Epsilon-Greedy exploration policy implementation.

Provides action selection with exploration probability epsilon supporting static,
exponential decay, and adaptive affective modulation modes.
"""

import random
from typing import Optional
from src.agent.q_table import QTable
from src.policies.base import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):
    """Epsilon-Greedy exploration strategy.

    Balances exploration (selecting random actions with probability epsilon) and
    exploitation (selecting greedy action maximizing Q-value).

    Modes:
        - 'static': Constant epsilon across all episodes and steps.
        - 'exponential_decay': Epsilon starts at epsilon_init and decays exponentially
          per episode towards epsilon_min. Resets on new seed.
        - 'adaptive_va': Epsilon is modulated dynamically based on Valence-Arousal
          emotions (configured for classic_va environment).

    Attributes:
        mode (str): Policy mode ('static', 'exponential_decay', 'adaptive_va').
        epsilon (float): Current epsilon exploration probability.
        epsilon_init (float): Initial epsilon value for exponential decay.
        epsilon_min (float): Lower bound for decaying epsilon.
        decay_rate (float): Multiplicative decay factor per episode.
        rng (random.Random): Pseudorandom generator instance.
    """

    def __init__(
        self,
        mode: str = "static",
        epsilon: float = 0.1,
        epsilon_init: float = 1.0,
        epsilon_min: float = 0.01,
        decay_rate: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize EpsilonGreedyPolicy.

        Args:
            mode (str, optional): Policy operating mode. Defaults to "static".
            epsilon (float, optional): Constant epsilon for static mode. Defaults to 0.1.
            epsilon_init (float, optional): Initial epsilon for decay. Defaults to 1.0.
            epsilon_min (float, optional): Floor value for decaying epsilon. Defaults to 0.01.
            decay_rate (float, optional): Multiplicative decay rate per episode. Defaults to 0.95.
            seed (Optional[int], optional): Random seed. Defaults to None.

        Raises:
            ValueError: If epsilon values are not within [0, 1] or decay_rate <= 0.
        """
        super().__init__(mode=mode)

        self.static_epsilon = float(epsilon)
        self.epsilon_init = float(epsilon_init)
        self.epsilon_min = float(epsilon_min)
        self.decay_rate = float(decay_rate)
        self.rng = random.Random(seed)

        if not (0.0 <= self.static_epsilon <= 1.0):
            raise ValueError(f"Static epsilon must be in [0, 1], got {self.static_epsilon}.")
        if not (0.0 <= self.epsilon_min <= self.epsilon_init <= 1.0):
            raise ValueError(
                f"Epsilon values must satisfy 0 <= epsilon_min ({self.epsilon_min}) "
                f"<= epsilon_init ({self.epsilon_init}) <= 1.0."
            )
        if not (0.0 < self.decay_rate <= 1.0):
            raise ValueError(f"decay_rate must be in (0, 1], got {self.decay_rate}.")

        # Current working epsilon
        if self.mode == "exponential_decay":
            self.epsilon = self.epsilon_init
        else:
            self.epsilon = self.static_epsilon

    def select_action(
        self,
        state: int,
        q_table: QTable,
        num_actions: int,
    ) -> int:
        """Select an action using epsilon-greedy strategy.

        Args:
            state (int): Discrete state index.
            q_table (QTable): Action-value function table.
            num_actions (int): Total number of discrete actions.

        Returns:
            int: Selected action index.
        """
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, num_actions - 1)
        return q_table.get_best_action(state, rng=self.rng)

    def decay_episode(self, episode: int) -> None:
        """Decay epsilon exponentially at the end of each episode.

        Formula:
            epsilon = max(epsilon_min, epsilon_init * (decay_rate ** (episode - 1)))

        Args:
            episode (int): Current episode index (1-based).
        """
        if self.mode == "exponential_decay":
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_init * (self.decay_rate ** (episode - 1)),
            )

    def update_affect(self, valence: float, arousal: float) -> None:
        """Modulate epsilon using Valence-Arousal core affect signals.

        Note:
            Reserved for 'adaptive_va' mode in classic_va environment.

        Args:
            valence (float): Current Valence V(t) in [-1.0, 1.0].
            arousal (float): Current Arousal A(t) in [-1.0, 1.0].
        """
        if self.mode == "adaptive_va":
            # Reserved for upcoming adaptive_va modulation task
            pass

    def reset(self) -> None:
        """Reset epsilon back to initial state when switching independent seeds."""
        if self.mode == "exponential_decay":
            self.epsilon = self.epsilon_init
        else:
            self.epsilon = self.static_epsilon

    def get_param_value(self) -> float:
        """Get the current epsilon value.

        Returns:
            float: Current epsilon.
        """
        return self.epsilon

