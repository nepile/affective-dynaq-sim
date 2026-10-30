"""Base exploration policy interface.

Provides abstract interface for reinforcement learning action-selection strategies
supporting static, exponential decay, and affective adaptive modes.
"""

from abc import ABC, abstractmethod
from typing import Optional
from src.agent.q_table import QTable


class BasePolicy(ABC):
    """Abstract base class for exploration policies.

    Attributes:
        mode (str): Operating mode ('static', 'exponential_decay', or 'adaptive_va').
    """

    def __init__(self, mode: str = "static") -> None:
        """Initialize the exploration policy.

        Args:
            mode (str, optional): Policy mode ('static', 'exponential_decay', 'adaptive_va').
                Defaults to "static".

        Raises:
            ValueError: If mode is not one of the supported modes.
        """
        valid_modes = {"static", "exponential_decay", "adaptive_va"}
        mode_clean = mode.strip().lower()
        if mode_clean not in valid_modes:
            raise ValueError(f"Invalid policy mode '{mode}'. Expected one of {valid_modes}.")
        self.mode = mode_clean

    @abstractmethod
    def select_action(
        self,
        state: int,
        q_table: QTable,
        num_actions: int,
    ) -> int:
        """Select an action based on the state and Q-values.

        Args:
            state (int): Current discrete state index.
            q_table (QTable): Agent's action-value table.
            num_actions (int): Number of available actions.

        Returns:
            int: Selected action index.
        """
        pass

    @abstractmethod
    def decay_episode(self, episode: int) -> None:
        """Update policy parameters according to episode progression.

        Args:
            episode (int): Current episode index (1-based).
        """
        pass

    @abstractmethod
    def update_affect(self, valence: float, arousal: float) -> None:
        """Modulate policy parameters based on Core Affect (Valence and Arousal).

        Args:
            valence (float): Current Valence V(t) in [-1.0, 1.0].
            arousal (float): Current Arousal A(t) in [-1.0, 1.0].
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset policy parameters back to initial configuration for a new seed."""
        pass

    @abstractmethod
    def get_param_value(self) -> float:
        """Retrieve current primary exploration parameter value.

        Returns:
            float: Parameter value (e.g., epsilon, temperature, c).
        """
        pass

