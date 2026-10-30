"""Tabular Q-value storage for reinforcement learning.

Manages state-action value table Q(s, a) with initialization, lookup, and updates.
"""

from typing import Dict, List, Optional, Tuple
import random


class QTable:
    """Tabular state-action value function representation.

    Attributes:
        num_states (int): Number of distinct states.
        num_actions (int): Number of actions available per state.
        table (Dict[Tuple[int, int], float]): Mapping from (state, action) to Q-value.
        default_value (float): Default initial Q-value (typically 0.0).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        default_value: float = 0.0,
    ) -> None:
        """Initialize the Q-table.

        Args:
            num_states (int): Total number of discrete states.
            num_actions (int): Total number of discrete actions.
            default_value (float, optional): Initial value for unseen state-actions. Defaults to 0.0.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.default_value = default_value
        self.table: Dict[Tuple[int, int], float] = {}

    def get(self, state: int, action: int) -> float:
        """Retrieve the Q-value for a given state-action pair.

        Args:
            state (int): Discrete state index.
            action (int): Action index.

        Returns:
            float: Estimated action-value Q(s, a).
        """
        return self.table.get((state, action), self.default_value)

    def set(self, state: int, action: int, value: float) -> None:
        """Set the Q-value for a given state-action pair.

        Args:
            state (int): Discrete state index.
            action (int): Action index.
            value (float): New Q-value.
        """
        self.table[(state, action)] = value

    def get_action_values(self, state: int) -> List[float]:
        """Get all action Q-values for a given state.

        Args:
            state (int): Discrete state index.

        Returns:
            List[float]: List of Q-values for actions 0 to num_actions - 1.
        """
        return [self.get(state, a) for a in range(self.num_actions)]

    def get_best_action(self, state: int, rng: Optional[random.Random] = None) -> int:
        """Get the greedy action with highest Q-value, breaking ties randomly.

        Args:
            state (int): Discrete state index.
            rng (Optional[random.Random], optional): Random generator for tie-breaking.

        Returns:
            int: Action index with highest Q-value.
        """
        values = self.get_action_values(state)
        max_val = max(values)
        best_actions = [a for a, v in enumerate(values) if v == max_val]

        if rng is not None:
            return rng.choice(best_actions)
        return random.choice(best_actions)

    def reset(self) -> None:
        """Clear all learned Q-values back to the initial default value."""
        self.table.clear()

