"""Dyna model learning and simulated planning memory.

Stores environment transition dynamics experienced by the agent and performs
planning updates via random sampling of previously experienced state-action pairs.
"""

import random
from typing import Dict, List, Optional, Set, Tuple
from src.agent.q_table import QTable


class DynaModel:
    """Deterministic environment transition model for Dyna-Q planning.

    Attributes:
        transitions (Dict[Tuple[int, int], Tuple[float, int]]):
            Mapping from (state, action) -> (reward, next_state).
        visited_states (List[int]): List of states previously encountered.
        actions_taken (Dict[int, Set[int]]): Set of actions taken in each visited state.
    """

    def __init__(self) -> None:
        """Initialize empty Dyna transition model."""
        self.transitions: Dict[Tuple[int, int], Tuple[float, int]] = {}
        self.visited_states: Set[int] = set()
        self.actions_taken: Dict[int, Set[int]] = {}

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Record an observed environment transition.

        Args:
            state (int): Origin state index.
            action (int): Executed action index.
            reward (float): Observed reward.
            next_state (int): Resulting state index.
        """
        self.transitions[(state, action)] = (reward, next_state)
        self.visited_states.add(state)
        if state not in self.actions_taken:
            self.actions_taken[state] = set()
        self.actions_taken[state].add(action)

    def plan(
        self,
        q_table: QTable,
        planning_steps: int,
        alpha: float,
        gamma: float,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Execute Dyna-Q planning updates using simulated experience.

        Randomly samples previously visited (state, action) pairs from memory
        and applies Q-learning update (Eq. 1).

        Args:
            q_table (QTable): The agent's Q-table to update.
            planning_steps (int): Number of simulated transitions to replay (p_steps).
            alpha (float): Step-size / learning rate.
            gamma (float): Discount rate.
            rng (Optional[random.Random], optional): Random number generator.
        """
        if not self.visited_states or planning_steps <= 0:
            return

        state_list = list(self.visited_states)
        _rng = rng if rng is not None else random

        for _ in range(planning_steps):
            s = _rng.choice(state_list)
            actions = list(self.actions_taken[s])
            a = _rng.choice(actions)

            reward, next_s = self.transitions[(s, a)]

            # Q-learning update on simulated transition
            current_q = q_table.get(s, a)
            max_next_q = max(q_table.get_action_values(next_s))
            td_target = reward + gamma * max_next_q
            new_q = current_q + alpha * (td_target - current_q)
            q_table.set(s, a, new_q)

    def reset(self) -> None:
        """Clear all stored transitions and model history."""
        self.transitions.clear()
        self.visited_states.clear()
        self.actions_taken.clear()

