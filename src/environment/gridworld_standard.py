"""Standard GridWorld environment without affective monitoring.

This environment implements the standard Dyna-Q reinforcement learning benchmark
without internal Valence-Arousal emotion dynamics or frequency memory tracking.
"""

from typing import Any, Dict, List, Optional, Tuple
from src.environment.base_gridworld import BaseGridWorld


class StandardGridWorld(BaseGridWorld):
    """Standard GridWorld environment.

    Used for baseline Dyna-Q benchmarks using static or exponentially decaying
    parameters without affective state calculations.
    """

    def __init__(
        self,
        grid: List[List[int]],
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        max_steps: int = 2000,
        wall_value: int = 1,
        path_value: int = 0,
    ) -> None:
        """Initialize the StandardGridWorld.

        Args:
            grid (List[List[int]]): 2D grid matrix.
            start_pos (Tuple[int, int]): Start coordinate (row, col).
            goal_pos (Tuple[int, int]): Goal coordinate (row, col).
            step_reward (float, optional): Non-goal transition reward. Defaults to -1.0.
            goal_reward (float, optional): Terminal reward at goal. Defaults to 10.0.
            max_steps (int, optional): Maximum step limit per episode. Defaults to 2000.
            wall_value (int, optional): Obstacle identifier value. Defaults to 1.
            path_value (int, optional): Path identifier value. Defaults to 0.
        """
        super().__init__(
            grid=grid,
            start_pos=start_pos,
            goal_pos=goal_pos,
            step_reward=step_reward,
            goal_reward=goal_reward,
            max_steps=max_steps,
            wall_value=wall_value,
            path_value=path_value,
        )

    def step(
        self, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Perform a standard environment step.

        Args:
            action (int): Action index (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).

        Returns:
            Tuple[int, float, bool, bool, Dict[str, Any]]:
                (next_state, reward, terminated, truncated, info).
        """
        return super().step(action)

