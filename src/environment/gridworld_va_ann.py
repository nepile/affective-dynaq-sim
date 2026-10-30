"""Affective GridWorld environment for Artificial Neural Network (ANN) integration.

Provides sensory signals and reward modification hooks for neural network-based
affect appraisal without parameter modulation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from src.environment.base_gridworld import BaseGridWorld
from src.maze.validation import has_line_of_sight
import math


class ANNGridWorld(BaseGridWorld):
    """GridWorld designed for neural network-based affective reward shaping.

    Captures low-level sensory signals (collision, direction changes, Bresenham
    visibility, distance progression) and provides an interface for an ANN model
    to predict affective values (Valence, Arousal) and modulate rewards.

    Attributes:
        reward_modifier (Optional[Callable[[float, float, float], float]]):
            Optional callback (original_reward, valence, arousal) -> modified_reward.
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
        reward_modifier: Optional[Callable[[float, float, float], float]] = None,
    ) -> None:
        """Initialize the ANNGridWorld environment.

        Args:
            grid (List[List[int]]): 2D grid matrix.
            start_pos (Tuple[int, int]): Start coordinate (row, col).
            goal_pos (Tuple[int, int]): Goal coordinate (row, col).
            step_reward (float, optional): Step transition reward. Defaults to -1.0.
            goal_reward (float, optional): Terminal reward at goal. Defaults to 10.0.
            max_steps (int, optional): Step limit per episode. Defaults to 2000.
            wall_value (int, optional): Obstacle value. Defaults to 1.
            path_value (int, optional): Path value. Defaults to 0.
            reward_modifier (Optional[Callable], optional): Function to modify reward.
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
        self.reward_modifier = reward_modifier
        self.last_action: Optional[int] = None
        self.prev_dist_to_start: float = 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (Optional[int], optional): Optional seed. Defaults to None.

        Returns:
            Tuple[int, Dict[str, Any]]: Initial state and info dictionary.
        """
        initial_state, info = super().reset(seed=seed)
        self.last_action = None
        self.prev_dist_to_start = 0.0
        return initial_state, info

    def step(
        self, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Perform step and gather raw sensory inputs for ANN processing.

        Args:
            action (int): Action index.

        Returns:
            Tuple[int, float, bool, bool, Dict[str, Any]]:
                (next_state, reward, terminated, truncated, info).
        """
        dir_changed = (self.last_action is not None and action != self.last_action)
        self.last_action = action

        next_state, reward, terminated, truncated, info = super().step(action)
        collision = info["collision"]

        goal_in_sight = has_line_of_sight(
            self.grid, self.agent_pos, self.goal_pos, self.wall_value
        )

        curr_dist = math.hypot(
            self.agent_pos[0] - self.start_pos[0],
            self.agent_pos[1] - self.start_pos[1],
        )
        distance_increasing = (curr_dist > self.prev_dist_to_start)
        self.prev_dist_to_start = curr_dist

        # Default valence/arousal placeholders before ANN inference
        valence = 0.0
        arousal = 0.0

        # Apply external reward modifier if hooked
        if self.reward_modifier is not None and not terminated:
            reward = self.reward_modifier(reward, valence, arousal)

        info.update({
            "collision": collision,
            "dir_changed": dir_changed,
            "goal_in_sight": goal_in_sight,
            "dist_increasing": distance_increasing,
            "dist_to_start": curr_dist,
            "valence": valence,
            "arousal": arousal,
        })

        return next_state, reward, terminated, truncated, info

