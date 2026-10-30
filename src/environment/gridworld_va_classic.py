"""Affective-monitored GridWorld using classic Valence-Arousal dynamics.

Implements sensory tracking (collision, direction change, Bresenham line of sight,
and distance from start) and discrete dynamical frequency calculations according
to Feldmaier and Diepold (2014).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from src.affect.va_classic import ClassicValenceArousal
from src.environment.base_gridworld import BaseGridWorld
from src.maze.validation import has_line_of_sight


class ClassicVAGridWorld(BaseGridWorld):
    """GridWorld environment with integrated Valence-Arousal sensory monitoring.

    Tracks environmental events and reflection-level emotional states:
        - Collision detection with walls/boundaries (c)
        - Direction change monitoring (d)
        - Bresenham line-of-sight raycasting to the goal (g)
        - Distance progression from the starting location (i)

    Attributes:
        affect (ClassicValenceArousal): Core affect dynamical model.
        bias_reward_with_affect (bool): If True, sets non-terminal reward to 0
            when valence > 0 (Feldmaier & Diepold experiment 2).
    """

    def __init__(
        self,
        grid: List[List[int]],
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        collision_reward: float = -5.0,
        max_steps: int = 2000,
        wall_value: int = 1,
        path_value: int = 0,
        nu: float = 0.2,
        theta1: float = 0.2,
        theta2: float = 0.2,
        kappa: float = 0.15,
        bias_reward_with_affect: bool = False,
    ) -> None:
        """Initialize ClassicVAGridWorld environment.

        Args:
            grid (List[List[int]]): 2D grid matrix.
            start_pos (Tuple[int, int]): Start coordinate (row, col).
            goal_pos (Tuple[int, int]): Goal coordinate (row, col).
            step_reward (float, optional): Non-goal transition reward. Defaults to -1.0.
            goal_reward (float, optional): Terminal reward at goal. Defaults to 10.0.
            collision_reward (float, optional): Penalty for collisions. Defaults to -5.0.
            max_steps (int, optional): Step limit per episode. Defaults to 2000.
            wall_value (int, optional): Value representing walls. Defaults to 1.
            path_value (int, optional): Value representing paths. Defaults to 0.
            nu (float, optional): Slope factor for collision frequency. Defaults to 0.2.
            theta1 (float, optional): Slope factor for direction frequency. Defaults to 0.2.
            theta2 (float, optional): Slope factor for goal in sight frequency. Defaults to 0.2.
            kappa (float, optional): Slope factor for right direction frequency. Defaults to 0.15.
            bias_reward_with_affect (bool, optional): Whether to bias environment reward
                using positive valence. Defaults to False.
        """
        super().__init__(
            grid=grid,
            start_pos=start_pos,
            goal_pos=goal_pos,
            step_reward=step_reward,
            goal_reward=goal_reward,
            collision_reward=collision_reward,
            max_steps=max_steps,
            wall_value=wall_value,
            path_value=path_value,
        )

        self.affect = ClassicValenceArousal(
            nu=nu,
            theta1=theta1,
            theta2=theta2,
            kappa=kappa,
        )
        self.bias_reward_with_affect = bias_reward_with_affect

        self.last_action: Optional[int] = None
        self.prev_dist_to_start: float = 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment and the internal emotional state.

        Args:
            seed (Optional[int], optional): Optional seed. Defaults to None.

        Returns:
            Tuple[int, Dict[str, Any]]: Initial state index and info dictionary.
        """
        initial_state, info = super().reset(seed=seed)
        self.affect.reset()
        self.last_action = None
        self.prev_dist_to_start = 0.0

        # Initial sensory check
        goal_in_sight = has_line_of_sight(
            self.grid, self.agent_pos, self.goal_pos, self.wall_value
        )

        info.update({
            "valence": self.affect.valence,
            "arousal": self.affect.arousal,
            "quadrant": self.affect.get_quadrant(),
            "quadrant_label": self.affect.get_quadrant_label(),
            "f_col": self.affect.f_col,
            "f_dir": self.affect.f_dir,
            "f_goal": self.affect.f_goal,
            "f_right": self.affect.f_right,
            "goal_in_sight": goal_in_sight,
        })
        return initial_state, info

    def step(
        self, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Execute one step with full affective evaluation.

        Args:
            action (int): Action index (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).

        Returns:
            Tuple[int, float, bool, bool, Dict[str, Any]]:
                (next_state, reward, terminated, truncated, info).
        """
        # Direction change detection: d = 1 if current action != previous action
        dir_changed = (self.last_action is not None and action != self.last_action)
        self.last_action = action

        # Execute physical grid transition
        next_state, reward, terminated, truncated, info = super().step(action)
        collision = info["collision"]

        # Goal in sight via Bresenham algorithm: g = 1 if unblocked line of sight
        goal_in_sight = has_line_of_sight(
            self.grid, self.agent_pos, self.goal_pos, self.wall_value
        )

        # Distance to start: i = 1 if moving away from start, -1 otherwise
        curr_dist = math.hypot(
            self.agent_pos[0] - self.start_pos[0],
            self.agent_pos[1] - self.start_pos[1],
        )
        distance_increasing = (curr_dist > self.prev_dist_to_start)
        self.prev_dist_to_start = curr_dist

        # Reflection level affective state update
        affect_metrics = self.affect.update(
            collision=collision,
            dir_changed=dir_changed,
            goal_in_sight=goal_in_sight,
            distance_increasing=distance_increasing,
        )

        # Optional reward biasing (Feldmaier & Diepold Exp 2)
        if self.bias_reward_with_affect and not terminated and self.affect.valence > 0.0:
            reward = 0.0

        info.update({
            "valence": affect_metrics["valence"],
            "arousal": affect_metrics["arousal"],
            "quadrant": self.affect.get_quadrant(),
            "quadrant_label": self.affect.get_quadrant_label(),
            "f_col": affect_metrics["f_col"],
            "f_dir": affect_metrics["f_dir"],
            "f_goal": affect_metrics["f_goal"],
            "f_right": affect_metrics["f_right"],
            "goal_in_sight": goal_in_sight,
            "dir_changed": dir_changed,
            "dist_increasing": distance_increasing,
            "dist_to_start": curr_dist,
        })

        return next_state, reward, terminated, truncated, info
