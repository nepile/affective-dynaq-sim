"""Base GridWorld environment for Dyna-Q reinforcement learning.

Provides core grid navigation mechanics, deterministic transition dynamics,
coordinate mappings, and collision detection.
"""

from typing import Any, Dict, List, Optional, Tuple


class BaseGridWorld:
    """Base 2D GridWorld navigation environment.

    Represents a rectangular grid where an agent navigates from a start state to
    a goal state while avoiding wall obstacles.

    Actions:
        0: UP    (-1,  0)
        1: DOWN  ( 1,  0)
        2: LEFT  ( 0, -1)
        3: RIGHT ( 0,  1)

    Attributes:
        height (int): Grid height (number of rows).
        width (int): Grid width (number of columns).
        grid (List[List[int]]): 2D matrix (0 for path, 1 for wall).
        start_pos (Tuple[int, int]): Coordinates of the starting location.
        goal_pos (Tuple[int, int]): Coordinates of the target location.
        agent_pos (Tuple[int, int]): Current coordinates of the agent.
        step_reward (float): Reward received at each non-goal step (default: -1.0).
        goal_reward (float): Reward received upon reaching the goal (default: 10.0).
        collision_reward (float): Penalty received upon collision (default: step_reward).
        max_steps (int): Maximum allowable steps per episode.
        current_step (int): Step count in the current episode.
    """

    # Action mappings: (row_delta, col_delta)
    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    ACTION_NAMES: Dict[int, str] = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    def __init__(
        self,
        grid: List[List[int]],
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        collision_reward: Optional[float] = None,
        max_steps: int = 2000,
        wall_value: int = 1,
        path_value: int = 0,
    ) -> None:
        """Initialize the BaseGridWorld environment.

        Args:
            grid (List[List[int]]): 2D grid matrix.
            start_pos (Tuple[int, int]): Start coordinate (row, col).
            goal_pos (Tuple[int, int]): Goal coordinate (row, col).
            step_reward (float, optional): Reward for normal transitions. Defaults to -1.0.
            goal_reward (float, optional): Reward for reaching goal. Defaults to 10.0.
            collision_reward (Optional[float], optional): Penalty for collisions.
                Defaults to step_reward if not specified.
            max_steps (int, optional): Episode step limit. Defaults to 2000.
            wall_value (int, optional): Cell value for obstacles. Defaults to 1.
            path_value (int, optional): Cell value for passable paths. Defaults to 0.

        Raises:
            ValueError: If start_pos or goal_pos is located on a wall.
        """
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.grid = [row[:] for row in grid]
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward if collision_reward is not None else step_reward
        self.max_steps = max_steps
        self.wall_value = wall_value
        self.path_value = path_value

        if self.grid[start_pos[0]][start_pos[1]] == wall_value:
            raise ValueError(f"Start position {start_pos} cannot be a wall cell.")
        if self.grid[goal_pos[0]][goal_pos[1]] == wall_value:
            raise ValueError(f"Goal position {goal_pos} cannot be a wall cell.")

        self.agent_pos = start_pos
        self.current_step = 0
        self.total_episodes = 0

    @property
    def num_states(self) -> int:
        """Total number of discrete states in the grid (height * width)."""
        return self.height * self.width

    @property
    def num_actions(self) -> int:
        """Total number of discrete actions available (4)."""
        return len(self.ACTIONS)

    def coord_to_state(self, coord: Tuple[int, int]) -> int:
        """Convert a 2D grid coordinate (row, col) to a 1D discrete state index.

        Args:
            coord (Tuple[int, int]): (row, col) coordinate.

        Returns:
            int: Flattened integer state index.
        """
        r, c = coord
        return r * self.width + c

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        """Convert a 1D discrete state index back to (row, col) coordinates.

        Args:
            state (int): Flattened integer state index.

        Returns:
            Tuple[int, int]: Corresponding (row, col) coordinate.
        """
        r = state // self.width
        c = state % self.width
        return (r, c)

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int], optional): Optional seed for random reset. Defaults to None.

        Returns:
            Tuple[int, Dict[str, Any]]: Initial state integer index and info dictionary.
        """
        self.agent_pos = self.start_pos
        self.current_step = 0
        self.total_episodes += 1
        initial_state = self.coord_to_state(self.agent_pos)
        info = {
            "coord": self.agent_pos,
            "step": self.current_step,
            "total_episodes": self.total_episodes,
        }
        return initial_state, info

    def is_valid_position(self, r: int, c: int) -> bool:
        """Check if a coordinate is within bounds and not an obstacle.

        Args:
            r (int): Row coordinate.
            c (int): Column coordinate.

        Returns:
            bool: True if navigable, False if out of bounds or a wall.
        """
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r][c] != self.wall_value
        return False

    def step(
        self, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Execute one environmental step given an action.

        Args:
            action (int): Action index (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).

        Returns:
            Tuple[int, float, bool, bool, Dict[str, Any]]:
                (next_state, reward, terminated, truncated, info).
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Expected one of {list(self.ACTIONS.keys())}.")

        self.current_step += 1
        dr, dc = self.ACTIONS[action]
        target_r = self.agent_pos[0] + dr
        target_c = self.agent_pos[1] + dc

        collision = False
        if self.is_valid_position(target_r, target_c):
            self.agent_pos = (target_r, target_c)
        else:
            # Wall or boundary collision; agent remains in current position
            collision = True

        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self.current_step >= self.max_steps) and not terminated

        if terminated:
            reward = self.goal_reward
        elif collision:
            reward = self.collision_reward
        else:
            reward = self.step_reward

        next_state = self.coord_to_state(self.agent_pos)

        info: Dict[str, Any] = {
            "coord": self.agent_pos,
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "collision": collision,
            "step": self.current_step,
        }

        return next_state, reward, terminated, truncated, info
