"""Classic Valence-Arousal dimensional emotion model.

Implements the discrete dynamical frequency system and Core Affect calculations
from Feldmaier and Diepold (2014): "Path-finding Using Reinforcement Learning
and Affective States".
"""

from typing import Dict, Tuple


class ClassicValenceArousal:
    """Valence-Arousal emotion model based on Feldmaier and Diepold (2014).

    Maintains four short-term memory frequency variables:
        - f_col: Frequency of wall collisions.
        - f_dir: Frequency of action direction changes.
        - f_goal: Frequency of target goal visibility.
        - f_right: Frequency of moving in the right direction (away from start).

    Attributes:
        nu (float): Slope factor for collision frequency update (default: 0.2).
        theta1 (float): Slope factor for direction change update (default: 0.2).
        theta2 (float): Slope factor for goal in sight update (default: 0.2).
        kappa (float): Slope factor for right direction update (default: 0.15).
        f_col (float): Current collision frequency in [-1.0, 1.0].
        f_dir (float): Current direction change frequency in [-1.0, 1.0].
        f_goal (float): Current goal in sight frequency in [-1.0, 1.0].
        f_right (float): Current right direction frequency in [-1.0, 1.0].
        valence (float): Current Valence V(t) in [-1.0, 1.0].
        arousal (float): Current Arousal A(t) in [-1.0, 1.0].
    """

    def __init__(
        self,
        nu: float = 0.2,
        theta1: float = 0.2,
        theta2: float = 0.2,
        kappa: float = 0.15,
        init_f_col: float = 0.0,
        init_f_dir: float = 0.0,
        init_f_goal: float = 0.0,
        init_f_right: float = 0.0,
    ) -> None:
        """Initialize parameters and frequency variables.

        Args:
            nu (float, optional): Collision frequency slope. Defaults to 0.2.
            theta1 (float, optional): Direction change frequency slope. Defaults to 0.2.
            theta2 (float, optional): Goal in sight frequency slope. Defaults to 0.2.
            kappa (float, optional): Right direction frequency slope. Defaults to 0.15.
            init_f_col (float, optional): Initial f_col value. Defaults to 0.0.
            init_f_dir (float, optional): Initial f_dir value. Defaults to 0.0.
            init_f_goal (float, optional): Initial f_goal value. Defaults to 0.0.
            init_f_right (float, optional): Initial f_right value. Defaults to 0.0.
        """
        self.nu = float(nu)
        self.theta1 = float(theta1)
        self.theta2 = float(theta2)
        self.kappa = float(kappa)

        self.init_f_col = float(init_f_col)
        self.init_f_dir = float(init_f_dir)
        self.init_f_goal = float(init_f_goal)
        self.init_f_right = float(init_f_right)

        self.f_col = init_f_col
        self.f_dir = init_f_dir
        self.f_goal = init_f_goal
        self.f_right = init_f_right

        self.valence = 0.0
        self.arousal = 0.0

    def reset(self) -> None:
        """Reset all frequency values and emotion coordinates to initial values."""
        self.f_col = self.init_f_col
        self.f_dir = self.init_f_dir
        self.f_goal = self.init_f_goal
        self.f_right = self.init_f_right
        self.valence = 0.0
        self.arousal = 0.0

    def get_quadrant(self) -> str:
        """Determine the emotional quadrant in the Valence-Arousal core affect plane.

        Returns:
            str: One of 'Q1' (Excited/Happy), 'Q2' (Relaxed/Calm),
                 'Q3' (Sad/Depressed), or 'Q4' (Anxious/Fearful).
        """
        if self.valence >= 0.0:
            return "Q1" if self.arousal >= 0.0 else "Q2"
        else:
            return "Q4" if self.arousal >= 0.0 else "Q3"

    def get_quadrant_label(self) -> str:
        """Get descriptive psychological label for the current quadrant.

        Returns:
            str: Affect label ('Excited/Happy', 'Relaxed/Calm', 'Sad/Depressed', 'Anxious/Fearful').
        """
        labels = {
            "Q1": "Excited / Happy",
            "Q2": "Relaxed / Calm",
            "Q3": "Sad / Depressed",
            "Q4": "Anxious / Fearful",
        }
        return labels[self.get_quadrant()]

    def update(
        self,
        collision: bool,
        dir_changed: bool,
        goal_in_sight: bool,
        distance_increasing: bool,
    ) -> Dict[str, float]:
        """Update discrete dynamical frequencies and compute Core Affect (V, A).

        Equations from Feldmaier & Diepold (2014):
            f_col(t)   = f_col(t-1) + ( (1 if c else -1) - f_col(t-1) ) * nu
            f_dir(t)   = f_dir(t-1) + ( (1 if d else -1) - f_dir(t-1) ) * theta1
            f_goal(t)  = f_goal(t-1) + ( (1 if g else -1) - f_goal(t-1) ) * theta2
            f_right(t) = f_right(t-1) + ( (1 if i else -1) - f_right(t-1) ) * kappa
            A(t) = (f_dir + f_col) / 2 if g == 0 else (f_dir + f_col + f_goal) / 3
            V(t) = (f_goal + f_right) / 2

        Args:
            collision (bool): Whether agent collided with a wall in current step (c).
            dir_changed (bool): Whether current action changed from previous action (d).
            goal_in_sight (bool): Whether unblocked line of sight exists to goal (g).
            distance_increasing (bool): Whether distance to start increased (i = 1 or -1).

        Returns:
            Dict[str, float]: Dictionary containing updated valence, arousal, and frequencies.
        """
        # 1. Collision frequency update (Eq. 2)
        target_col = 1.0 if collision else -1.0
        self.f_col = self.f_col + (target_col - self.f_col) * self.nu

        # 2. Direction change frequency update (Eq. 3)
        target_dir = 1.0 if dir_changed else -1.0
        self.f_dir = self.f_dir + (target_dir - self.f_dir) * self.theta1

        # 3. Goal in sight frequency update (Eq. Section IV)
        target_goal = 1.0 if goal_in_sight else -1.0
        self.f_goal = self.f_goal + (target_goal - self.f_goal) * self.theta2

        # 4. Right way / distance frequency update (Eq. 5)
        target_right = 1.0 if distance_increasing else -1.0
        self.f_right = self.f_right + (target_right - self.f_right) * self.kappa

        # 5. Arousal calculation (Eq. 4)
        if goal_in_sight:
            self.arousal = (self.f_dir + self.f_col + self.f_goal) / 3.0
        else:
            self.arousal = (self.f_dir + self.f_col) / 2.0

        # 6. Valence calculation (Eq. 6)
        self.valence = (self.f_goal + self.f_right) / 2.0

        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "f_col": self.f_col,
            "f_dir": self.f_dir,
            "f_goal": self.f_goal,
            "f_right": self.f_right,
        }