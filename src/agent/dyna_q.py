"""Dyna-Q reinforcement learning agent implementation.

Implements the tabular Dyna-Q architecture combining direct Q-learning,
transition model learning, simulated planning sampling, modular exploration
policies, and learning rate scheduling (Sutton & Barto 1998; Feldmaier & Diepold 2014).
"""

import random
from typing import Optional
from src.agent.planning import DynaModel
from src.agent.q_table import QTable
from src.policies.base import BasePolicy
from src.policies.epsilon_greedy import EpsilonGreedyPolicy


class DynaQAgent:
    """Tabular Dyna-Q reinforcement learning agent.

    Combines direct reinforcement learning, environment model learning,
    planning updates from simulated experience, dynamic exploration policies,
    and adaptive learning rate schedules.

    Attributes:
        num_states (int): Number of discrete environment states.
        num_actions (int): Number of actions available.
        alpha (float): Current effective learning rate.
        alpha_mode (str): Learning rate schedule mode ('static', 'exponential_decay', 'adaptive_va').
        alpha_init (float): Initial learning rate for exponential decay.
        alpha_min (float): Floor value for decaying learning rate.
        alpha_decay (float): Multiplicative decay rate per episode.
        gamma (float): Discount rate parameter.
        planning_steps (int): Number of planning updates per real step (p_steps).
        policy (BasePolicy): Exploration strategy policy instance.
        q_table (QTable): Internal action-value function table.
        model (DynaModel): Internal environment transition memory.
        last_updated_state (Optional[int]): State index updated in the latest step.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.01,
        alpha_mode: str = "static",
        alpha_init: float = 0.1,
        alpha_min: float = 0.005,
        alpha_decay: float = 0.95,
        gamma: float = 0.95,
        planning_steps: int = 50,
        policy: Optional[BasePolicy] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the Dyna-Q agent.

        Args:
            num_states (int): Total number of states.
            num_actions (int): Total number of actions.
            alpha (float, optional): Baseline learning rate for static mode. Defaults to 0.01.
            alpha_mode (str, optional): Learning rate mode. Defaults to "static".
            alpha_init (float, optional): Initial alpha for decay. Defaults to 0.1.
            alpha_min (float, optional): Minimum alpha for decay. Defaults to 0.005.
            alpha_decay (float, optional): Alpha decay multiplier. Defaults to 0.95.
            gamma (float, optional): Discount factor. Defaults to 0.95.
            planning_steps (int, optional): Planning steps per real step. Defaults to 50.
            policy (Optional[BasePolicy], optional): Exploration policy instance.
            seed (Optional[int], optional): Random seed. Defaults to None.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.planning_steps = planning_steps

        # Learning rate modes
        self.alpha_mode = alpha_mode.strip().lower()
        self.static_alpha = float(alpha)
        self.alpha_init = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_decay = float(alpha_decay)

        if self.alpha_mode == "exponential_decay":
            self.alpha = self.alpha_init
        else:
            self.alpha = self.static_alpha

        self.rng = random.Random(seed)
        self.q_table = QTable(num_states, num_actions)
        self.model = DynaModel()
        self.last_updated_state: Optional[int] = None

        # Policy configuration
        if policy is not None:
            self.policy = policy
        else:
            self.policy = EpsilonGreedyPolicy(mode="static", epsilon=0.1, seed=seed)

    @property
    def epsilon(self) -> float:
        """Convenience property for current exploration parameter."""
        return self.policy.get_param_value()

    def select_action(self, state: int) -> int:
        """Select action using the configured exploration policy.

        Args:
            state (int): Discrete state index.

        Returns:
            int: Selected action index.
        """
        return self.policy.select_action(state, self.q_table, self.num_actions)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> None:
        """Update Q-table directly, record transition into model, and execute planning.

        Args:
            state (int): State before action.
            action (int): Executed action.
            reward (float): Received reward.
            next_state (int): Resulting state.
            terminated (bool): Whether the episode ended at next_state.
        """
        # 1. Direct Q-learning update (Eq. 1)
        self.last_updated_state = state
        current_q = self.q_table.get(state, action)
        max_next_q = 0.0 if terminated else max(self.q_table.get_action_values(next_state))
        td_target = reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.q_table.set(state, action, new_q)

        # 2. Model learning
        self.model.update(state, action, reward, next_state)

        # 3. Simulated planning updates
        if self.planning_steps > 0:
            self.model.plan(
                q_table=self.q_table,
                planning_steps=self.planning_steps,
                alpha=self.alpha,
                gamma=self.gamma,
                rng=self.rng,
            )

    def decay_episode(self, episode: int) -> None:
        """Decay exploration parameter and learning rate after an episode finishes.

        Args:
            episode (int): Completed episode index (1-based).
        """
        # Decay exploration parameter
        self.policy.decay_episode(episode)

        # Decay learning rate if exponential_decay mode
        if self.alpha_mode == "exponential_decay":
            self.alpha = max(
                self.alpha_min,
                self.alpha_init * (self.alpha_decay ** (episode - 1)),
            )

    def update_affect(self, valence: float, arousal: float) -> None:
        """Propagate Core Affect signals to policy and learning rate modules.

        Args:
            valence (float): Current Valence V(t).
            arousal (float): Current Arousal A(t).
        """
        self.policy.update_affect(valence, arousal)

    def reset(self) -> None:
        """Completely reset agent Q-table, model, policy parameters, and learning rate.

        Used when switching between multi-fixed seeds to guarantee run independence.
        """
        self.q_table.reset()
        self.model.reset()
        self.last_updated_state = None

        if self.alpha_mode == "exponential_decay":
            self.alpha = self.alpha_init
        else:
            self.alpha = self.static_alpha

        self.policy.reset()
