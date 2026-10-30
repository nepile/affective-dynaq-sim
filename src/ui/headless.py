"""Headless runner for reinforcement learning experiments.

Executes simulation benchmarks in the terminal without a GUI, manages multi-fixed
seeds independently, logs training performance, and triggers image exports.
"""

import os
import csv
from typing import Any, Dict, List, Optional
from src.agent.dyna_q import DynaQAgent
from src.environment import ClassicVAGridWorld, StandardGridWorld, ANNGridWorld, BaseGridWorld
from src.evaluation import export_evaluation_suite
from src.maze import (
    compute_maximum_geodesic_distance,
    generate_ca_maze,
    generate_kruskal_maze,
)
from src.policies import create_policy
from src.ui.renderer import resolve_env_image_path, save_environment_image
from src.ui.theme import COLOR_GOAL, COLOR_GRID_LINE, COLOR_START, COLOR_WALL


def create_maze(
    generator_name: str,
    height: int,
    width: int,
    seed: int,
) -> List[List[int]]:
    """Generate a maze grid using the requested algorithm and seed.

    Args:
        generator_name (str): Maze generator ('kruskal' or 'cellular_automata').
        height (int): Grid height.
        width (int): Grid width.
        seed (int): Random seed.

    Returns:
        List[List[int]]: Generated 2D grid matrix.
    """
    gen_lower = generator_name.strip().lower()
    if gen_lower in ("kruskal", "randomized_kruskal"):
        return generate_kruskal_maze(height=height, width=width, seed=seed)
    elif gen_lower in ("ca", "cellular_automata"):
        return generate_ca_maze(height=height, width=width, seed=seed)
    else:
        raise ValueError(f"Unknown generator '{generator_name}'. Choose 'kruskal' or 'cellular_automata'.")


def create_environment(
    env_type: str,
    grid: List[List[int]],
    start_pos: tuple,
    goal_pos: tuple,
    max_steps: int,
    env_cfg: Dict[str, Any],
    affect_config: Dict[str, Any],
) -> BaseGridWorld:
    """Instantiate the requested environment type.

    Args:
        env_type (str): Environment type ('standard', 'classic_va', 'ann').
        grid (List[List[int]]): 2D grid matrix.
        start_pos (tuple): Origin coordinate.
        goal_pos (tuple): Target coordinate.
        max_steps (int): Step limit per episode.
        env_cfg (Dict[str, Any]): General environment configuration.
        affect_config (Dict[str, Any]): Affective parameters.

    Returns:
        BaseGridWorld: Configured environment instance.
    """
    step_rew = float(env_cfg.get("step_reward", -1.0))
    goal_rew = float(env_cfg.get("goal_reward", 10.0))

    env_lower = env_type.strip().lower()
    if env_lower in ("classic_va", "affective", "va_classic"):
        # Neville Jeremy's research environment: collision penalty of -5.0
        collision_rew = float(env_cfg.get("collision_reward", -5.0))
        return ClassicVAGridWorld(
            grid=grid,
            start_pos=start_pos,
            goal_pos=goal_pos,
            step_reward=step_rew,
            goal_reward=goal_rew,
            collision_reward=collision_rew,
            max_steps=max_steps,
            nu=float(affect_config.get("nu", 0.2)),
            theta1=float(affect_config.get("theta1", 0.2)),
            theta2=float(affect_config.get("theta2", 0.2)),
            kappa=float(affect_config.get("kappa", 0.15)),
            bias_reward_with_affect=str(affect_config.get("bias_reward_with_affect", "false")).lower() == "true",
        )
    elif env_lower in ("ann", "gridworld_va_ann"):
        # Friend's research environment: standard step reward on collision
        return ANNGridWorld(
            grid=grid,
            start_pos=start_pos,
            goal_pos=goal_pos,
            step_reward=step_rew,
            goal_reward=goal_rew,
            max_steps=max_steps,
        )
    else:
        # Standard baseline environment
        return StandardGridWorld(
            grid=grid,
            start_pos=start_pos,
            goal_pos=goal_pos,
            step_reward=step_rew,
            goal_reward=goal_rew,
            max_steps=max_steps,
        )


def run_headless_experiment(config: Dict[str, Dict[str, Any]]) -> None:
    """Run full reinforcement learning simulation across all specified seeds.

    Args:
        config (Dict[str, Dict[str, Any]]): Parsed configuration dictionary.
    """
    env_cfg = config.get("environment", {})
    agent_cfg = config.get("agent", {})
    exploration_cfg = config.get("exploration", {})
    affect_cfg = config.get("affect", {})
    sys_cfg = config.get("system", {})

    generator = env_cfg.get("generator", "kruskal")
    height = int(env_cfg.get("height", 9))
    width = int(env_cfg.get("width", 9))
    env_type = env_cfg.get("env_type", "classic_va")
    max_steps = int(env_cfg.get("max_steps", 2000))

    # Parse seeds list (e.g., [1, 3, 5, 6] or "1, 3, 5, 6")
    raw_seeds = env_cfg.get("seeds", [1])
    if isinstance(raw_seeds, str):
        seeds = [int(s.strip()) for s in raw_seeds.split(",") if s.strip()]
    elif isinstance(raw_seeds, (list, tuple)):
        seeds = [int(s) for s in raw_seeds]
    else:
        seeds = [int(raw_seeds)]

    episodes = int(agent_cfg.get("episodes", 20))
    alpha = float(agent_cfg.get("alpha", 0.01))
    alpha_mode = agent_cfg.get("alpha_mode", "static").strip().lower()
    alpha_init = float(agent_cfg.get("alpha_init", 0.1))
    alpha_min = float(agent_cfg.get("alpha_min", 0.005))
    alpha_decay = float(agent_cfg.get("alpha_decay", 0.95))

    # Validate adaptive_va constraint for alpha
    if alpha_mode == "adaptive_va" and env_type.strip().lower() not in ("classic_va", "affective", "va_classic"):
        raise ValueError(
            f"Learning rate mode 'adaptive_va' is strictly restricted to 'classic_va' environment, "
            f"but current environment is '{env_type}'. Use 'static' or 'exponential_decay'."
        )

    gamma = float(agent_cfg.get("gamma", 0.95))
    p_steps = int(agent_cfg.get("planning_steps", 50))

    policy_strat = exploration_cfg.get("strategy", "epsilon_greedy")
    policy_mode = exploration_cfg.get("mode", "static")

    eval_cfg = config.get("evaluation", {})
    initial_phase_ratio = float(eval_cfg.get("initial_phase_ratio", sys_cfg.get("initial_phase_ratio", 0.8)))
    group_name = sys_cfg.get("group_name", None)
    experiment_name = sys_cfg.get("experiment_name", "dyna_q_default")
    reports_dir = sys_cfg.get("reports_dir", "reports")
    save_images = str(sys_cfg.get("save_env_images", "true")).lower() == "true"
    save_env_in_exp = str(sys_cfg.get("save_env_in_experiment_dir", "true")).lower() == "true"

    all_seed_episodes: Dict[int, List[Dict[str, Any]]] = {}
    optimal_steps_dict: Dict[int, int] = {}

    print("=" * 70)
    print(f"STARTING DYNA-Q AFFECTIVE SIMULATION: {experiment_name.upper()} (HEADLESS)")
    if group_name and str(group_name).strip().lower() not in ("", "none"):
        print(f"Group: {group_name.strip()}")
    print(f"Generator: {generator} | Grid: {height}x{width} | Type: {env_type}")
    print(f"Policy: {policy_strat} ({policy_mode}) | Alpha: {alpha} ({alpha_mode})")
    print(f"Seeds: {seeds} | Episodes per seed: {episodes} | Planning steps: {p_steps}")
    print(f"Evaluation Phase Ratio: {int(initial_phase_ratio * 100)}% Initial / {int((1 - initial_phase_ratio) * 100)}% Stable")
    print("=" * 70)

    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"\n--- Running Seed [{seed}] ({seed_idx}/{len(seeds)}) ---")

        # 1. Generate maze and find optimal path
        grid = create_maze(generator, height, width, seed)
        start_pos, goal_pos, max_geodesic, optimal_path = compute_maximum_geodesic_distance(grid)
        optimal_steps_dict[seed] = max_geodesic
        print(f"Start: {start_pos} -> Goal: {goal_pos} | Max Geodesic Distance: {max_geodesic}")

        # 2. Save environment visualization image
        if save_images:
            img_dest = resolve_env_image_path(
                reports_dir=reports_dir,
                experiment_name=experiment_name,
                generator_name=generator,
                seed=seed,
                save_in_experiment_dir=save_env_in_exp,
                group_name=group_name,
            )
            img_path = save_environment_image(
                grid=grid,
                start_pos=start_pos,
                goal_pos=goal_pos,
                generator_name=generator,
                seed=seed,
                output_dir=reports_dir,
                geodesic_distance=max_geodesic,
                optimal_path=optimal_path,
                obstacle_color=COLOR_WALL,
                grid_line_color=COLOR_GRID_LINE,
                start_color=COLOR_START,
                goal_color=COLOR_GOAL,
                save_path=img_dest,
            )
            print(f"Saved environment figure: {img_path}")

        # 3. Create independent Environment, Exploration Policy, and Agent
        env = create_environment(
            env_type=env_type,
            grid=grid,
            start_pos=start_pos,
            goal_pos=goal_pos,
            max_steps=max_steps,
            env_cfg=env_cfg,
            affect_config=affect_cfg,
        )

        policy = create_policy(exploration_cfg, env_type, seed=seed)

        agent = DynaQAgent(
            num_states=env.num_states,
            num_actions=env.num_actions,
            alpha=alpha,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_decay=alpha_decay,
            gamma=gamma,
            planning_steps=p_steps,
            policy=policy,
            seed=seed,
        )

        # 4. Run episodes
        seed_results: List[Dict[str, Any]] = []

        for ep in range(1, episodes + 1):
            state, info = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            collisions = 0
            valences: List[float] = []
            arousals: List[float] = []

            curr_alpha = agent.alpha
            curr_eps = agent.epsilon

            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)

                if step_info.get("collision", False):
                    collisions += 1

                if "valence" in step_info:
                    agent.update_affect(step_info["valence"], step_info["arousal"])
                    valences.append(step_info["valence"])
                    arousals.append(step_info["arousal"])

                state = next_state
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            # Decay parameters at the end of the episode
            agent.decay_episode(ep + 1)

            avg_v = sum(valences) / len(valences) if valences else 0.0
            avg_a = sum(arousals) / len(arousals) if arousals else 0.0
            status_str = "GOAL REACHED" if not truncated else "TRUNCATED"

            seed_results.append({
                "seed": seed,
                "episode": ep,
                "steps": steps,
                "reward": round(total_reward, 2),
                "collisions": collisions,
                "alpha": round(curr_alpha, 4),
                "epsilon": round(curr_eps, 4),
                "avg_valence": round(avg_v, 4),
                "avg_arousal": round(avg_a, 4),
                "status": status_str,
            })

            if ep % max(1, episodes // 5) == 0 or ep == 1 or ep == episodes:
                print(
                    f"  Ep {ep:2d}/{episodes} | Steps: {steps:4d} | "
                    f"Reward: {total_reward:6.1f} | Collisions: {collisions:2d} | "
                    f"Alpha: {curr_alpha:.4f} | Eps: {curr_eps:.4f} | "
                    f"Avg V: {avg_v:+.3f} | Avg A: {avg_a:+.3f} | {status_str}"
                )

        all_seed_episodes[seed] = seed_results

        # 5. Completely reset agent, Q-table, and policy for next independent seed
        agent.reset()
        print(f"Reset agent Q-table, learning rate, and policy for next independent seed.")

    # 6. Run comprehensive learning evaluation suite across both phases
    if all_seed_episodes and any(len(v) > 0 for v in all_seed_episodes.values()):
        export_evaluation_suite(
            seed_episodes=all_seed_episodes,
            optimal_steps_dict=optimal_steps_dict,
            experiment_name=experiment_name,
            reports_dir=reports_dir,
            initial_phase_ratio=initial_phase_ratio,
            group_name=group_name,
        )

    print("\n" + "=" * 70)
    print("ALL SEEDS AND EVALUATION REPORTS COMPLETED SUCCESSFULLY.")
    print("=" * 70)

