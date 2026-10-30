"""Interactive Graphical User Interface (GUI) for Dyna-Q affective simulation.

Uses Pygame to visualize agent navigation, environment layouts, real-time
Valence-Arousal emotion dynamics, a scrollable Q-table inspector with order
sorting (State Ascending vs Recently Updated with green highlight), and an
interactive Start/Pause control button.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import pygame

from src.agent.dyna_q import DynaQAgent
from src.environment import ClassicVAGridWorld, StandardGridWorld, ANNGridWorld, BaseGridWorld
from src.evaluation import export_evaluation_suite
from src.maze import compute_maximum_geodesic_distance
from src.policies import create_policy
from src.ui.headless import create_environment, create_maze
from src.ui.renderer import resolve_env_image_path, save_environment_image
from src.ui.theme import (
    COLOR_ACCENT,
    COLOR_AGENT,
    COLOR_BG,
    COLOR_BTN_HOVER,
    COLOR_BTN_PAUSE,
    COLOR_BTN_START,
    COLOR_DROPDOWN_BG,
    COLOR_DROPDOWN_BORDER,
    COLOR_DROPDOWN_HOVER,
    COLOR_GOAL,
    COLOR_GRID_LINE,
    COLOR_PATH,
    COLOR_Q_BEST,
    COLOR_ROW_ACTIVE,
    COLOR_ROW_ALT,
    COLOR_ROW_UPDATED,
    COLOR_SCROLLBAR,
    COLOR_SCROLLBAR_THUMB,
    COLOR_SIDEBAR_BG,
    COLOR_SIDEBAR_PANEL,
    COLOR_START,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    COLOR_WALL,
    COLOR_WALL_BORDER,
    COLOR_WALL_HIT,
    COLOR_WALL_HIT_BORDER,
)


def run_gui_experiment(config: Dict[str, Dict[str, Any]]) -> None:
    """Run reinforcement learning simulation with an interactive Pygame GUI.

    Features:
        - Animated agent navigation with physical collision recoil and shockwave on the grid.
        - Clickable Start/Pause control button.
        - Scrollable real-time Q-table inspector with dropdown sorting:
          * State (Ascending): Ordered by coordinate (0..N).
          * Recently Updated: State just updated immediately jumps to the top with green highlight.
        - Full display of all 4 short-term memory frequencies (f_col, f_dir, f_goal, f_right)
          and Core Affect (Valence & Arousal) for classic_va environment.

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

    if alpha_mode == "adaptive_va" and env_type.strip().lower() not in ("classic_va", "affective", "va_classic"):
        raise ValueError(
            f"Learning rate mode 'adaptive_va' is strictly restricted to 'classic_va' environment, "
            f"but current environment is '{env_type}'. Use 'static' or 'exponential_decay'."
        )

    gamma = float(agent_cfg.get("gamma", 0.95))
    p_steps = int(agent_cfg.get("planning_steps", 50))

    reports_dir = sys_cfg.get("reports_dir", "reports")
    save_images = str(sys_cfg.get("save_env_images", "true")).lower() == "true"

    # Pygame initialization with fallback if headless display
    try:
        pygame.init()
        pygame.font.init()
    except Exception as e:
        print(f"Warning: Cannot initialize Pygame display ({e}). Falling back to headless mode.")
        from src.ui.headless import run_headless_experiment
        run_headless_experiment(config)
        return

    # Configurable GUI window dimensions
    gui_cell_size_cfg = str(sys_cfg.get("gui_cell_size", "auto")).strip().lower()
    gui_win_w_cfg = str(sys_cfg.get("gui_window_width", "auto")).strip().lower()
    gui_win_h_cfg = str(sys_cfg.get("gui_window_height", "auto")).strip().lower()
    sidebar_width = int(sys_cfg.get("gui_sidebar_width", 350))

    if gui_win_w_cfg != "auto" and gui_win_h_cfg != "auto":
        WINDOW_WIDTH = int(gui_win_w_cfg)
        WINDOW_HEIGHT = int(gui_win_h_cfg)
        available_grid_w = max(50, WINDOW_WIDTH - sidebar_width)
        available_grid_h = WINDOW_HEIGHT
        CELL_SIZE = max(8, min(available_grid_w // width, available_grid_h // height))
        GRID_PIXEL_WIDTH = width * CELL_SIZE
        GRID_PIXEL_HEIGHT = height * CELL_SIZE
        SIDEBAR_WIDTH = WINDOW_WIDTH - GRID_PIXEL_WIDTH
    elif gui_cell_size_cfg != "auto":
        CELL_SIZE = max(8, int(gui_cell_size_cfg))
        GRID_PIXEL_WIDTH = width * CELL_SIZE
        GRID_PIXEL_HEIGHT = height * CELL_SIZE
        SIDEBAR_WIDTH = sidebar_width
        WINDOW_WIDTH = GRID_PIXEL_WIDTH + SIDEBAR_WIDTH
        WINDOW_HEIGHT = max(GRID_PIXEL_HEIGHT, 620)
    else:
        # Default auto-scaling based on grid dimensions
        TARGET_GRID_PIXELS = 450
        CELL_SIZE = max(10, min(26, TARGET_GRID_PIXELS // max(height, width)))
        GRID_PIXEL_WIDTH = width * CELL_SIZE
        GRID_PIXEL_HEIGHT = height * CELL_SIZE
        SIDEBAR_WIDTH = sidebar_width
        WINDOW_WIDTH = GRID_PIXEL_WIDTH + SIDEBAR_WIDTH
        WINDOW_HEIGHT = max(GRID_PIXEL_HEIGHT, 620)

    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Affective Dyna-Q Simulator")
    except Exception as e:
        print(f"Display unavailable ({e}). Falling back to headless mode.")
        pygame.quit()
        from src.ui.headless import run_headless_experiment
        run_headless_experiment(config)
        return

    clock = pygame.time.Clock()
    font_main = pygame.font.SysFont("DejaVu Sans", 11)
    font_header = pygame.font.SysFont("DejaVu Sans", 20, bold=True)
    font_title = pygame.font.SysFont("DejaVu Sans", 14, bold=True)
    font_qtable = pygame.font.SysFont("DejaVu Sans Mono", 10)
    font_btn = pygame.font.SysFont("DejaVu Sans", 11, bold=True)
    font_dropdown = pygame.font.SysFont("DejaVu Sans", 10)

    fps = 15
    paused = False
    q_scroll_offset = 0

    # Start/Pause Button Rect
    btn_start_pause_rect = pygame.Rect(GRID_PIXEL_WIDTH + 14, 38, SIDEBAR_WIDTH - 28, 28)

    # Dropdown sorting states: default to 'recently_updated'
    sort_mode = "recently_updated"  # Options: 'recently_updated' or 'state_asc'
    dropdown_open = False
    init_dd_w = min(136, SIDEBAR_WIDTH - 24)
    init_dd_x = WINDOW_WIDTH - 10 - init_dd_w if SIDEBAR_WIDTH >= 220 else GRID_PIXEL_WIDTH + 10
    dropdown_box_rect = pygame.Rect(init_dd_x, 244, init_dd_w, 22)
    dropdown_opt1_rect = pygame.Rect(dropdown_box_rect.x, dropdown_box_rect.bottom, dropdown_box_rect.w, 22)
    dropdown_opt2_rect = pygame.Rect(dropdown_box_rect.x, dropdown_box_rect.bottom + 22, dropdown_box_rect.w, 22)

    # Q-table dimensions
    q_table_x = GRID_PIXEL_WIDTH + 10
    q_table_w = SIDEBAR_WIDTH - 20
    row_height = 18

    # Tracking recently updated states for 'recently_updated' mode
    recently_updated_order: List[Tuple[int, int]] = []
    last_updated_coord: Optional[Tuple[int, int]] = None

    def draw_board(hit_pos: Optional[Tuple[int, int]] = None) -> None:
        """Render grid cells, obstacles, start, and goal positions."""
        # 1. Fill cell backgrounds (obstacles and paths)
        for r in range(height):
            for c in range(width):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell_color = COLOR_WALL if grid[r][c] == 1 else COLOR_PATH
                pygame.draw.rect(screen, cell_color, rect)

        # 2. Draw single crisp 1-pixel thin grid lines across the maze
        for r in range(height + 1):
            y = r * CELL_SIZE
            pygame.draw.line(screen, COLOR_GRID_LINE, (0, y), (GRID_PIXEL_WIDTH, y), 1)

        for c in range(width + 1):
            x = c * CELL_SIZE
            pygame.draw.line(screen, COLOR_GRID_LINE, (x, 0), (x, GRID_PIXEL_HEIGHT), 1)

        # 3. Draw obstacle borders (normal state, non-collision)
        if COLOR_WALL_BORDER is not None:
            for r in range(height):
                for c in range(width):
                    if grid[r][c] == 1:
                        rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(screen, COLOR_WALL_BORDER, rect, 1)

        # 4. Collision hit flash on obstacle if collision occurred
        if hit_pos is not None:
            hr, hc = hit_pos
            hit_rect = pygame.Rect(hc * CELL_SIZE, hr * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLOR_WALL_HIT, hit_rect)
            pygame.draw.rect(screen, COLOR_WALL_HIT_BORDER, hit_rect, 2)

        # 4. Draw Start (S)
        sr_rect = pygame.Rect(start_pos[1] * CELL_SIZE, start_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLOR_START, sr_rect)
        s_lbl = font_header.render("S", True, (255, 255, 255))
        screen.blit(s_lbl, (sr_rect.centerx - s_lbl.get_width() // 2, sr_rect.centery - s_lbl.get_height() // 2))

        # 5. Draw Goal (G)
        gr_rect = pygame.Rect(goal_pos[1] * CELL_SIZE, goal_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLOR_GOAL, gr_rect)
        g_lbl = font_header.render("G", True, (255, 255, 255))
        screen.blit(g_lbl, (gr_rect.centerx - g_lbl.get_width() // 2, gr_rect.centery - g_lbl.get_height() // 2))

    def draw_sidebar_hud(
        ep: int,
        steps: int,
        total_reward: float,
        step_info: Dict[str, Any],
        agent_instance: DynaQAgent,
        open_cells_list: List[Tuple[int, int]],
        current_agent_pos: Tuple[int, int],
    ) -> None:
        """Render the complete sidebar including Start/Pause, metrics with all frequencies, and Q-table."""
        sidebar_rect = pygame.Rect(GRID_PIXEL_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(screen, COLOR_SIDEBAR_BG, sidebar_rect)

        mouse_pos = pygame.mouse.get_pos()

        # 1. Title
        title_surf = font_title.render("DYNA-Q SIMULATOR", True, COLOR_ACCENT)
        screen.blit(title_surf, (GRID_PIXEL_WIDTH + 14, 12))

        # 2. Interactive Start / Pause Button
        btn_hovered = btn_start_pause_rect.collidepoint(mouse_pos)
        if paused:
            btn_color = COLOR_BTN_HOVER if btn_hovered else COLOR_BTN_START
            btn_text = "START / RESUME"
        else:
            btn_color = COLOR_BTN_HOVER if btn_hovered else COLOR_BTN_PAUSE
            btn_text = "PAUSE"

        pygame.draw.rect(screen, btn_color, btn_start_pause_rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), btn_start_pause_rect, 1, border_radius=5)
        btn_txt_surf = font_btn.render(btn_text, True, (255, 255, 255))
        screen.blit(
            btn_txt_surf,
            (btn_start_pause_rect.centerx - btn_txt_surf.get_width() // 2,
             btn_start_pause_rect.centery - btn_txt_surf.get_height() // 2),
        )

        # 3. Simulation & Affective Metrics (Preserves all 4 frequencies f_col, f_dir, f_goal, f_right)
        hud_y = 72
        info_lines = [
            f"Gen: {generator.title()} | Seed: {seed}",
            f"Policy: {agent_instance.policy.mode} (eps={agent_instance.epsilon:.3f})",
            f"Episode: {ep} / {episodes}  |  Step: {steps} / {max_steps}",
            f"Reward: {total_reward:.1f}  |  Alpha: {agent_instance.alpha:.3f}",
            f"Speed: {fps} FPS [Up/Dn]",
        ]

        if "valence" in step_info:
            info_lines.extend([
                f"Affect: {step_info.get('quadrant', '')} ({step_info.get('quadrant_label', '')})",
                f"V: {step_info['valence']:+.3f}  |  A: {step_info['arousal']:+.3f}",
                f"f_col: {step_info.get('f_col', 0.0):+.3f}  |  f_dir: {step_info.get('f_dir', 0.0):+.3f}",
                f"f_goal: {step_info.get('f_goal', 0.0):+.3f} |  f_right: {step_info.get('f_right', 0.0):+.3f}",
                f"Sensory: Col={step_info.get('collision', False)} | LoS={step_info.get('goal_in_sight', False)}",
            ])

        for line in info_lines:
            txt_surf = font_main.render(line, True, COLOR_TEXT)
            screen.blit(txt_surf, (GRID_PIXEL_WIDTH + 14, hud_y))
            hud_y += 16

        # 4. Divider Line
        hud_y += 4
        pygame.draw.line(
            screen,
            (55, 65, 80),
            (GRID_PIXEL_WIDTH + 10, hud_y),
            (WINDOW_WIDTH - 10, hud_y),
            1,
        )
        hud_y += 8

        # 5. Q-Table Header & Order Dropdown
        dropdown_h = 22
        dropdown_w = min(136, SIDEBAR_WIDTH - 24)
        single_line = (SIDEBAR_WIDTH >= 220)

        if single_line:
            dropdown_box_rect.x = WINDOW_WIDTH - 10 - dropdown_w
            dropdown_box_rect.y = hud_y
            dropdown_box_rect.w = dropdown_w
            dropdown_box_rect.h = dropdown_h

            # Q-TABLE Title aligned horizontally with dropdown
            q_title = font_header.render("Q-TABLE", True, COLOR_ACCENT)
            title_y = hud_y + (dropdown_h - q_title.get_height()) // 2
            screen.blit(q_title, (GRID_PIXEL_WIDTH + 12, title_y))

            hud_y = dropdown_box_rect.bottom + 8
        else:
            # Narrow sidebar: separate lines to ensure no overlap
            q_title = font_header.render("Q-TABLE", True, COLOR_ACCENT)
            screen.blit(q_title, (GRID_PIXEL_WIDTH + 12, hud_y))
            hud_y += q_title.get_height() + 6

            dropdown_box_rect.x = GRID_PIXEL_WIDTH + 10
            dropdown_box_rect.y = hud_y
            dropdown_box_rect.w = dropdown_w
            dropdown_box_rect.h = dropdown_h

            hud_y = dropdown_box_rect.bottom + 8

        # Always update dropdown option rects to match current dropdown_box_rect position
        dropdown_opt1_rect.x = dropdown_box_rect.x
        dropdown_opt1_rect.y = dropdown_box_rect.bottom
        dropdown_opt1_rect.w = dropdown_box_rect.w
        dropdown_opt1_rect.h = dropdown_h

        dropdown_opt2_rect.x = dropdown_box_rect.x
        dropdown_opt2_rect.y = dropdown_box_rect.bottom + dropdown_h
        dropdown_opt2_rect.w = dropdown_box_rect.w
        dropdown_opt2_rect.h = dropdown_h

        # Render Dropdown Button
        dropdown_label = "▼ By State (Asc)" if sort_mode == "state_asc" else "▼ Recently Updated"
        dd_hovered = dropdown_box_rect.collidepoint(mouse_pos)
        dd_bg_color = COLOR_DROPDOWN_HOVER if dd_hovered else COLOR_DROPDOWN_BG
        pygame.draw.rect(screen, dd_bg_color, dropdown_box_rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_DROPDOWN_BORDER, dropdown_box_rect, 1, border_radius=4)
        dd_surf = font_dropdown.render(dropdown_label, True, COLOR_TEXT)
        dd_txt_y = dropdown_box_rect.y + (dropdown_box_rect.h - dd_surf.get_height()) // 2
        screen.blit(dd_surf, (dropdown_box_rect.x + 6, dd_txt_y))

        # 6. Q-Table Column Headers
        col_header_rect = pygame.Rect(q_table_x, hud_y, q_table_w, 18)
        pygame.draw.rect(screen, COLOR_SIDEBAR_PANEL, col_header_rect)
        pygame.draw.rect(screen, (60, 75, 95), col_header_rect, 1)

        avail_w = q_table_w - 12
        state_col_w = max(46, int(avail_w * 0.22))
        action_col_w = max(36, (avail_w - state_col_w) // 4)

        headers = [("State", state_col_w), ("UP", action_col_w), ("DOWN", action_col_w), ("LEFT", action_col_w), ("RIGHT", action_col_w)]
        col_x = q_table_x + 6
        for title, col_w in headers:
            h_surf = font_qtable.render(title, True, COLOR_ACCENT)
            screen.blit(h_surf, (col_x, hud_y + 3))
            col_x += col_w

        hud_y += 20

        # 7. Q-Table Body
        body_h = WINDOW_HEIGHT - hud_y - 10
        body_rect = pygame.Rect(q_table_x, hud_y, q_table_w, body_h)
        pygame.draw.rect(screen, (22, 27, 34), body_rect)
        pygame.draw.rect(screen, (55, 68, 85), body_rect, 1)

        visible_rows = max(1, body_h // row_height)

        # Determine display order based on sort_mode
        if sort_mode == "recently_updated":
            # Display recently updated order first, then remaining open cells
            display_cells = recently_updated_order[:]
            existing_set = set(display_cells)
            for c in open_cells_list:
                if c not in existing_set:
                    display_cells.append(c)
        else:
            # Ascending coordinate order
            display_cells = open_cells_list

        total_open = len(display_cells)
        max_scroll = max(0, total_open - visible_rows)

        nonlocal q_scroll_offset
        q_scroll_offset = max(0, min(max_scroll, q_scroll_offset))

        # Render visible rows
        curr_y = hud_y + 2
        for i in range(q_scroll_offset, min(total_open, q_scroll_offset + visible_rows)):
            cell_coord = display_cells[i]
            s_idx = cell_coord[0] * width + cell_coord[1]
            q_vals = [agent_instance.q_table.get(s_idx, a) for a in range(4)]
            max_q = max(q_vals)
            best_a = agent_instance.q_table.get_best_action(s_idx)

            is_agent_here = (cell_coord == current_agent_pos)
            is_just_updated = (cell_coord == last_updated_coord)

            row_rect = pygame.Rect(q_table_x + 1, curr_y - 1, q_table_w - 10, row_height)

            # Row background coloring: highlight active agent position with blue
            if is_agent_here:
                pygame.draw.rect(screen, COLOR_ROW_ACTIVE, row_rect)
            elif (i % 2) == 1:
                pygame.draw.rect(screen, COLOR_ROW_ALT, row_rect)

            # State Coordinate label
            if is_agent_here:
                state_label = f">({cell_coord[0]:2d},{cell_coord[1]:2d})"
            else:
                state_label = f" ({cell_coord[0]:2d},{cell_coord[1]:2d})"

            text_color = (255, 255, 255) if is_agent_here else COLOR_TEXT
            s_surf = font_qtable.render(state_label, True, text_color)
            screen.blit(s_surf, (q_table_x + 2, curr_y))

            # Action Q-Values
            val_x = q_table_x + 6 + state_col_w
            for a_idx, val in enumerate(q_vals):
                val_str = f"{val:+.2f}"
                is_best = (a_idx == best_a and max_q != 0.0)
                if is_best:
                    val_color = COLOR_Q_BEST
                elif is_agent_here:
                    val_color = (255, 255, 255)
                else:
                    val_color = COLOR_TEXT_DIM

                val_surf = font_qtable.render(val_str, True, val_color)
                screen.blit(val_surf, (val_x, curr_y))
                val_x += action_col_w

            curr_y += row_height

        # Scrollbar on the right edge
        if total_open > visible_rows:
            scrollbar_track = pygame.Rect(q_table_x + q_table_w - 8, hud_y + 2, 6, body_h - 4)
            pygame.draw.rect(screen, COLOR_SCROLLBAR, scrollbar_track, border_radius=3)

            thumb_height = max(15, int((visible_rows / total_open) * (body_h - 4)))
            scroll_ratio = q_scroll_offset / max_scroll if max_scroll > 0 else 0
            thumb_y = hud_y + 2 + int(scroll_ratio * (body_h - 4 - thumb_height))
            thumb_rect = pygame.Rect(q_table_x + q_table_w - 8, thumb_y, 6, thumb_height)
            pygame.draw.rect(screen, COLOR_SCROLLBAR_THUMB, thumb_rect, border_radius=3)

        # 8. Dropdown Overlay Menu (Drawn on top when opened)
        if dropdown_open:
            opt1_hover = dropdown_opt1_rect.collidepoint(mouse_pos)
            opt2_hover = dropdown_opt2_rect.collidepoint(mouse_pos)

            menu_bg_rect = pygame.Rect(
                dropdown_box_rect.x, dropdown_box_rect.bottom,
                dropdown_box_rect.w, dropdown_opt1_rect.h + dropdown_opt2_rect.h
            )
            pygame.draw.rect(screen, COLOR_DROPDOWN_BG, menu_bg_rect)
            pygame.draw.rect(screen, COLOR_DROPDOWN_BORDER, menu_bg_rect, 1)

            # Option 1
            if opt1_hover:
                pygame.draw.rect(screen, COLOR_DROPDOWN_HOVER, dropdown_opt1_rect)
            lbl1 = font_dropdown.render("  1. By State (Asc)", True, (255, 255, 255) if opt1_hover else COLOR_TEXT)
            screen.blit(lbl1, (dropdown_opt1_rect.x + 4, dropdown_opt1_rect.y + (dropdown_opt1_rect.h - lbl1.get_height()) // 2))

            # Option 2
            if opt2_hover:
                pygame.draw.rect(screen, COLOR_DROPDOWN_HOVER, dropdown_opt2_rect)
            lbl2 = font_dropdown.render("  2. Recently Updated", True, (255, 255, 255) if opt2_hover else COLOR_TEXT)
            screen.blit(lbl2, (dropdown_opt2_rect.x + 4, dropdown_opt2_rect.y + (dropdown_opt2_rect.h - lbl2.get_height()) // 2))

    eval_cfg = config.get("evaluation", {})
    initial_phase_ratio = float(eval_cfg.get("initial_phase_ratio", sys_cfg.get("initial_phase_ratio", 0.8)))
    group_name = sys_cfg.get("group_name", None)
    experiment_name = sys_cfg.get("experiment_name", "dyna_q_experiment")
    save_env_in_exp = str(sys_cfg.get("save_env_in_experiment_dir", "true")).lower() == "true"

    all_seed_episodes: Dict[int, List[Dict[str, Any]]] = {}
    optimal_steps_dict: Dict[int, int] = {}

    for seed_idx, seed in enumerate(seeds, start=1):
        grid = create_maze(generator, height, width, seed)
        start_pos, goal_pos, max_geodesic, optimal_path = compute_maximum_geodesic_distance(grid)
        optimal_steps_dict[seed] = max_geodesic
        all_seed_episodes[seed] = []

        # Collect open cells for Q-table display
        open_cells = [
            (r, c) for r in range(height) for c in range(width)
            if grid[r][c] == 0
        ]
        open_cells.sort(key=lambda coord: (coord[0], coord[1]))

        # Initialize recently updated order
        recently_updated_order = open_cells[:]
        last_updated_coord = None

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

        for ep in range(1, episodes + 1):
            state, info = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            collisions = 0
            running = True

            while running:
                # -------------------------------------------------------------
                # EVENT HANDLING (Mouse clicks, Dropdown, Buttons, Scroll)
                # -------------------------------------------------------------
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left Click
                            if dropdown_open:
                                if dropdown_opt1_rect.collidepoint(event.pos):
                                    sort_mode = "state_asc"
                                    dropdown_open = False
                                elif dropdown_opt2_rect.collidepoint(event.pos):
                                    sort_mode = "recently_updated"
                                    dropdown_open = False
                                    q_scroll_offset = 0
                                else:
                                    dropdown_open = False
                            elif dropdown_box_rect.collidepoint(event.pos):
                                dropdown_open = not dropdown_open
                            elif btn_start_pause_rect.collidepoint(event.pos):
                                paused = not paused

                        elif event.button == 4:  # Wheel Up
                            q_scroll_offset = max(0, q_scroll_offset - 2)
                        elif event.button == 5:  # Wheel Down
                            q_scroll_offset += 2

                    elif event.type == pygame.MOUSEWHEEL:
                        q_scroll_offset = max(0, q_scroll_offset - event.y * 2)

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_UP:
                            fps = min(120, fps + 10)
                        elif event.key == pygame.K_DOWN:
                            fps = max(5, fps - 10)
                        elif event.key == pygame.K_PAGEUP:
                            q_scroll_offset = max(0, q_scroll_offset - 10)
                        elif event.key == pygame.K_PAGEDOWN:
                            q_scroll_offset += 10

                if paused:
                    screen.fill(COLOR_BG)
                    draw_board()

                    ar, ac = env.agent_pos
                    agent_center = (ac * CELL_SIZE + CELL_SIZE // 2, ar * CELL_SIZE + CELL_SIZE // 2)
                    agent_radius = max(3, CELL_SIZE // 3)
                    pygame.draw.circle(screen, COLOR_AGENT, agent_center, agent_radius)

                    draw_sidebar_hud(
                        ep=ep,
                        steps=steps,
                        total_reward=total_reward,
                        step_info=info if steps == 0 else step_info,
                        agent_instance=agent,
                        open_cells_list=open_cells,
                        current_agent_pos=env.agent_pos,
                    )
                    pygame.display.flip()
                    clock.tick(15)
                    continue

                # -------------------------------------------------------------
                # AGENT DECISION & ENVIRONMENT STEP
                # -------------------------------------------------------------
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)

                if step_info.get("collision", False):
                    collisions += 1

                if "valence" in step_info:
                    agent.update_affect(step_info["valence"], step_info["arousal"])

                # Track updated state for Q-table sorting and highlight
                updated_coord = env.state_to_coord(state)
                last_updated_coord = updated_coord

                if updated_coord in recently_updated_order:
                    recently_updated_order.remove(updated_coord)
                recently_updated_order.insert(0, updated_coord)

                # Auto-keep top in view if in recently_updated mode
                if sort_mode == "recently_updated" and q_scroll_offset == 0:
                    q_scroll_offset = 0

                state = next_state
                total_reward += reward
                steps += 1

                collision_occurred = step_info.get("collision", False)

                # -------------------------------------------------------------
                # ANIMATION: Collision on Gridworld Only (Steady sidebar)
                # -------------------------------------------------------------
                if collision_occurred:
                    dr, dc = BaseGridWorld.ACTIONS[action]
                    target_r = env.agent_pos[0] + dr
                    target_c = env.agent_pos[1] + dc
                    hit_pos = (
                        (target_r, target_c)
                        if (0 <= target_r < height and 0 <= target_c < width and grid[target_r][target_c] == 1)
                        else None
                    )

                    collision_subframes = [
                        {"jolt": 0.35, "color": (255, 69, 58), "shock_r": 3, "flash_wall": True},
                        {"jolt": 0.20, "color": (255, 149, 0), "shock_r": 8, "flash_wall": True},
                        {"jolt": -0.10, "color": (255, 180, 80), "shock_r": 13, "flash_wall": False},
                        {"jolt": 0.0, "color": COLOR_AGENT, "shock_r": 0, "flash_wall": False},
                    ]

                    for frame_data in collision_subframes:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit(0)
                            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                                if btn_start_pause_rect.collidepoint(event.pos):
                                    paused = not paused

                        screen.fill(COLOR_BG)
                        draw_board(hit_pos=hit_pos if frame_data["flash_wall"] else None)

                        # Agent jolt offset calculation
                        jolt_px = frame_data["jolt"] * CELL_SIZE
                        ar, ac = env.agent_pos
                        center_x = ac * CELL_SIZE + CELL_SIZE // 2 + int(dc * jolt_px)
                        center_y = ar * CELL_SIZE + CELL_SIZE // 2 + int(dr * jolt_px)
                        agent_radius = max(3, CELL_SIZE // 3)

                        # Contact shockwave ring on grid
                        if frame_data["shock_r"] > 0:
                            contact_x = ac * CELL_SIZE + CELL_SIZE // 2 + int(dc * (CELL_SIZE // 2))
                            contact_y = ar * CELL_SIZE + CELL_SIZE // 2 + int(dr * (CELL_SIZE // 2))
                            pygame.draw.circle(
                                screen,
                                (255, 75, 75),
                                (contact_x, contact_y),
                                frame_data["shock_r"],
                                2,
                            )

                        pygame.draw.circle(screen, frame_data["color"], (center_x, center_y), agent_radius)

                        # Draw sidebar HUD steadily
                        draw_sidebar_hud(
                            ep=ep,
                            steps=steps,
                            total_reward=total_reward,
                            step_info=step_info,
                            agent_instance=agent,
                            open_cells_list=open_cells,
                            current_agent_pos=env.agent_pos,
                        )

                        pygame.display.flip()
                        clock.tick(max(30, fps * 2))

                # Standard Frame Rendering
                screen.fill(COLOR_BG)
                draw_board()

                # Draw Agent normally
                ar, ac = env.agent_pos
                agent_center = (ac * CELL_SIZE + CELL_SIZE // 2, ar * CELL_SIZE + CELL_SIZE // 2)
                agent_radius = max(3, CELL_SIZE // 3)
                pygame.draw.circle(screen, COLOR_AGENT, agent_center, agent_radius)

                # Draw Sidebar HUD and Q-table
                draw_sidebar_hud(
                    ep=ep,
                    steps=steps,
                    total_reward=total_reward,
                    step_info=step_info,
                    agent_instance=agent,
                    open_cells_list=open_cells,
                    current_agent_pos=env.agent_pos,
                )

                pygame.display.flip()
                clock.tick(fps)

                if terminated or truncated:
                    running = False

            # Decay parameters at the end of episode
            agent.decay_episode(ep + 1)

            all_seed_episodes[seed].append({
                "episode": ep,
                "steps": steps,
                "reward": round(total_reward, 2),
                "collisions": collisions,
                "status": "GOAL REACHED" if not truncated else "TRUNCATED",
            })

        agent.reset()

    if all_seed_episodes and any(len(v) > 0 for v in all_seed_episodes.values()):
        export_evaluation_suite(
            seed_episodes=all_seed_episodes,
            optimal_steps_dict=optimal_steps_dict,
            experiment_name=experiment_name,
            reports_dir=reports_dir,
            initial_phase_ratio=initial_phase_ratio,
            group_name=group_name,
        )

    pygame.quit()
