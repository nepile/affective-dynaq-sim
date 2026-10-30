"""Environment rendering and image export utilities.

Provides functions to export publication-quality figures of generated gridworlds
with obstacles, start, and goal positions, formatted for research papers.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from typing import Any, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


def _normalize_color(color: Any, default: Any) -> Any:
    """Normalize color input from tuple (0-255), hex string, or float list for Matplotlib.

    Args:
        color (Any): Input color specification.
        default (Any): Fallback default color.

    Returns:
        Any: Matplotlib-compatible color format.
    """
    if color is None:
        return default
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        if any(isinstance(c, int) and c > 1 for c in color[:3]):
            return [c / 255.0 for c in color[:3]]
        return list(color[:3])
    return color


def resolve_env_image_path(
    reports_dir: str,
    experiment_name: str,
    generator_name: str,
    seed: int,
    save_in_experiment_dir: bool = True,
    group_name: Optional[str] = None,
) -> str:
    """Resolve destination path for environment visualization image.

    Args:
        reports_dir (str): Root reports directory.
        experiment_name (str): Experiment name from configuration.
        generator_name (str): Maze generator name (e.g. 'kruskal', 'cellular_automata').
        seed (int): Environment random seed.
        save_in_experiment_dir (bool, optional): If True, saves in
            reports/[group_name/]<experiment_name>/environment/seed_<seed>.png.
            If False, saves in reports/gridworld/<generator>/seed_<seed>.png.
            Defaults to True.
        group_name (Optional[str], optional): Optional group name grouping experiments.

    Returns:
        str: Absolute or relative file path for the environment image.
    """
    if save_in_experiment_dir:
        if group_name and str(group_name).strip().lower() not in ("", "none"):
            return os.path.join(reports_dir, group_name.strip(), experiment_name, "environment", f"seed_{seed}.png")
        return os.path.join(reports_dir, experiment_name, "environment", f"seed_{seed}.png")
    return os.path.join(reports_dir, "gridworld", generator_name.lower(), f"seed_{seed}.png")


def save_environment_image(
    grid: List[List[int]],
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    generator_name: str,
    seed: int,
    output_dir: str = "reports",
    geodesic_distance: Optional[int] = None,
    optimal_path: Optional[List[Tuple[int, int]]] = None,
    dpi: int = 300,
    obstacle_color: Optional[Any] = None,
    grid_line_color: Optional[Any] = None,
    start_color: Optional[Any] = None,
    goal_color: Optional[Any] = None,
    save_path: Optional[str] = None,
) -> str:
    """Save a publication-quality visual image matching research paper specifications.

    Renders grid cells, obstacles, start (S), goal (G), shortest geodesic path,
    detailed subtitle metrics (obstacle percentage, distance), and a clean legend.

    Args:
        grid (List[List[int]]): 2D grid matrix (0 for path, 1 for wall).
        start_pos (Tuple[int, int]): Start coordinate (row, col).
        goal_pos (Tuple[int, int]): Goal coordinate (row, col).
        generator_name (str): Name of the obstacle generator ('kruskal', 'cellular_automata').
        seed (int): Random seed used to generate the environment.
        output_dir (str, optional): Base directory for reports. Defaults to "reports".
        geodesic_distance (Optional[int], optional): Length of maximum geodesic distance.
        optimal_path (Optional[List[Tuple[int, int]]], optional): Coordinate path for overlay.
        dpi (int, optional): Image resolution DPI. Defaults to 300.
        obstacle_color (Optional[Any], optional): Color for obstacle cells. Defaults to Dark Navy (40, 55, 71).
        grid_line_color (Optional[Any], optional): Color for grid divider lines. Defaults to (210, 210, 210).
        start_color (Optional[Any], optional): Color for start cell (S). Defaults to (46, 204, 113).
        goal_color (Optional[Any], optional): Color for goal cell (G). Defaults to (255, 140, 0).
        save_path (Optional[str], optional): Explicit destination path. If omitted, uses output_dir.

    Returns:
        str: Absolute or relative path to the generated image file.
    """
    if save_path:
        file_path = save_path
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    else:
        subfolder = os.path.join(output_dir, generator_name.lower())
        os.makedirs(subfolder, exist_ok=True)
        file_path = os.path.join(subfolder, f"seed_{seed}.png")

    height = len(grid)
    width = len(grid[0])

    # Calculate statistics
    obstacle_count = sum(grid[r][c] for r in range(height) for c in range(width))
    total_cells = height * width
    obstacle_pct = (obstacle_count / total_cells) * 100.0 if total_cells > 0 else 0.0

    # Dynamic figure size maintaining square grid aspect
    fig_dim = max(7.0, min(12.0, max(height, width) * 0.35 + 2.0))
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim + 1.2), dpi=dpi)

    # Color Palette matching GUI theme (defaults to dark navy obstacles, light gray grid)
    COLOR_OBSTACLE = _normalize_color(obstacle_color, (40, 55, 71))           # Dark navy (#283747)
    COLOR_FREE = [1.0, 1.0, 1.0]                                             # Pure white
    COLOR_GRID_LINE = _normalize_color(grid_line_color, (210, 210, 210))     # Light gray (#d2d2d2)
    COLOR_PATH_BLUE = "#3498db"                                              # Sky blue
    COLOR_START = _normalize_color(start_color, (46, 204, 113))              # Start (#2ecc71)
    COLOR_GOAL = _normalize_color(goal_color, (255, 140, 0))                 # Goal (#ff8c00)

    color_matrix = np.zeros((height, width, 3))
    for r in range(height):
        for c in range(width):
            color_matrix[r, c] = COLOR_OBSTACLE if grid[r][c] == 1 else COLOR_FREE

    ax.imshow(color_matrix, interpolation="nearest", origin="upper")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color=COLOR_GRID_LINE, linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Draw Shortest Geodesic Path if provided
    if optimal_path:
        path_r = [p[0] for p in optimal_path]
        path_c = [p[1] for p in optimal_path]
        ax.plot(
            path_c,
            path_r,
            color=COLOR_PATH_BLUE,
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=max(3, min(6, int(150 / max(height, width)))),
            markerfacecolor=COLOR_PATH_BLUE,
            markeredgecolor=COLOR_PATH_BLUE,
            alpha=0.85,
            zorder=4,
        )

    # Draw Start cell (S)
    sr, sc = start_pos
    start_patch = mpatches.Rectangle(
        (sc - 0.48, sr - 0.48), 0.96, 0.96,
        facecolor=COLOR_START, edgecolor=COLOR_START, linewidth=1.0, zorder=5
    )
    ax.add_patch(start_patch)
    font_cell = max(7, min(14, int(260 / max(height, width))))
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontsize=font_cell, fontweight="bold", zorder=6)

    # Draw Goal cell (G)
    gr, gc = goal_pos
    goal_patch = mpatches.Rectangle(
        (gc - 0.48, gr - 0.48), 0.96, 0.96,
        facecolor=COLOR_GOAL, edgecolor=COLOR_GOAL, linewidth=1.0, zorder=5
    )
    ax.add_patch(goal_patch)
    ax.text(gc, gr, "G", ha="center", va="center", color="white", fontsize=font_cell, fontweight="bold", zorder=6)

    # Axes configuration and ticks
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)  # Inverted Y so row 0 is at top

    # Set tick marks for each integer coordinate
    step_tick = 1 if max(height, width) <= 30 else 5
    ticks_x = list(range(0, width, step_tick))
    ticks_y = list(range(0, height, step_tick))
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)

    tick_font = max(6, min(10, int(180 / max(height, width))))
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.set_xlabel("Column Index", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Row Index", fontsize=11, fontweight="bold", labelpad=8)

    # Title and Subtitle formatting
    title_gen_str = generator_name.replace("_", " ").title()
    title_text = f"Dyna Maze - {title_gen_str}"
    steps_val = geodesic_distance if geodesic_distance is not None else (len(optimal_path) - 1 if optimal_path else 0)
    subtitle_text = f"Size: {height}x{width} | Obstacles: {obstacle_count} ({obstacle_pct:.1f}%) | Max Geodesic Distance: {steps_val} steps"

    ax.set_title(f"{title_text}\n{subtitle_text}", fontsize=11, fontweight="bold", pad=12)

    # Build Legend matching publication style in 2 neat rows
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_START, edgecolor=COLOR_START, label=f"Start S: ({start_pos[0]}, {start_pos[1]})"),
        mpatches.Patch(facecolor=COLOR_OBSTACLE, edgecolor=COLOR_OBSTACLE, label=f"Obstacle ({obstacle_count} cells)"),
        Line2D([0], [0], color=COLOR_PATH_BLUE, linestyle="--", marker="o", markersize=4, label=f"Shortest Path ({steps_val} steps)"),
        mpatches.Patch(facecolor=COLOR_GOAL, edgecolor=COLOR_GOAL, label=f"Goal G: ({goal_pos[0]}, {goal_pos[1]})"),
        mpatches.Patch(facecolor="#ffffff", edgecolor=COLOR_GRID_LINE, label="Free Cell"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=9,
        frameon=True,
        edgecolor="#cbd5e0",
        fancybox=False,
    )

    plt.tight_layout()
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return file_path


def render_ascii_grid(
    grid: List[List[int]],
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    agent_pos: Optional[Tuple[int, int]] = None,
) -> str:
    """Generate an ASCII string representation of the grid.

    Args:
        grid (List[List[int]]): 2D grid matrix.
        start_pos (Tuple[int, int]): Start coordinate.
        goal_pos (Tuple[int, int]): Goal coordinate.
        agent_pos (Optional[Tuple[int, int]], optional): Agent's current coordinate.

    Returns:
        str: Multi-line string representation of the grid.
    """
    height = len(grid)
    width = len(grid[0])
    lines: List[str] = []

    for r in range(height):
        row_str = ""
        for c in range(width):
            pos = (r, c)
            if agent_pos is not None and pos == agent_pos:
                row_str += " A "
            elif pos == start_pos:
                row_str += " S "
            elif pos == goal_pos:
                row_str += " G "
            elif grid[r][c] == 1:
                row_str += "###"
            else:
                row_str += " . "
        lines.append(row_str)

    return "\n".join(lines)
