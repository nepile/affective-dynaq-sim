"""Cellular Automata based maze and cave generator.

Generates organic, cave-like environments with natural obstacle clusters using 2D
cellular automata without artificial perimeter wall enclosures.
"""

import random
from typing import List, Optional
from src.maze.validation import find_connected_components


def count_wall_neighbors_inbounds(
    grid: List[List[int]],
    r: int,
    c: int,
    wall_value: int = 1,
) -> int:
    """Count obstacle neighbors in the 8-Moore neighborhood within grid boundaries.

    Only in-bounds neighbors are counted to prevent artificial border clustering.

    Args:
        grid (List[List[int]]): 2D grid matrix.
        r (int): Row coordinate.
        c (int): Column coordinate.
        wall_value (int, optional): Cell value for obstacles. Defaults to 1.

    Returns:
        int: Number of neighboring wall cells.
    """
    height = len(grid)
    width = len(grid[0])
    count = 0

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if grid[nr][nc] == wall_value:
                    count += 1
    return count


def generate_ca_maze(
    height: int,
    width: int,
    seed: Optional[int] = None,
    initial_wall_prob: float = 0.42,
    iterations: int = 2,
    survive_thresh: int = 3,
    birth_thresh: int = 5,
    wall_value: int = 1,
    path_value: int = 0,
    fill_small_components: bool = True,
) -> List[List[int]]:
    """Generate an organic cave/maze with obstacle clusters using Cellular Automata.

    Initializes random noise across the full grid without forced border walls,
    then applies birth/survival cellular automata rules iteratively to generate
    natural organic obstacle clusters and corridors. Disconnected open pockets
    are filled into obstacles to guarantee a unified navigable open space.

    Args:
        height (int): Grid height (number of rows).
        width (int): Grid width (number of columns).
        seed (Optional[int], optional): Random seed. Defaults to None.
        initial_wall_prob (float, optional): Initial random obstacle density. Defaults to 0.42.
        iterations (int, optional): Smoothing iterations. Defaults to 2.
        survive_thresh (int, optional): Min wall neighbors for wall survival. Defaults to 3.
        birth_thresh (int, optional): Min wall neighbors to create wall from path. Defaults to 5.
        wall_value (int, optional): Value representing obstacle. Defaults to 1.
        path_value (int, optional): Value representing free path. Defaults to 0.
        fill_small_components (bool, optional): Whether to fill isolated open pockets. Defaults to True.

    Returns:
        List[List[int]]: 2D grid matrix of the generated maze.

    Raises:
        ValueError: If dimensions < 3 or probability outside [0, 1].
    """
    if height < 3 or width < 3:
        raise ValueError(f"Grid dimensions must be at least 3x3, got ({height}, {width}).")
    if not (0.0 <= initial_wall_prob <= 1.0):
        raise ValueError(f"initial_wall_prob must be between 0 and 1, got {initial_wall_prob}.")

    rng = random.Random(seed)

    # 1. Initialize random grid without artificial border walls
    grid = [
        [wall_value if rng.random() < initial_wall_prob else path_value for _ in range(width)]
        for _ in range(height)
    ]

    # 2. Cellular Automata iterative updates
    for _ in range(iterations):
        new_grid = [[path_value for _ in range(width)] for _ in range(height)]
        for r in range(height):
            for c in range(width):
                wall_neighbors = count_wall_neighbors_inbounds(grid, r, c, wall_value)
                if grid[r][c] == wall_value:
                    new_grid[r][c] = wall_value if wall_neighbors >= survive_thresh else path_value
                else:
                    new_grid[r][c] = wall_value if wall_neighbors >= birth_thresh else path_value
        grid = new_grid

    # 3. Post-processing: keep largest open connected component
    if fill_small_components:
        components = find_connected_components(grid, wall_value)
        if components:
            largest = components[0]
            for r in range(height):
                for c in range(width):
                    if (r, c) not in largest:
                        grid[r][c] = wall_value

    return grid
