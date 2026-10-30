"""Maze validation and geometric analysis utilities.

This module provides functions for path verification, geodesic distance calculation,
connected components extraction, and Bresenham's line-of-sight determination.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple


def bresenham_line(
    start: Tuple[int, int], end: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Generate grid coordinates along a line using Bresenham's algorithm.

    Args:
        start (Tuple[int, int]): Starting coordinates (row, col).
        end (Tuple[int, int]): Ending coordinates (row, col).

    Returns:
        List[Tuple[int, int]]: Ordered list of integer coordinates forming the line.

    Example:
        >>> bresenham_line((0, 0), (2, 2))
        [(0, 0), (1, 1), (2, 2)]
    """
    r0, c0 = start
    r1, c1 = end

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    points: List[Tuple[int, int]] = []
    curr_r, curr_c = r0, c0

    while True:
        points.append((curr_r, curr_c))
        if curr_r == r1 and curr_c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            curr_r += sr
        if e2 < dr:
            err += dr
            curr_c += sc

    return points


def has_line_of_sight(
    grid: List[List[int]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    wall_value: int = 1,
) -> bool:
    """Check if there is an unblocked line of sight between two grid positions.

    Uses Bresenham's line algorithm to trace points between start and end.
    Intermediate wall cells block the line of sight.

    Args:
        grid (List[List[int]]): 2D grid matrix where wall_value represents obstacles.
        start (Tuple[int, int]): Origin coordinates (row, col).
        end (Tuple[int, int]): Target coordinates (row, col).
        wall_value (int, optional): Integer representing a wall obstacle. Defaults to 1.

    Returns:
        bool: True if there is a clear line of sight, False if blocked by a wall.
    """
    points = bresenham_line(start, end)
    # Check intermediate points between start and end
    for pt in points[1:-1]:
        r, c = pt
        if grid[r][c] == wall_value:
            return False
    return True


def get_open_cells(grid: List[List[int]], wall_value: int = 1) -> Set[Tuple[int, int]]:
    """Retrieve all open (non-wall) coordinates from the grid.

    Args:
        grid (List[List[int]]): 2D grid matrix.
        wall_value (int, optional): Value representing obstacle/wall. Defaults to 1.

    Returns:
        Set[Tuple[int, int]]: Set of (row, col) tuples corresponding to open cells.
    """
    open_cells: Set[Tuple[int, int]] = set()
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    for r in range(height):
        for c in range(width):
            if grid[r][c] != wall_value:
                open_cells.add((r, c))
    return open_cells


def find_connected_components(
    grid: List[List[int]], wall_value: int = 1
) -> List[Set[Tuple[int, int]]]:
    """Identify all 4-connected components of open cells in the grid.

    Args:
        grid (List[List[int]]): 2D grid matrix.
        wall_value (int, optional): Obstacle identifier value. Defaults to 1.

    Returns:
        List[Set[Tuple[int, int]]]: List of connected component sets sorted by size descending.
    """
    open_cells = get_open_cells(grid, wall_value)
    visited: Set[Tuple[int, int]] = set()
    components: List[Set[Tuple[int, int]]] = []

    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    for cell in open_cells:
        if cell in visited:
            continue

        component: Set[Tuple[int, int]] = set()
        queue = deque([cell])
        visited.add(cell)

        while queue:
            curr_r, curr_c = queue.popleft()
            component.add((curr_r, curr_c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if grid[nr][nc] != wall_value and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        components.append(component)

    components.sort(key=len, reverse=True)
    return components


def bfs_shortest_distances(
    start: Tuple[int, int],
    valid_cells: Set[Tuple[int, int]],
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Tuple[int, int]]]:
    """Compute shortest distances and predecessors from start cell to all reachable cells.

    Args:
        start (Tuple[int, int]): Origin cell (row, col).
        valid_cells (Set[Tuple[int, int]]): Set of traversable cells.

    Returns:
        Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Tuple[int, int]]]:
            A tuple of (distances_dict, predecessors_dict).
    """
    distances: Dict[Tuple[int, int], int] = {start: 0}
    predecessors: Dict[Tuple[int, int], Tuple[int, int]] = {}
    queue = deque([start])

    while queue:
        curr = queue.popleft()
        curr_dist = distances[curr]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (curr[0] + dr, curr[1] + dc)
            if neighbor in valid_cells and neighbor not in distances:
                distances[neighbor] = curr_dist + 1
                predecessors[neighbor] = curr
                queue.append(neighbor)

    return distances, predecessors


def reconstruct_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    predecessors: Dict[Tuple[int, int], Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Reconstruct coordinate path from start to goal using predecessor map.

    Args:
        start (Tuple[int, int]): Origin coordinate.
        goal (Tuple[int, int]): Goal coordinate.
        predecessors (Dict[Tuple[int, int], Tuple[int, int]]): Predecessor mapping from BFS.

    Returns:
        List[Tuple[int, int]]: Complete path from start to goal inclusive.
    """
    path = [goal]
    curr = goal
    while curr != start:
        curr = predecessors[curr]
        path.append(curr)
    path.reverse()
    return path


def compute_maximum_geodesic_distance(
    grid: List[List[int]],
    wall_value: int = 1,
) -> Tuple[Tuple[int, int], Tuple[int, int], int, List[Tuple[int, int]]]:
    """Find the pair of open cells with the maximum geodesic distance.

    Searches within the largest connected component of traversable cells using
    all-pairs shortest paths via BFS. This guarantees start and goal are as far
    apart as possible along the maze paths and that a valid path exists.

    Args:
        grid (List[List[int]]): 2D grid where wall_value represents obstacles.
        wall_value (int, optional): Obstacle cell value. Defaults to 1.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int], int, List[Tuple[int, int]]]:
            (start_coord, goal_coord, max_geodesic_distance, shortest_path).

    Raises:
        ValueError: If no open connected component with at least 2 cells is found.
    """
    components = find_connected_components(grid, wall_value)
    if not components or len(components[0]) < 2:
        raise ValueError("Grid does not contain a connected component of at least 2 open cells.")

    largest_component = components[0]
    max_distance = -1
    best_start: Optional[Tuple[int, int]] = None
    best_goal: Optional[Tuple[int, int]] = None
    best_predecessors: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None

    # Iterate over all cells in largest component to find global diameter
    for node in largest_component:
        distances, predecessors = bfs_shortest_distances(node, largest_component)
        for target, dist in distances.items():
            if dist > max_distance:
                max_distance = dist
                best_start = node
                best_goal = target
                best_predecessors = predecessors

    if best_start is None or best_goal is None or best_predecessors is None:
        raise ValueError("Failed to determine maximum geodesic distance.")

    path = reconstruct_path(best_start, best_goal, best_predecessors)
    return best_start, best_goal, max_distance, path

