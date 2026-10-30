"""Randomized Kruskal's Algorithm for maze generation.

This module generates perfect mazes (spanning trees) or braided mazes using
Randomized Kruskal's algorithm with Disjoint Set Union (DSU) and seed reproducibility.
"""

import random
from typing import Dict, List, Optional, Set, Tuple


class DisjointSetUnion:
    """Disjoint Set Union (DSU) / Union-Find data structure with path compression."""

    def __init__(self) -> None:
        """Initialize empty DSU structure."""
        self.parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.rank: Dict[Tuple[int, int], int] = {}

    def find(self, item: Tuple[int, int]) -> Tuple[int, int]:
        """Find the representative element of the set containing item.

        Args:
            item (Tuple[int, int]): Coordinate tuple.

        Returns:
            Tuple[int, int]: Root representative coordinate.
        """
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Union the sets containing items a and b.

        Args:
            a (Tuple[int, int]): First item.
            b (Tuple[int, int]): Second item.

        Returns:
            bool: True if sets were merged, False if already in the same set.
        """
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
        return True


def generate_kruskal_maze(
    height: int,
    width: int,
    seed: Optional[int] = None,
    braid_ratio: float = 0.0,
    wall_value: int = 1,
    path_value: int = 0,
) -> List[List[int]]:
    """Generate a maze using Randomized Kruskal's algorithm.

    Creates a 2D grid filled with walls and carves passages connecting rooms
    in a spanning tree topology. Optional braiding can remove dead ends to
    create multiple alternative paths.

    Args:
        height (int): Grid height (number of rows). Must be >= 3.
        width (int): Grid width (number of columns). Must be >= 3.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        braid_ratio (float, optional): Ratio (0.0 to 1.0) of extra openings to carve
            to remove dead-ends and introduce loops. Defaults to 0.0.
        wall_value (int, optional): Cell value representing walls. Defaults to 1.
        path_value (int, optional): Cell value representing open paths. Defaults to 0.

    Returns:
        List[List[int]]: 2D grid matrix where 0 represents path and 1 represents wall.

    Raises:
        ValueError: If height < 3 or width < 3.
    """
    if height < 3 or width < 3:
        raise ValueError(f"Grid dimensions must be at least 3x3, got ({height}, {width}).")

    rng = random.Random(seed)

    # Ensure effective grid has odd dimensions for traditional cell-and-wall lattice
    eff_height = height if height % 2 == 1 else height - 1
    eff_width = width if width % 2 == 1 else width - 1

    grid = [[wall_value for _ in range(width)] for _ in range(height)]

    # Rooms are located at odd row and column indices
    rooms: List[Tuple[int, int]] = []
    for r in range(1, eff_height, 2):
        for c in range(1, eff_width, 2):
            rooms.append((r, c))
            grid[r][c] = path_value

    # Build list of walls separating adjacent rooms
    # Each wall entry: (wall_coord, room1, room2)
    walls: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
    for r, c in rooms:
        # Check right neighbor
        if c + 2 < eff_width:
            walls.append(((r, c + 1), (r, c), (r, c + 2)))
        # Check bottom neighbor
        if r + 2 < eff_height:
            walls.append(((r + 1, c), (r, c), (r + 2, c)))

    rng.shuffle(walls)
    dsu = DisjointSetUnion()
    remaining_walls: List[Tuple[int, int]] = []

    for wall_pos, room1, room2 in walls:
        if dsu.union(room1, room2):
            wr, wc = wall_pos
            grid[wr][wc] = path_value
        else:
            remaining_walls.append(wall_pos)

    # Optional braiding to introduce loops and multiple paths
    if braid_ratio > 0.0 and remaining_walls:
        rng.shuffle(remaining_walls)
        extra_count = int(len(remaining_walls) * max(0.0, min(1.0, braid_ratio)))
        for wr, wc in remaining_walls[:extra_count]:
            grid[wr][wc] = path_value

    # Handle padding rows/cols if original height/width was even
    if height > eff_height:
        for c in range(1, width - 1):
            if grid[eff_height - 1][c] == path_value and rng.random() < 0.5:
                grid[eff_height][c] = path_value
    if width > eff_width:
        for r in range(1, height - 1):
            if grid[r][eff_width - 1] == path_value and rng.random() < 0.5:
                grid[r][eff_width] = path_value

    return grid

