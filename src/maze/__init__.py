"""Maze generation and validation package.

Provides algorithms for maze synthesis (Randomized Kruskal, Cellular Automata)
and geometric validation (geodesic distance, connectivity, Bresenham line of sight).
"""

from src.maze.cellular_automata import generate_ca_maze
from src.maze.kruskal import DisjointSetUnion, generate_kruskal_maze
from src.maze.validation import (
    bfs_shortest_distances,
    bresenham_line,
    compute_maximum_geodesic_distance,
    find_connected_components,
    get_open_cells,
    has_line_of_sight,
    reconstruct_path,
)

__all__ = [
    "generate_kruskal_maze",
    "generate_ca_maze",
    "DisjointSetUnion",
    "bresenham_line",
    "has_line_of_sight",
    "get_open_cells",
    "find_connected_components",
    "bfs_shortest_distances",
    "reconstruct_path",
    "compute_maximum_geodesic_distance",
]

