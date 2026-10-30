"""User interface package for simulation execution and visual rendering.

Provides headless execution, Pygame GUI viewer, and publication-ready figure exports.
"""

from src.ui.headless import run_headless_experiment
from src.ui.renderer import render_ascii_grid, resolve_env_image_path, save_environment_image

__all__ = [
    "run_headless_experiment",
    "save_environment_image",
    "resolve_env_image_path",
    "render_ascii_grid",
]

