"""Main entry point for the Affective Dyna-Q GridWorld simulator.

Loads configuration parameters from .cfg files in settings/, initializes
the environment and agent, runs independent multi-seed experiments, and
saves visual outputs and metric reports. Supports both single-configuration
runs and batch sequential execution of all configurations via --all, with
an optional --exclude flag to skip specific configuration files.
"""

import argparse
import configparser
import os
import sys
from typing import Any, Dict, List


def parse_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse key-value configuration file with support for sections and lists.

    Args:
        config_path (str): File path to .cfg configuration file.

    Returns:
        Dict[str, Dict[str, Any]]: Nested dictionary organized by sections.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(config_path, encoding="utf-8")

    config: Dict[str, Dict[str, Any]] = {}
    for section in parser.sections():
        config[section] = {}
        for key, value in parser.items(section):
            # Clean string and strip quotes or brackets
            val_clean = value.strip().strip("'\"")

            # Parse seeds list if formatted as [1, 3, 5, 6]
            if key == "seeds" and val_clean.startswith("[") and val_clean.endswith("]"):
                inner = val_clean[1:-1].strip()
                config[section][key] = [int(s.strip()) for s in inner.split(",") if s.strip()]
            else:
                config[section][key] = val_clean

    return config


def discover_config_files(target_path: str = "settings") -> List[str]:
    """Recursively discover all .cfg files in target_path, sorted alphabetically.

    Args:
        target_path (str): File path or directory path.

    Returns:
        List[str]: List of sorted .cfg file paths.
    """
    if os.path.isfile(target_path):
        return [target_path] if target_path.endswith(".cfg") else []

    if not os.path.isdir(target_path):
        return []

    cfg_files = []
    for root, _, files in os.walk(target_path):
        for f in sorted(files):
            if f.endswith(".cfg") and not f.startswith("."):
                cfg_files.append(os.path.join(root, f))
    return sorted(cfg_files)


def is_file_excluded(file_path: str, exclude_patterns: List[str]) -> bool:
    """Check if a file path matches any exclusion pattern or explicit file name.

    Matches by:
    - Exact absolute path
    - Normalized relative path (e.g. 'settings/default.cfg')
    - Filename with extension (e.g. 'default.cfg')
    - Filename without extension (e.g. 'default')

    Args:
        file_path (str): Path of configuration file to check.
        exclude_patterns (List[str]): List of exclusion strings/paths.

    Returns:
        bool: True if file should be excluded, False otherwise.
    """
    if not exclude_patterns:
        return False

    abs_file = os.path.abspath(file_path)
    norm_file = os.path.normpath(file_path)
    base_file = os.path.basename(file_path)
    stem_file = os.path.splitext(base_file)[0]

    for item in exclude_patterns:
        # Split by comma if passed as comma-separated string
        for sub_item in item.split(","):
            pattern = sub_item.strip()
            if not pattern:
                continue

            abs_pat = os.path.abspath(pattern)
            norm_pat = os.path.normpath(pattern)
            base_pat = os.path.basename(pattern)
            stem_pat = os.path.splitext(base_pat)[0]

            if abs_file == abs_pat:
                return True
            if norm_file == norm_pat or norm_file.endswith(os.sep + norm_pat):
                return True
            if base_file == base_pat:
                return True
            if stem_file == stem_pat:
                return True
    return False


def run_single_config(config_path: str, mode: str) -> bool:
    """Run simulation for a single configuration file.

    Args:
        config_path (str): Path to .cfg file.
        mode (str): Execution mode ('headless' or 'gui').

    Returns:
        bool: True if execution succeeded, False otherwise.
    """
    print(f"\nLoading configuration from: {config_path}")
    try:
        config = parse_config(config_path)
    except Exception as e:
        print(f"Error parsing configuration file '{config_path}': {e}")
        return False

    try:
        if mode == "gui":
            from src.ui.gui import run_gui_experiment
            run_gui_experiment(config)
        else:
            from src.ui.headless import run_headless_experiment
            run_headless_experiment(config)
        return True
    except Exception as e:
        print(f"Error executing experiment for '{config_path}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main execution function.

    Parses command-line arguments and launches the simulation in either GUI
    or headless mode, with support for single file or batch (--all) execution,
    and optional exclusion (--exclude) of specific configuration files.

    CLI Usage:
        - Run with GUI (default for single config):
            python3 main.py settings/nama_file.cfg
        - Run in headless (CLI terminal) mode:
            python3 main.py --headless settings/nama_file.cfg
        - Run all configuration files in settings/ sequentially:
            python3 main.py --all
            python3 main.py --headless --all
        - Run all configuration files while excluding specific files:
            python3 main.py --all --exclude settings/default.cfg
            python3 main.py --all -e default.cfg kruskal_static.cfg
        - Run all configuration files in a specific sub-folder:
            python3 main.py --all settings/exponential_decay
        - Default (loads settings/default.cfg in GUI):
            python3 main.py
    """
    arg_parser = argparse.ArgumentParser(
        description="Affective Dyna-Q GridWorld Reinforcement Learning Simulator"
    )
    arg_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to .cfg configuration file or directory (default: settings/default.cfg)",
    )
    arg_parser.add_argument(
        "--all",
        action="store_true",
        help="Execute all configuration files (.cfg) in settings/ (or target directory) sequentially one by one",
    )
    arg_parser.add_argument(
        "-e",
        "--exclude",
        nargs="+",
        default=[],
        help="Exclude one or more configuration files from execution (e.g. --exclude settings/default.cfg or -e default.cfg file2.cfg)",
    )
    arg_parser.add_argument(
        "--headless",
        action="store_true",
        help="Execute in headless CLI mode (terminal benchmark without GUI)",
    )
    arg_parser.add_argument(
        "--gui",
        action="store_true",
        help="Execute in graphical user interface mode (Pygame viewer)",
    )
    arg_parser.add_argument(
        "--config",
        dest="config_flag",
        type=str,
        default=None,
        help="Alternative flag to specify .cfg file or directory path",
    )
    args = arg_parser.parse_args()

    # Determine execution mode:
    # 1. If --headless flag is passed: headless mode
    # 2. If --gui flag is passed: GUI mode
    # 3. If --all is passed without --gui: default to headless for batch runs
    # 4. If config file is passed as positional argument without --headless: GUI mode
    # 5. Default single run without flags: GUI mode
    if args.headless:
        mode = "headless"
    elif args.gui:
        mode = "gui"
    elif args.all:
        mode = "headless"
    elif args.config or args.config_flag:
        mode = "gui"
    else:
        mode = "gui"

    # --------------------------------------------------------------------------
    # BATCH EXECUTION: Execute all .cfg files sequentially (--all)
    # --------------------------------------------------------------------------
    if args.all:
        target_path = args.config or args.config_flag or "settings"
        if os.path.isfile(target_path):
            target_path = os.path.dirname(target_path) or "settings"

        discovered = discover_config_files(target_path)

        if not discovered:
            print(f"Error: No .cfg configuration files found in '{target_path}'.")
            sys.exit(1)

        # Apply exclusion filter if --exclude is specified
        excluded_files = [f for f in discovered if is_file_excluded(f, args.exclude)]
        cfg_files = [f for f in discovered if not is_file_excluded(f, args.exclude)]

        print("=" * 75)
        print(f"BATCH EXECUTION (--all): Found {len(discovered)} file(s) | Excluded: {len(excluded_files)} | To Run: {len(cfg_files)}")
        print(f"Target Directory: {target_path}")
        print(f"Execution Mode:   {mode.upper()}")
        if excluded_files:
            print("-" * 75)
            print(f"Excluded File(s) via --exclude ({len(excluded_files)}):")
            for ex in excluded_files:
                print(f"  [–] {ex}")
        print("=" * 75)

        if not cfg_files:
            print("Notice: All configuration files were excluded. Nothing to execute.")
            sys.exit(0)

        print(f"Execution Queue ({len(cfg_files)} file(s)):")
        for i, cfg_f in enumerate(cfg_files, 1):
            print(f"  {i}. {cfg_f}")
        print("=" * 75)

        results: Dict[str, bool] = {}
        for idx, cfg_f in enumerate(cfg_files, 1):
            print("\n" + "#" * 75)
            print(f"[{idx}/{len(cfg_files)}] EXECUTING: {cfg_f}")
            print("#" * 75)

            success = run_single_config(cfg_f, mode=mode)
            results[cfg_f] = success

        # Print batch execution summary
        succeeded = [f for f, s in results.items() if s]
        failed = [f for f, s in results.items() if not s]

        print("\n" + "=" * 75)
        print("BATCH EXECUTION SUMMARY (--all)")
        print(f"Total: {len(cfg_files)} | Succeeded: {len(succeeded)} | Failed: {len(failed)}")
        print("=" * 75)
        for f in succeeded:
            print(f"  [✓] {f}")
        for f in failed:
            print(f"  [✗] {f} (FAILED)")
        if excluded_files:
            for f in excluded_files:
                print(f"  [–] {f} (EXCLUDED)")
        print("=" * 75)

        if failed:
            sys.exit(1)
        return

    # --------------------------------------------------------------------------
    # SINGLE EXECUTION: Execute a single configuration file
    # --------------------------------------------------------------------------
    config_path = args.config or args.config_flag or "settings/default.cfg"

    if os.path.isdir(config_path):
        print(f"Notice: '{config_path}' is a directory.")
        print(f"To run all configuration files in this directory, use: python3 main.py --all {config_path}")
        sys.exit(1)

    # Check if single file is excluded
    if is_file_excluded(config_path, args.exclude):
        print(f"Notice: Configuration file '{config_path}' is excluded by --exclude. Skipping execution.")
        sys.exit(0)

    # Fallback to [system] mode if neither --headless, --gui, nor explicit config was specified
    if not (args.headless or args.gui or args.config or args.config_flag):
        try:
            cfg_peek = parse_config(config_path)
            mode = cfg_peek.get("system", {}).get("mode", "gui").strip().lower()
        except Exception:
            mode = "gui"

    success = run_single_config(config_path, mode=mode)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
