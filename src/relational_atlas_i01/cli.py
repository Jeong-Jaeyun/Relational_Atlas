"""Dispatch an experiment by its historical module name."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys


GROUPS = ("geometry", "phase_reference", "holdout")


def experiments() -> dict[str, str]:
    """Map stable experiment names to importable modules."""
    package = importlib.import_module("relational_atlas_i01")
    available = {}
    for group in GROUPS:
        subpackage = importlib.import_module(f"{package.__name__}.{group}")
        for item in pkgutil.iter_modules(subpackage.__path__):
            if item.name.startswith("_"):
                continue
            if item.name in available:
                raise RuntimeError(f"duplicate experiment name: {item.name}")
            available[item.name] = f"{subpackage.__name__}.{item.name}"
    return available


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Run an I01 experiment; remaining options go to that experiment.",
        add_help=False,
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("experiment", nargs="?")
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        print("Use --list to see experiment names; use NAME --help for its options.")
        return 0
    args, remaining = parser.parse_known_args(argv)
    available = experiments()
    if args.list:
        print("\n".join(sorted(available)))
        return 0
    if not args.experiment or args.experiment not in available:
        parser.error("choose an experiment from --list")
    module = importlib.import_module(available[args.experiment])
    if not hasattr(module, "main"):
        raise RuntimeError(f"{args.experiment} has no main() entry point")
    original_argv = sys.argv
    try:
        sys.argv = [args.experiment, *remaining]
        module.main()
    finally:
        sys.argv = original_argv
    return 0
