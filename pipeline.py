"""
Entry point for the relative velocity statistics pipeline.

Usage:
    python pipeline.py                   # full pipeline (calc + plot)
    python pipeline.py --calc-only       # calculation only
    python pipeline.py --plot-only       # plotting only (requires existing results)
    python pipeline.py --generate-test   # generate test data, then run full pipeline
    python pipeline.py --validate        # generate test data, run pipeline, overlay Maxwell PDFs
"""

import argparse
import sys

from config import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Relative velocity statistics of galaxy pairs from SAM outputs."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--calc-only",
        action="store_true",
        help="Run calculation only; skip plotting.",
    )
    group.add_argument(
        "--plot-only",
        action="store_true",
        help="Make plots from existing results; skip calculation.",
    )
    group.add_argument(
        "--generate-test",
        action="store_true",
        help="Generate test data, then run full pipeline.",
    )
    group.add_argument(
        "--validate",
        action="store_true",
        help="Generate test data, run pipeline, overlay Maxwell PDFs on plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.validate or args.generate_test:
        print("Generating test data...")
        from generate_test_data import generate_all_snapshots
        generate_all_snapshots(config)

    if not args.plot_only:
        print("Running pair-finding calculation...")
        from calc import run_calculation
        run_calculation(config)

    if not args.calc_only:
        print("Making plots...")
        from plot import make_plots
        validation_mode = args.validate
        make_plots(config, validation_mode=validation_mode)

    print("Done.")


if __name__ == "__main__":
    main()
