#!/usr/bin/env python
"""Thin CLI dispatcher for the plethysmography package.

Usage:
    python run.py exp1                          full experiment-1 pipeline
    python run.py exp2                          full experiment-2 pipeline
    python run.py exp4                          experiment-4 stats + plots only
    python run.py all                           runs exp1, exp2, then exp4
    python run.py preprocess --experiment 1     only preprocess (no analysis)
    python run.py analyze --experiment 1        only analyze + stats + plots
    python run.py stats --experiment 1          only stats (uses cached CSV)
    python run.py plots --experiment 1          only plots (uses cached CSV)
    python run.py normalize-filenames           one-shot rename of trailing-space EDFs

Add ``-v`` / ``--verbose`` for debug-level logging. Configuration overrides:
``--config path/to/config.json`` (any frozen-dataclass field can be overridden;
unknown keys raise).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def _load_config(config_path: Optional[Path]):
    from plethysmography.core.config import PlethConfig
    if config_path is None:
        return PlethConfig()
    return PlethConfig.from_json(config_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("exp1", "exp2", "exp4", "all", "preprocess", "analyze", "stats", "plots", "normalize-filenames"),
    )
    parser.add_argument(
        "--experiment", type=int, choices=(1, 2),
        help="Required for preprocess/analyze/stats/plots commands.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "normalize-filenames":
        from plethysmography.data_loading.filename_normalize import normalize_edf_filenames
        normalize_edf_filenames(Path("Data"))
        return 0

    config = _load_config(args.config)

    if args.command in ("exp1", "all"):
        from plethysmography.pipelines import experiment1
        experiment1.run(config=config)
    if args.command in ("exp2", "all"):
        from plethysmography.pipelines import experiment2
        experiment2.run(config=config)
    if args.command in ("exp4", "all"):
        from plethysmography.pipelines import experiment4
        experiment4.run()
    if args.command in ("preprocess", "analyze", "stats", "plots"):
        if args.experiment is None:
            parser.error(f"--experiment is required for {args.command}")
        if args.experiment == 1:
            from plethysmography.pipelines import experiment1 as exp_mod
        else:
            from plethysmography.pipelines import experiment2 as exp_mod
        flags = {
            "preprocess": dict(do_preprocess=True, do_analyze=False, do_stats=False, do_plots=False),
            "analyze":    dict(do_preprocess=False, do_analyze=True, do_stats=True, do_plots=True),
            "stats":      dict(do_preprocess=False, do_analyze=False, do_stats=True, do_plots=False),
            "plots":      dict(do_preprocess=False, do_analyze=False, do_stats=False, do_plots=True),
        }[args.command]
        exp_mod.run(config=config, **flags)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
