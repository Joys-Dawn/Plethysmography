"""
Tests for Item E: ``pipelines._common.experiment_output_dirs`` derives the
two sibling top-level folders (interactive HTML vs publication+stats) from
an experiment registry's ``results_folder``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plethysmography.data_loading.data_log import get_experiment_registry
from plethysmography.pipelines._common import experiment_output_dirs


_EXPECTED = {
    1: (
        "Experiment 1 - LR vs HR comparison - interactive plots",
        "Experiment 1 - LR vs HR comparison - publication plots and stats",
    ),
    2: (
        "Experiment 2 - chronic FFA vs vehicle - interactive plots",
        "Experiment 2 - chronic FFA vs vehicle - publication plots and stats",
    ),
    4: (
        "Experiment 4 - survivors vs SUDEP - interactive plots",
        "Experiment 4 - survivors vs SUDEP - publication plots and stats",
    ),
}


@pytest.mark.parametrize("experiment_id", [1, 2, 4])
def test_experiment_output_dirs_exact(experiment_id, tmp_path):
    registry = get_experiment_registry(experiment_id)
    interactive_root, pub_root = experiment_output_dirs(registry, tmp_path)

    exp_int, exp_pub = _EXPECTED[experiment_id]
    assert interactive_root == tmp_path / exp_int
    assert pub_root == tmp_path / exp_pub


@pytest.mark.parametrize("experiment_id", [1, 2, 4])
def test_dirs_are_distinct_siblings(experiment_id, tmp_path):
    registry = get_experiment_registry(experiment_id)
    interactive_root, pub_root = experiment_output_dirs(registry, tmp_path)
    assert interactive_root != pub_root
    # Both are direct children of results_root (siblings at the top level).
    assert interactive_root.parent == tmp_path
    assert pub_root.parent == tmp_path


def test_leading_experiment_token_is_title_cased(tmp_path):
    """The registry stores ``"experiment N - …"`` (lowercase); the folders
    must read ``"Experiment N - …"`` so the two siblings sort together."""
    registry = get_experiment_registry(1)
    interactive_root, pub_root = experiment_output_dirs(registry, tmp_path)
    assert interactive_root.name.startswith("Experiment 1 - ")
    assert pub_root.name.startswith("Experiment 1 - ")


@pytest.mark.parametrize("experiment_id", [1, 2, 4])
def test_creates_no_directories(experiment_id, tmp_path):
    """``experiment_output_dirs`` is pure path derivation — callers mkdir
    only what they actually write."""
    registry = get_experiment_registry(experiment_id)
    interactive_root, pub_root = experiment_output_dirs(registry, tmp_path)
    assert not interactive_root.exists()
    assert not pub_root.exists()


def test_non_experiment_prefixed_name_is_passed_through(tmp_path):
    """A registry name that does not start with ``"experiment "`` is used
    verbatim (defensive: future registry entries / SUDEP-events naming)."""
    interactive_root, pub_root = experiment_output_dirs(
        {"results_folder": "SUDEP fatal seizures"}, tmp_path
    )
    assert interactive_root == tmp_path / "SUDEP fatal seizures - interactive plots"
    assert pub_root == tmp_path / "SUDEP fatal seizures - publication plots and stats"
