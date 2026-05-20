"""
Tests for Item G (experiment 1b):

* the cohort loader returns exactly the two developmental groups
  (het HR P19 / het LR P22) and nothing else;
* ``experiment1b.run`` — with every heavy step mocked — reuses exp1's
  breathing CSV, derives the E-style sibling folders, and routes the binned
  plots through the net-new ``condition_col="developmental"`` variant.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plethysmography.data_loading.data_log import (
    load_recordings_for_experiment1b,
)
from plethysmography.pipelines import experiment1b


# ---------------------------------------------------------------------------
# Cohort loader
# ---------------------------------------------------------------------------
def test_cohort_loader_returns_only_two_developmental_groups():
    recs = load_recordings_for_experiment1b(data_root=".")
    assert recs, "expected a non-empty exp1b developmental cohort"
    for r in recs:
        assert r.genotype == "het", f"{r.file_basename}: non-het leaked in"
        ok = (r.age == "P19" and r.risk == "HR") or (
            r.age == "P22" and r.risk == "LR"
        )
        assert ok, f"{r.file_basename}: {r.age}/{r.risk} is not a dev group"
    # Both groups should actually be present (not a degenerate one-group run).
    combos = {(r.age, r.risk) for r in recs}
    assert ("P19", "HR") in combos
    assert ("P22", "LR") in combos


# ---------------------------------------------------------------------------
# run() with heavy steps mocked
# ---------------------------------------------------------------------------
def test_run_creates_estyle_folders_and_uses_developmental_binned(
    tmp_path: Path, monkeypatch
):
    results_root = tmp_path / "results"

    # exp1's breathing CSV must already exist at exp1's pub_root (Item E
    # path); exp1b reuses it and must not re-preprocess.
    exp1_pub = (
        results_root
        / "Experiment 1 - LR vs HR comparison - publication plots and stats"
    )
    exp1_pub.mkdir(parents=True)
    (exp1_pub / "breathing_analysis_results.csv").write_text("file_basename\nx\n")

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        experiment1b, "load_recordings_for_experiment1b",
        lambda **kw: ["rec_a", "rec_b"],
    )
    monkeypatch.setattr(experiment1b, "load_data_log", lambda *a, **k: pd.DataFrame())

    def _fake_prepare(_bre, _log):
        return pd.DataFrame(
            {
                "genotype_clean": ["het", "het"],
                "risk_clean": ["high_risk", "low_risk"],
                "age_clean": [19, 22],
                "period": ["Ictal", "Ictal"],
            }
        )

    monkeypatch.setattr(experiment1b, "prepare_breathing_data", _fake_prepare)
    monkeypatch.setattr(experiment1b, "preprocess_all",
                        lambda *a, **k: calls.__setitem__("preprocess", True))
    monkeypatch.setattr(experiment1b, "analyze_all",
                        lambda *a, **k: calls.__setitem__("analyze", k))
    monkeypatch.setattr(experiment1b, "run_statistics", lambda *a, **k: [])
    monkeypatch.setattr(experiment1b, "write_stats_xlsx",
                        lambda *a, **k: calls.__setitem__("stats", True))
    monkeypatch.setattr(experiment1b, "plot_developmental_comparison",
                        lambda *a, **k: calls.__setitem__("within", True))
    monkeypatch.setattr(experiment1b, "draw_developmental_timeseries",
                        lambda *a, **k: calls.__setitem__("across", True))
    monkeypatch.setattr(experiment1b, "load_period_data_for_bins",
                        lambda *a, **k: [])
    monkeypatch.setattr(experiment1b, "metadata_for_bins", lambda *a, **k: {})
    monkeypatch.setattr(experiment1b, "baseline_median_ttot_by_basename",
                        lambda *a, **k: {})

    binned_cond: list[str] = []
    monkeypatch.setattr(
        experiment1b, "plot_postictal_binned",
        lambda *a, **k: binned_cond.append(k.get("condition_col")),
    )
    monkeypatch.setattr(
        experiment1b, "plot_ictal_binned",
        lambda *a, **k: binned_cond.append(k.get("condition_col")),
    )

    experiment1b.run(results_root=results_root)

    pub_root = (
        results_root
        / "Experiment 1b - HR P19 vs LR P22 - publication plots and stats"
    )
    interactive_root = (
        results_root / "Experiment 1b - HR P19 vs LR P22 - interactive plots"
    )
    # E-style sibling folders derived from the registry.
    assert pub_root.name.endswith("- publication plots and stats")
    assert interactive_root.name.endswith("- interactive plots")
    # stats/ and plots/ were created under pub_root.
    assert (pub_root / "stats").is_dir()
    assert (pub_root / "plots").is_dir()
    # Every exp1 family ran its 2-group analog ...
    assert calls.get("within") and calls.get("across") and calls.get("stats")
    assert calls.get("preprocess") and "analyze" in calls
    # ... and the binned plots used the net-new developmental variant.
    assert binned_cond == ["developmental", "developmental"]
    # population ictal dir was threaded into analyze_all.
    analyze_kwargs = calls["analyze"]
    assert "population_ictal_dir" in analyze_kwargs


def test_run_bails_when_exp1_breathing_csv_absent(tmp_path: Path, monkeypatch, caplog):
    """exp1b must fail soft (log + return) when exp1 hasn't been run, not
    explode trying to read a missing CSV."""
    monkeypatch.setattr(
        experiment1b, "load_recordings_for_experiment1b", lambda **kw: []
    )
    called = {"prepare": False}
    monkeypatch.setattr(
        experiment1b, "prepare_breathing_data",
        lambda *a, **k: called.__setitem__("prepare", True) or pd.DataFrame(),
    )
    with caplog.at_level("WARNING"):
        experiment1b.run(results_root=tmp_path / "results")
    assert not called["prepare"], "should have returned before prepare"
    assert any("exp1" in r.message for r in caplog.records)
