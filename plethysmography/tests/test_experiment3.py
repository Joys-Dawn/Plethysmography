"""
Tests for experiment 3 (acute FFA vs vehicle):

  * the cohort loader returns the 22 acute P22 recordings (4-cell genotype x
    treatment table with the expected counts) and the registry entry resolves
    to the new sibling folders;
  * ``experiment3.run`` — with every heavy step mocked — drives the
    publication plot bundle with ``palette=ACUTE_FFA_PALETTE`` and
    ``condition_col="treatment_clean"``, creates the E-style sibling
    folders, and does **not** import / call ``plot_ffa_subgroups`` (the
    multi-age subgroup driver that would yield degenerate panels for the
    single-age acute cohort);
  * when no recordings are loaded (e.g., the data log lists zero acute
    rows on this machine), ``run()`` logs a warning and returns
    cleanly without exploding.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plethysmography.data_loading.data_log import (
    get_experiment_registry,
    load_recordings_for_experiment,
)
from plethysmography.pipelines import experiment3
from plethysmography.visualization.colors import ACUTE_FFA_PALETTE


# ---------------------------------------------------------------------------
# Cohort loader + registry + whitelist (combined per plan)
# ---------------------------------------------------------------------------
def test_cohort_loader_and_registry():
    # Registry entry resolves to the acute folders.
    reg = get_experiment_registry(3)
    assert reg["cohort_folder"] == "experiment 3 - acute FFA vs vehicle"
    assert reg["raw_subfolder"] == "experiment 3 - raw data"
    assert reg["preprocessed_subfolder"] == "experiment 3 - preprocessed data"
    assert reg["results_folder"] == "experiment 3 - acute FFA vs vehicle"
    assert reg["experiment_values"] == frozenset({"FFA vs vehicle - acute"})

    # Whitelist was widened: id 3 must not raise. Use ``data_root="Data"``
    # so edf_path resolves to actual EDFs on disk (matches the CLI's
    # ``DATA_ROOT = Path("Data")``).
    recs = load_recordings_for_experiment(3, data_root="Data")

    # The data log lists 26 acute rows; 4 mice have log rows but no EDFs on
    # disk. The loader returns all 26 Recording objects (those with missing
    # EDFs are silently skipped at ``preprocess_all`` time, identical to how
    # exp1 / exp2 already handle missing files).
    assert len(recs) >= 22, f"expected at least 22 acute recordings, got {len(recs)}"

    # Every acute mouse is P22; treatment is FFA / Vehicle; risk + SUDEP +
    # survivor flags are unset.
    for r in recs:
        assert r.age == "P22", f"{r.file_basename}: age={r.age!r}, expected P22"
        assert r.treatment in {"FFA", "Vehicle"}, (
            f"{r.file_basename}: treatment={r.treatment!r}"
        )
        assert r.risk is None, f"{r.file_basename}: unexpected risk={r.risk!r}"
        assert not r.is_sudep, f"{r.file_basename}: is_sudep is True"
        assert not r.is_survivor, f"{r.file_basename}: is_survivor is True"
        assert r.seizure_offset_s is not None, (
            f"{r.file_basename}: missing seizure offset"
        )
        assert r.cohort == "experiment 3 - acute FFA vs vehicle"

    # Cell counts restricted to the EDFs actually on disk match the locked
    # Context table (WT Vehicle=9 / WT FFA=4 / het Vehicle=4 / het FFA=5 = 22).
    on_disk = [r for r in recs if r.edf_path.exists()]
    assert len(on_disk) == 22, (
        f"expected 22 acute EDFs on disk, got {len(on_disk)}"
    )
    counts: dict[tuple[str, str], int] = {}
    for r in on_disk:
        counts[(r.genotype, r.treatment)] = counts.get(
            (r.genotype, r.treatment), 0,
        ) + 1
    assert counts == {
        ("WT", "Vehicle"): 9,
        ("WT", "FFA"): 4,
        ("het", "Vehicle"): 4,
        ("het", "FFA"): 5,
    }, f"acute cell counts drifted: {counts}"


# ---------------------------------------------------------------------------
# run() smoke test with every heavy step mocked
# ---------------------------------------------------------------------------
def test_run_creates_estyle_folders_and_forwards_acute_palette(
    tmp_path: Path, monkeypatch
):
    results_root = tmp_path / "results"
    calls: dict[str, object] = {}

    # Two synthetic acute recordings so the (non-empty) breathing_df / merged
    # paths inside run() execute.
    from plethysmography.core.data_models import Recording

    edf_path = tmp_path / "fake.EDF"
    edf_path.write_bytes(b"")
    fake_recs = [
        Recording(
            file_basename="040226 5336 p22", edf_path=edf_path,
            mouse_id="5336", age="P22", genotype="WT",
            cohort="experiment 3 - acute FFA vs vehicle",
            treatment="Vehicle", seizure_offset_s=50.0,
        ),
        Recording(
            file_basename="040226 5337 p22", edf_path=edf_path,
            mouse_id="5337", age="P22", genotype="het",
            cohort="experiment 3 - acute FFA vs vehicle",
            treatment="FFA", seizure_offset_s=60.0,
        ),
    ]

    monkeypatch.setattr(
        experiment3, "load_recordings_for_experiment",
        lambda *_args, **_kw: fake_recs,
    )
    monkeypatch.setattr(
        experiment3, "load_data_log", lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(
        experiment3, "preprocess_all",
        lambda *a, **k: (calls.__setitem__("preprocess", True) or fake_recs),
    )

    fake_breathing = pd.DataFrame({
        "file_basename": ["040226 5336 p22", "040226 5337 p22"],
        "period": ["Ictal", "Ictal"],
    })
    fake_apnea = pd.DataFrame()
    monkeypatch.setattr(
        experiment3, "analyze_all",
        lambda *a, **k: (calls.__setitem__("analyze", k)
                        or (fake_breathing, fake_apnea)),
    )
    monkeypatch.setattr(
        experiment3, "write_breathing_outputs",
        lambda *a, **k: calls.__setitem__("write", True),
    )
    monkeypatch.setattr(
        experiment3, "prepare_breathing_data",
        lambda *_a, **_k: pd.DataFrame({
            "genotype_clean": ["WT", "het"],
            "treatment_clean": ["Vehicle", "FFA"],
            "age_clean": [22, 22],
            "period": ["Ictal", "Ictal"],
        }),
    )
    monkeypatch.setattr(
        experiment3, "run_statistics",
        lambda *a, **k: (calls.__setitem__("stats_kwargs", k) or []),
    )
    monkeypatch.setattr(
        experiment3, "write_stats_xlsx",
        lambda *a, **k: calls.__setitem__("stats_xlsx", True),
    )
    monkeypatch.setattr(
        experiment3, "load_period_data_for_bins", lambda *a, **k: [],
    )
    monkeypatch.setattr(
        experiment3, "metadata_for_bins", lambda *a, **k: {},
    )
    monkeypatch.setattr(
        experiment3, "baseline_median_ttot_by_basename", lambda *a, **k: {},
    )
    monkeypatch.setattr(
        experiment3, "generate_publication_plots",
        lambda *a, **k: calls.__setitem__("plot_kwargs", k) or {},
    )

    experiment3.run(results_root=results_root)

    # E-style sibling folders derived from the registry.
    pub_root = (
        results_root
        / "Experiment 3 - acute FFA vs vehicle - publication plots and stats"
    )
    interactive_root = (
        results_root / "Experiment 3 - acute FFA vs vehicle - interactive plots"
    )
    assert pub_root.name.endswith("- publication plots and stats")
    assert interactive_root.name.endswith("- interactive plots")
    assert (pub_root / "stats").is_dir()
    assert (pub_root / "plots").is_dir()

    # Every heavy step ran.
    assert calls.get("preprocess") is True
    assert "analyze" in calls and "write" in calls and "stats_xlsx" in calls

    # Stats was called with the acute condition + levels (same as chronic).
    stats_kwargs = calls["stats_kwargs"]
    assert stats_kwargs["condition_col"] == "treatment_clean"
    assert stats_kwargs["condition_levels"] == ("FFA", "Vehicle")

    # generate_publication_plots received palette=ACUTE_FFA_PALETTE and the
    # treatment_clean condition col.
    plot_kwargs = calls["plot_kwargs"]
    assert plot_kwargs["condition_col"] == "treatment_clean"
    assert plot_kwargs["palette"] is ACUTE_FFA_PALETTE

    # plot_ffa_subgroups is intentionally NOT imported by experiment3.
    assert not hasattr(experiment3, "plot_ffa_subgroups"), (
        "experiment3 must not import plot_ffa_subgroups — the multi-age "
        "subgroup driver would render degenerate panels for the acute "
        "single-age cohort (Item G risk pattern)."
    )


# ---------------------------------------------------------------------------
# No-EDFs / no-recordings soft-fail
# ---------------------------------------------------------------------------
def test_run_bails_when_no_recordings(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setattr(
        experiment3, "load_recordings_for_experiment", lambda *_a, **_kw: [],
    )
    called = {"preprocess": False}
    monkeypatch.setattr(
        experiment3, "preprocess_all",
        lambda *a, **k: called.__setitem__("preprocess", True) or [],
    )

    with caplog.at_level("WARNING"):
        experiment3.run(results_root=tmp_path / "results")

    assert not called["preprocess"], "should have returned before preprocess"
    assert any("experiment 3" in r.message.lower() for r in caplog.records)
