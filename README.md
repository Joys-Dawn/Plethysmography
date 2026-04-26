# Plethysmography

Pipeline for whole-body plethysmography signal analysis in mice — covers
preprocessing of raw EDF recordings, breath segmentation and per-period
metrics, sigh / apnea detection, multi-design statistics with FDR correction,
and publication plot generation.

The package mirrors the architectural style of the sibling
`Fiber_Photometry_ECoG/` and `EphysAutomatedAnalysis/` projects in the parent
`Mattis_Lab/` directory: frozen-dataclass configs, type-hinted dataclasses for
domain objects, and one entry point per pipeline stage.

## Layout

```
Plethysmography/
├── plethysmography/                  package source
│   ├── core/                         config, data models, exclusions/overrides
│   ├── data_loading/                 EDF reader, data log, lid detection, filename normalization
│   ├── preprocessing/                period slicing, HPF, artifact removal
│   ├── analysis/                     breath segmentation, sigh/apnea, baseline cache
│   ├── stats/                        ANOVA / GEE / t-tests / FDR / 10-sheet xlsx writer
│   ├── visualization/                bar / timeseries / binned / interactive plots
│   ├── pipelines/                    experiment 1, 2, 4 drivers
│   └── tests/                        pytest suite
├── run.py                            CLI dispatcher
├── requirements.txt
├── Data/
│   ├── experiment 1 - LR vs HR comparison/
│   │   ├── experiment 1 - raw data/        EDF inputs
│   │   └── experiment 1 - preprocessed data/   period CSVs (created by pipeline)
│   ├── experiment 2 - chronic FFA vs vehicle/
│   │   └── …
│   └── experiment 4 - survivors vs SUDEP/      (no raw — pulled from 1+2)
├── results/                          all outputs land here
│   ├── experiment 1 - LR vs HR comparison/
│   │   ├── breathing_analysis_results.csv
│   │   ├── apnea_list.xlsx
│   │   ├── stats/statistical_results.xlsx
│   │   └── plots/
│   │       ├── Across time periods/        timeseries (1 line per group)
│   │       ├── Within each time period/    bar plots per (period, parameter)
│   │       ├── HR P19 vs LR P22/           developmental t-test bar plots
│   │       ├── Postictal_Binned/           postictal in 30 s bins
│   │       └── Ictal_Binned/               ictal in 1 s bins (per docs)
│   ├── experiment 2 - chronic FFA vs vehicle/
│   │   └── plots/
│   │       ├── Across time periods/
│   │       ├── Within each time period/
│   │       ├── FFA/{By_age, By_drug, By_genotype}/   subgroup timeseries
│   │       ├── Postictal_Binned/
│   │       └── Ictal_Binned/
│   └── experiment 4 - survivors vs SUDEP/
│       └── plots/                          Survival_<param>.png (one bar per period)
├── docs/                             pleth data log.xlsx and supplementary docs
├── old_code/                         previous monolithic scripts (kept as reference)
└── old_results/                      regression baseline (kept as reference)
```

## Setup

Requires Python 3.13+ on Windows (the EDF reader uses pyedflib, which on this
Python version needs a conda-forge wheel — see the install notes at the top of
`requirements.txt` if pip can't find a wheel).

```bash
python -m pip install -r requirements.txt
```

## Workflow

1. **Run a single experiment**:
   ```bash
   python run.py exp1     # LR vs HR comparison
   python run.py exp2     # chronic FFA vs vehicle
   python run.py exp4     # P19 het: survivors vs SUDEP (reuses 1+2 outputs)
   python run.py all      # all three in order
   ```

2. **Replay a stage** without re-doing the slow earlier ones:
   ```bash
   python run.py preprocess --experiment 1   # only EDF → period CSVs
   python run.py analyze --experiment 1      # CSVs → breathing CSV + apnea xlsx + stats + plots
   python run.py stats --experiment 1        # only stats
   python run.py plots --experiment 1        # only plots
   ```

3. **Custom config** — override any frozen-dataclass field via JSON:
   ```bash
   python run.py exp1 --config my_config.json
   ```

## Tests

```bash
python -m pytest plethysmography/tests/ -v
```
