# Plethysmography

Pipeline for plethysmography signal analysis in heated mice during seizure episodes. Covers
preprocessing of raw EDF recordings, breath segmentation and per-period
metrics, sigh / apnea detection, multi-design statistics with FDR correction,
and publication plot generation.

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
