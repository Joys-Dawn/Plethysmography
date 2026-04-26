"""Pytest test suite for the plethysmography package.

The end-to-end tests (``test_e2e_exp1``) require the experiment-1 raw EDFs to
be present under ``Data/experiment 1 - .../experiment 1 - raw data/``. The
unit tests (``test_stats_families``, ``test_breath_metrics``,
``test_sigh_threshold``) run on synthetic / cached data and don't need the
EDFs.
"""
