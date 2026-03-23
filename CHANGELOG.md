# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-23

### Added

- `ProfileReport` — 3-line data profiling for Polars and Pandas DataFrames
- 5 column analyzers: numeric, categorical, boolean, datetime, text
- Dataset overview: row/column counts, missing totals, duplicate rows, memory usage, type distribution
- 7 data quality alerts: HIGH_MISSING, CONSTANT, HIGH_CARDINALITY, DUPLICATES, HIGH_ZEROS, SKEWED, IMBALANCED
- Pearson correlation heatmap (Polars-native)
- Interactive HTML report with ECharts (histogram, bar, pie, word cloud, heatmap)
- Programmatic access: `to_dict()`, `to_json()`, `.stats`, `.alerts`, `.correlations`
- Two profiling modes: `"complete"` (deep analysis) and `"overview"` (fast summary)
- Pandas auto-conversion via `pl.from_pandas()`
- CSV and Parquet file path ingestion
- PEP 561 `py.typed` marker

[0.1.0]: https://github.com/dataxid/dataxid-profiling/releases/tag/v0.1.0
