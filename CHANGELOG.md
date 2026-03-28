# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-03-28

### Added

- Interactions tab — ECharts scatter plot (numeric × numeric) and box plot (categorical × numeric)
- Dynamic column selection with smart dropdown filtering (prevents same-column and cat × cat pairs)
- Configurable sampling for large datasets (`interaction_sample_size`, default 100K)
- Lazy chart rendering — charts initialize only when columns are selected
- Zoom, pan, and dynamic point scaling on scatter plots
- Box plot with 1.5 IQR whiskers and outlier overlay
- `interactions` property and serialization in `to_dict()` / `to_json()`

## [0.2.0] - 2026-03-27

### Added

- 5 correlation types: Spearman, Kendall tau-b, Cramér's V, Phi K — with interactive tab-based HTML heatmaps
- Phi K displayed first as universal metric (all column types, single comparable scale)
- Kendall tau-b via scipy C merge-sort — O(n log n) replacing O(n²) polars-statistics
- Cramér's V via Polars `group_by` + `pivot` contingency tables + `ps.cramers_v` (Rust)
- Phi K via Polars contingency tables + `ps.chisq_test` (Rust) + `phik_from_chi2` with noise correction
- `HIGH_CORRELATION` alert — scans Phi K matrix for pairs above threshold (default 0.8)
- `UNIFORM` alert — chi-squared goodness-of-fit test via `ps.chisq_goodness_of_fit` (Rust)
- Extensible `Alert.details` dict for structured metadata (column_b, method, p_value)
- Alert HTML rendering: color-coded badges per type, HIGH_CORRELATION shows both columns

### Changed

- Correlation threshold default: 0.9 → 0.8
- Pipeline order: correlations computed before alerts (alerts now receive correlation results)

### Dependencies

- Added `scipy>=1.12` (Kendall tau-b C implementation)
- Added `phik>=0.12` (Phi K scalar conversion)

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

[0.3.0]: https://github.com/dataxid/dataxid-profiling/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/dataxid/dataxid-profiling/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dataxid/dataxid-profiling/releases/tag/v0.1.0
