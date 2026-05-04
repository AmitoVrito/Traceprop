# Changelog

All notable changes to Traceprop are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.5.0] — 2025-12-01

### Added
- Sparse Johnson-Lindenstrauss projection in `GradientStore` (Achlioptas 2003 binary coins)
- `StreamingTrainingContext` for online and continual learning
- `data_valuation()` with KNN-Shapley, aggregated by source and preprocessing op
- Multi-source provenance case study notebook (`homecredit_multisource_provenance_colab.ipynb`)
- Vectorised lineage index build via `_build_child_index()` (numpy argsort+split), reducing ETL overhead to 2.93×
- PostgreSQL provenance store
- `eu_ai_act.py` compliance report generator (Article 26 audit trail)

### Changed
- `GradientStore` projection: removed upward projection guard (proj_dim must be ≤ param_dim)
- Default `proj_dim` raised from 1024 to 4096 for better attribution quality

### Fixed
- `defaultdict(lambda: ...)` pickle failure in `LineageGraph`; replaced with named `_inner_dict()` factory

## [0.4.0] — 2025-09-01

### Added
- `AttributionEngine` and `compute_influence_scores` with L-inf normalisation
- `TrainingContext` context manager for recording per-sample gradients
- `unlearn()` via gradient correction (provenance-guided)
- SQLite provenance store
- JAX backend (`traceprop.backends.jax_backend`)
- PyTorch backend (`traceprop.backends.torch_backend`)
- Cython C extension (`_c_ext/graph_ops`) for graph traversal acceleration

### Changed
- `ProvenanceTensor` now supports all NumPy ufuncs via `__array_ufunc__`

## [0.3.0] — 2025-06-01

### Added
- `ProvenanceView` query API: `ancestors()`, `ops()`, `sources()`
- `from_csv()`, `from_numpy()`, `from_jax()`, `from_torch()` entry points
- ProvRC range compression for large operation graphs
- Parquet and OpenTelemetry exporters
- `Granularity` modes: OP, BATCH, EPOCH

## [0.2.0] — 2025-04-01

### Added
- `ProvenanceTensor` wrapping NumPy arrays with lineage tracking
- Lineage DAG (`graph.py`) with parent-child edge recording
- Op-level interception via `__array_function__` and `__array_ufunc__`

## [0.1.0] — 2025-02-01

### Added
- Initial release: proof-of-concept lineage tracking for NumPy
