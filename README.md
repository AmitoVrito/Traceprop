# Traceprop

**Computation-level data lineage, gradient attribution, and provenance-guided unlearning in production ML.**

Traceprop is a Python library that connects raw source files through preprocessing, through model training, to individual predictions — and lets you act on that lineage via attribution, unlearning, and compliance reporting.

```
pip install traceprop
```

[![PyPI](https://img.shields.io/pypi/v/traceprop)](https://pypi.org/project/traceprop/)
[![Python](https://img.shields.io/pypi/pyversions/traceprop)](https://pypi.org/project/traceprop/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20035922.svg)](https://doi.org/10.5281/zenodo.20035922)
[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/Nautiverse/traceprop-demo)

---

## 🤗 Live Demo

Try Traceprop interactively — no install needed:

**[huggingface.co/spaces/Nautiverse/traceprop-demo](https://huggingface.co/spaces/Nautiverse/traceprop-demo)**

The demo covers all three core capabilities on the Wisconsin Breast Cancer dataset (CPU-only):

| Tab | What it shows |
|-----|--------------|
| 🎯 Attribution | Pick any test sample — see top-K training points that drove the prediction, with influence scores in milliseconds |
| 🗂️ Provenance | Adjust a multi-source preprocessing pipeline and watch the lineage graph update live |
| 🧹 Unlearning | Choose a training sample to forget — see loss increase on that sample while test accuracy is preserved |

### Run the demo locally

```bash
git clone https://github.com/AmitoVrito/Traceprop.git
cd Traceprop/hf_space
pip install -r requirements.txt
python app.py
# → opens at http://127.0.0.1:7860
```

---

## What it does

A single Traceprop query answers:

> *"This model made prediction X on input Z. Which rows in which source files, through which preprocessing steps, most influenced that prediction - and can we reduce that influence without retraining?"*

| Capability | What you get |
|---|---|
| **Lineage tracking** | Sub-1% overhead in op-mode; tracks every NumPy, PyTorch, and JAX operation |
| **Attribution** | LDS 0.976 on Covertype 50K, 0.884 on Adult Income — at 0.22–5.2 s CPU, no GPU needed |
| **Source-stratified attribution (SS)** | Aggregates per-sample scores to source-table level via the lineage graph; SAB v1 benchmark on 4 real datasets (COMPAS +16.4 pp, Lending Club +5.0 pp on 1.3M loans, Home Credit +4.2 pp, Bank Marketing −2.2 pp); paired t-test + Wilcoxon + bootstrap with Bonferroni at α=0.0125 |
| **Approximate unlearning** | Provenance-guided gradient correction; reliable separation from random baseline (~0%) across 10 seeds on Adult Income and Covertype |
| **Compliance reporting** | Structured JSON audit trail for EU AI Act Article 26 obligations |
| **Data valuation** | KNN-Shapley values aggregated by source file and preprocessing op |

---

## Installation

```bash
# Core (NumPy only)
pip install traceprop

# With PyTorch support
pip install "traceprop[torch]"

# With JAX support
pip install "traceprop[jax]"

# With PostgreSQL provenance store
pip install "traceprop[postgres]"

# Everything
pip install "traceprop[all]"
```

Requires Python 3.10+.

---

## Quick start

```python
import traceprop as tp
import numpy as np

# 1. Load source data with provenance tracking
data_a = tp.from_csv("hospital_a.csv", source_id="hospital_a")
data_b = tp.from_csv("hospital_b.csv", source_id="hospital_b")

# 2. Preprocessing — every op is recorded in the lineage graph
norm_a = (data_a - data_a.mean(axis=0)) / (data_a.std(axis=0) + 1e-8)
norm_b = (data_b - data_b.mean(axis=0)) / (data_b.std(axis=0) + 1e-8)

# 3. Train with gradient recording
with tp.training_context(source_id="hospital_a") as ctx:
    train(model, X_train, y_train)   # your training loop here

# 4. Attribute a prediction back to source rows
engine = tp.attribution_engine(ctx.gradient_store)
result = engine.attribute(test_gradient, top_k=10)

for entry in result.top(5):
    print(entry["source_id"], entry["sample_index"], entry["influence_score"])

# 5. Trace the top sample back to its source file and preprocessing ops
trace = result.trace_to_file(rank=0)
print(trace["sources"], trace["ops"])

# 6. Unlearn a data source without retraining
unlearn_result = tp.unlearn(
    gradient_store=ctx.gradient_store,
    source_id="hospital_a",
    n_steps=300,
    lr=1e-2,
)
print(f"Verified: {unlearn_result.verified}")

# 7. Generate EU AI Act compliance report
report = tp.compliance_report(
    tensor=norm_a,
    system_name="CreditScorer-v1",
    system_version="1.0.0",
    deployer_name="Amit N.",
    high_risk_category="credit_scoring",
    output_path="compliance_report.json",
)
```

---

## Core API

### Provenance tracking

| Function | Description |
|---|---|
| `tp.from_numpy(arr, source_id=...)` | Wrap a NumPy array with lineage tracking |
| `tp.from_csv(path, source_id=...)` | Load CSV with lineage tracking |
| `tp.from_torch(data, source_id=...)` | Wrap a PyTorch tensor |
| `tp.from_jax(data, source_id=...)` | Wrap a JAX array |
| `tp.array(data, source_id=...)` | Like `np.array` but tracked |
| `tp.provenance(tensor)` | Get a `ProvenanceView` to query lineage |
| `tp.reset_graph()` | Start a fresh lineage graph |

### ProvenanceView

```python
view = tp.provenance(tensor)
view.ancestors()      # set of ancestor node IDs
view.ops()            # list of preprocessing operations
view.sources()        # list of source_ids in lineage
```

### Attribution

```python
# Record gradients during training
with tp.training_context(model, X_train, y_train, source_id="data", proj_dim=4096) as ctx:
    ...  # training loop

# Attribute a test prediction
engine = tp.attribution_engine(ctx.gradient_store)
result = engine.attribute(test_gradient, top_k=50)

result.top(10)            # list of dicts: sample_index, source_id, influence_score
result.trace_to_file(0)   # trace rank-0 sample to source file + ops
result.by_source()        # aggregate influence by source_id
```

`GradientStore` uses a sparse Johnson-Lindenstrauss projection (Achlioptas 2003) with `{-1, 0, +1}` coins. Default `proj_dim=4096` works well for tabular models; use lower values for memory-constrained environments.

### Unlearning

```python
result = tp.unlearn(
    gradient_store=ctx.gradient_store,
    source_id="hospital_a",   # data source to forget
    n_steps=300,
    lr=1e-2,
    verification_threshold=0.05,
)
result.verified             # bool
result.influence_before     # float
result.influence_after      # float
result.compliance_report    # dict
```

### Data valuation

```python
val_result = tp.data_valuation(
    gradient_store=ctx.gradient_store,
    val_gradients=val_grads,   # (n_val, grad_dim) array
    k=10,
)
val_result.by_source()    # Shapley values aggregated by source
val_result.by_op()        # Shapley values aggregated by preprocessing op
```

### Compliance

```python
report = tp.compliance_report(
    tensor=output_tensor,
    system_name="MyModel",
    system_version="1.0.0",
    deployer_name="Amit N.",
    high_risk_category="credit_scoring",
    output_path="report.json",   # optional: write to file
)
```

Produces a structured JSON report covering EU AI Act Article 26 audit trail requirements for high-risk AI systems (enforcement backstop: 2 December 2027).

### Granularity modes

```python
tp.set_granularity(tp.Granularity.OP)      # default: track every op
tp.set_granularity(tp.Granularity.BATCH)   # batch-level only (lower overhead)
tp.set_granularity(tp.Granularity.EPOCH)   # epoch-level only
```

---

## Benchmarks

### Attribution quality (LDS — Linear Datamodeling Score)

Higher is better. Measured on 500 held-out retraining subsets.

**Tabular / linear models**

| Method | Dataset | LDS | Std | Time | Hardware |
|---|---|---|---|---|---|
| Traceprop-LL | Adult Income (n=6K, d=105) | 0.622 | ±0.180 | 0.22 s | CPU |
| Traceprop-LL + TRAK est. | Adult Income (n=6K, d=105) | 0.884 | ±0.096 | 0.6 s | CPU |
| Traceprop-LL | Covertype (n=50K, d=54) | 0.7513 | ±0.1292 | 3.4 s | CPU |
| Traceprop-LL + TRAK est. | Covertype (n=50K, d=54) | **0.9763** | ±0.1052 | 5.2 s | CPU |
| Traceprop-BM | Adult Income | 0.0127 | ±0.0436 | 0.16 s | CPU |
| Random | — | ~0.000 | — | — | — |

**Deep vision — end-to-end (BatchNorm)**

| Method | Dataset | LDS | Std | Time | Hardware |
|---|---|---|---|---|---|
| TRAK (5 ckpts) | CIFAR-2 / ResNet-9 | 0.0290 | ±0.0523 | 691 s | GPU (T4) |
| Traceprop-LL | CIFAR-2 / ResNet-9 | 0.0168 | ±0.0684 | 2.6 s | CPU |
| Traceprop-BM | CIFAR-2 / ResNet-9 | 0.0033 | ±0.0334 | 14.2 s | CPU |
| Random | CIFAR-2 / ResNet-9 | 0.0205 | ±0.0357 | — | — |

**Deep vision — frozen backbone + linear probe (no BatchNorm)**

| Method | Dataset | LDS | Std | Time | Hardware |
|---|---|---|---|---|---|
| Traceprop-LL (dot) | CIFAR-2 / frozen ResNet-18 | **0.2642** | ±0.1037 | 10.2 s | CPU |
| Traceprop-LL + TRAK est. | CIFAR-2 / frozen ResNet-18 | 0.2307 | ±0.0459 | 1.4 s | CPU |
| Random | — | 0.0018 | — | — | — |

**PyTorch MLP**

| Method | Dataset | LDS | Std | Time | Hardware |
|---|---|---|---|---|---|
| Traceprop-LL + TRAK est. | MNIST 4 vs 9 (784→256→1, n=6K) | 0.1930 | ±0.0581 | 0.82 s | CPU |
| Random | — | 0.0005 | — | — | — |

**Recommendation**: Traceprop-LL is exact for linear models and frozen-backbone architectures (no BatchNorm). Use it for tabular data — it matches or beats TRAK at CPU speeds. For end-to-end deep vision with BatchNorm, TRAK is preferred; Traceprop-LL is 266× faster but scores near random due to BatchNorm corrupting per-sample gradients. The fix is a frozen backbone: LDS improves 15.7× (0.0168 → 0.2642).

### Lineage overhead

| Platform | Overhead | Mode |
|---|---|---|
| macOS (M-series) | 1.007× | op-mode |
| Linux (x86-64) | 0.979× | op-mode |

Sub-1% overhead at 10⁶+ array elements.

### Unlearning

Multi-seed gap-closed (mean ± std over 10 seeds; `C=100`, $|\mathcal{F}|$ selected by top-attribution; `exp19b`):

| Dataset | Method | Gap Closed | Test Acc. |
|---|---|---|---|
| Adult Income (n=6K) | **Traceprop (3 steps, tuned)** | **248% ± 149%** | 0.846 ± 0.025 |
| Adult Income (n=6K) | Traceprop (5 steps, sensitivity) | 534% ± 281% | 0.840 ± 0.027 |
| Adult Income (n=6K) | Random | 0.7% ± 6.7% | 0.840 ± 0.025 |
| Covertype (n=50K) | **Traceprop (2 steps, tuned)** | **52% ± 22%** | 0.754 ± 0.015 |
| Covertype (n=50K) | Traceprop (5 steps, sensitivity) | 150% ± 63% | 0.748 ± 0.020 |
| Covertype (n=50K) | Random | 0.1% ± 0.7% | 0.760 ± 0.012 |

Gap-closed is highly seed-dependent because the influence-ranked forget set $\mathcal{F}$ shifts with the seed used to fit the original model, and small shifts cause large multiplicative swings in $L_{\text{method}}$. The load-bearing signal is the random baseline (~0% at both scales, reliably distinct from every Traceprop row) — provenance-identified forget sets are materially more influential than randomly chosen samples. Test accuracy stays within 1.1 pp of the unmodified model.

### Source-stratified attribution (SAB v1)

Traceprop-SS answers "which source *table* drove this prediction?" by aggregating per-sample TRAK scores to source-file level via the lineage graph and applying a prior-corrected block-magnitude aggregator (`exp22b/c/d/e/f`). SAB v1 ships a four-dataset benchmark with paired t-test, Wilcoxon, and bootstrap CI; we report each below over 20 disjoint held-out seeds.

| Dataset | n | SS | gradmag baseline | Δ (pp) | paired p | wins |
|---|---|---|---|---|---|---|
| COMPAS recidivism | 5,278 | 0.630 | 0.466 | **+16.4** | <10⁻⁴ | 20/20 |
| Home Credit Default Risk | 307,511 | 0.567 | 0.524 | **+4.2** | 0.008 | 15/20 |
| Lending Club (1.3M subsampled to 50K/seed) | 1,303,607 | 0.670 | 0.619 | **+5.0** | <10⁻¹⁰ | 20/20 |
| Bank Marketing | 45,211 | 0.481 | 0.503 | −2.2 | 0.034 | 6/20 |

The first three datasets clear Bonferroni at α=0.05/4=0.0125 on paired t-test, Wilcoxon signed-rank, and bootstrap 95% CI (all positive throughout); Bank Marketing is the regime boundary — under Bonferroni the negative result is "absence of significant positive effect," not a significant negative. The contribution is regime-specific: SS helps when source partitions correspond to data-collection or ETL provenance over structurally different upstream artifacts (joined tables with distinct aggregations), not when source groups are partitions of a single shallow categorical-heavy table.

---

## Backends

| Backend | Install | Usage |
|---|---|---|
| NumPy | built-in | `tp.from_numpy(arr)` |
| PyTorch | `pip install "traceprop[torch]"` | `tp.from_torch(tensor)` |
| JAX | `pip install "traceprop[jax]"` | `tp.from_jax(array)` |

---

## Provenance stores

By default Traceprop uses an in-memory store. For persistence:

```python
# SQLite
from traceprop.stores.sqlite_store import SQLiteStore
store = SQLiteStore("lineage.db")

# PostgreSQL
from traceprop.stores.postgres_store import PostgresStore
store = PostgresStore("postgresql://user:pass@localhost/mydb")
```

---

## Examples

- [`examples/full_pipeline_demo.py`](examples/full_pipeline_demo.py) — full end-to-end demo: two hospital CSVs → preprocessing → training → attribution → unlearning → compliance report
- [`notebooks/tabular_logistic_lds_colab.ipynb`](notebooks/tabular_logistic_lds_colab.ipynb) — LDS benchmark on Adult Income (Colab, CPU)
- [`notebooks/cifar2_resnet9_lds_colab.ipynb`](notebooks/cifar2_resnet9_lds_colab.ipynb) — LDS benchmark on CIFAR-2/ResNet-9 (Colab, GPU T4)
- [`notebooks/homecredit_multisource_provenance_colab.ipynb`](notebooks/homecredit_multisource_provenance_colab.ipynb) — multi-source provenance case study (3-table credit risk data)

---

## Project structure

```
traceprop/
  __init__.py            # public API
  tensor.py              # ProvenanceTensor (NumPy wrapper)
  graph.py               # lineage DAG
  query.py               # ProvenanceView
  interceptor.py         # op-level interception
  granularity.py         # Granularity modes
  compression.py         # ProvRC range compression
  exporters.py           # Parquet / OpenTelemetry exporters
  exceptions.py
  attribution/
    training_context.py  # TrainingContext, GradientStore
    gradient_store.py    # sparse JL projection
    influence.py         # compute_influence_scores
    attribution_engine.py
    streaming_context.py # online / continual learning
  backends/
    numpy_backend.py
    torch_backend.py
    jax_backend.py
  stores/
    memory_store.py
    sqlite_store.py
    postgres_store.py
  compliance/
    eu_ai_act.py         # EU AI Act Article 26 report generator
  unlearning/
    gradient_correction.py
  valuation/
    knn_shapley.py
  _c_ext/
    graph_ops.pyx        # optional Cython acceleration
```

---

## Contributing

Issues and pull requests are welcome. Please open an issue before submitting a large PR.

```bash
git clone https://github.com/AmitoVrito/Traceprop.git
cd Traceprop
pip install -e ".[dev]"
pytest
```

---

## Citation

If you use Traceprop in research, please cite:

```bibtex
@misc{traceprop2027,
  author    = {Amit Nautiyal},
  title     = {{Traceprop}: Computation-Level Data Lineage, Gradient Attribution,
               and Provenance-Guided Unlearning in Production {ML}},
  year      = {2027},
  doi       = {10.5281/zenodo.20036000},
  url       = {https://zenodo.org/records/20036000},
  note      = {Software: \url{https://pypi.org/project/traceprop/}}
}
```

A Zenodo preprint is available at **https://zenodo.org/records/20036000** (DOI: 10.5281/zenodo.20036000).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
