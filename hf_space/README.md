---
title: Traceprop — Training Data Attribution & Provenance
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Training data attribution & provenance in <1s.
tags:
  - data-attribution
  - machine-unlearning
  - data-provenance
  - explainability
  - interpretability
  - influence-functions
  - eu-ai-act
  - compliance
  - scikit-learn
  - pytorch
---

# 🔍 Traceprop

> **Training-data attribution · Computation lineage · Machine unlearning — all in one library**

Traceprop answers three questions every production ML system should be able to answer:

| Question | Feature | Latency |
|----------|---------|---------|
| *Which training samples drove this prediction?* | Training Data Attribution | **0.22 s** (CPU, tabular) |
| *Where did this data come from?* | Computation Lineage | **< 1 ms** |
| *Can I remove a sample's influence without retraining?* | Machine Unlearning | **seconds** |

## Try it here 👆

Use the tabs above to explore each capability on the Wisconsin Breast Cancer dataset.

## Install

```bash
pip install traceprop
```

## Quick start

```python
import traceprop as tp

# 1 — Record gradients during training
with tp.training_context(model, X_train, y_train, source_id="my_dataset") as ctx:
    train(model, X_train, y_train)

# 2 — Attribution: which samples drove a test prediction?
engine = tp.attribution_engine(ctx.gradient_store, estimator="trak")
result = engine.attribute(test_gradient, top_k=10)
for entry in result.top(10):
    print(entry["sample_index"], entry["influence_score"])

# 3 — Unlearning: surgically remove a sample's influence
tp.unlearn(model=model, gradient_store=ctx.gradient_store, source_id="bad_source")
```

## Benchmark Results

| Dataset | Method | LDS ↑ | Time |
|---------|--------|-------|------|
| Adult Income (tabular) | Traceprop-LL | 0.622 | 0.22 s |
| Adult Income (tabular) | Traceprop-LL + TRAK | **0.884** | 0.6 s |
| Covertype 50K | Traceprop-LL + TRAK | **0.976** | 5.2 s |
| CIFAR-2 / ResNet-9 | TRAK (5 ckpts, GPU) | 0.029 | 691 s |
| CIFAR-2 / Frozen ResNet18 | Traceprop-LL | **0.264** | 10.2 s CPU |

*LDS = Linear Datamodel Score (Park et al. 2023). Higher is better.*

## Links

- 📦 **PyPI:** `pip install traceprop`
- 🐙 **GitHub:** [AmitoVrito/Traceprop](https://github.com/AmitoVrito/Traceprop)
- 📄 **Paper:** VLDB 2027 (under submission)

## Feedback

Found a bug, want a feature, or have questions? Use the **💬 Feedback** tab in the demo or open an issue on [GitHub](https://github.com/AmitoVrito/Traceprop/issues).
