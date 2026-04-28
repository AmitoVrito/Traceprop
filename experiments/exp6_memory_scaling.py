"""Experiment 6: GradientStore Memory Scaling — theoretical O(n*k)."""

import json

PROJ_DIMS = [256, 512, 1024, 2048, 4096]
N_SAMPLES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

results = {}

for k in PROJ_DIMS:
    for n in N_SAMPLES:
        matrix_bytes = n * k * 4  # float32
        matrix_gb = matrix_bytes / 1024**3
        key = f"k={k}_n={n}"
        results[key] = {
            "proj_dim": k,
            "n_samples": n,
            "theoretical_memory_gb": round(matrix_gb, 3),
        }
        print(f"k={k:5}, n={n:>10}: theoretical memory = {matrix_gb:.3f} GB")

with open("results/exp6_memory.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/exp6_memory.json")
