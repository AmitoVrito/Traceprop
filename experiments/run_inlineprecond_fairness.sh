#!/bin/bash
# Multi-seed inline-precond fairness campaign (Checks 1-3 + per-module damping).
# Args: $1=seeds (default "0 1 2 3 4"), $2=prefix (default "fair"),
#       $3=extra exp35 args (e.g. "--logix_permodule_damping").
# For each target-model seed: run exp35 (--inline_precond --tune_logix [extra])
# then the held-out two-way bootstrap (inline vs LogIX_tuned). All -> results/.
set -e
cd "$(dirname "$0")"
PY=../.venv310/bin/python
SEEDS="${1:-0 1 2 3 4}"
PREFIX="${2:-fair}"
EXTRA="${3:-}"
GRID="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1"

for s in $SEEDS; do
  echo "==================== SEED $s : exp35 ($PREFIX) ===================="
  $PY exp35_logix_lds.py --backend tiny --device cpu --track 0 \
    --factored --inline_precond --tune_logix $EXTRA \
    --n_train 400 --n_test 100 --n_subsets 200 --subset_frac 0.5 --epochs 3 \
    --batch 16 --proj_dim 512 --lora_init pca --seed "$s" \
    --precond_damping_grid "$GRID" --precond_val_frac 0.3 \
    --out "results/exp35_tiny_track0_${PREFIX}_seed${s}.json" --force --skip_gpu_check \
    2>&1 | grep -E "target test accuracy|LogIX damping sweep|default_bug|LogIX tuned|inline-precond kfac|subset 200/|saved ->|Traceback|Error" | grep -vE "warn"
  echo "==================== SEED $s : bootstrap ($PREFIX) ===================="
  $PY exp35_inlineprecond_bootstrap.py \
    "results/exp35_tiny_track0_${PREFIX}_seed${s}_raw.npz" --n_boot 2000 --seed 0 \
    2>&1 | grep -E "inlineprecond -|SEED VERDICT|saved ->" | grep -vE "warn"
done
echo "==================== CAMPAIGN DONE ($PREFIX) ===================="
