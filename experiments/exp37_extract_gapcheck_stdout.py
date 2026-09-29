"""Extract the mislabel-detection (and backdoor) results dict that
exp37_planted_detection.py prints via `print(json.dumps(out, indent=2))`
before saving, from a captured stdout transcript.

Why this script exists rather than a raw-npz recomputation: the run's
results/exp37_gapcheck.json and results/exp37_gapcheck_raw.npz were written
to the Colab runtime's local disk but never copied to Drive before the
runtime recycled (confirmed: the notebook's gap-check cell only backs up
stdout.txt, unlike the full n_seeds=5 run cell which also copies the json
and _raw.npz). Only the stdout transcript survived. Since the script prints
the complete JSON blob to stdout before writing files, that transcript is a
faithful, unmodified copy of the same data -- this parses it back into a
real JSON file rather than anyone retyping numbers from memory.

Usage: python exp37_extract_gapcheck_stdout.py
"""
import json
import re

SRC = "results/exp37_gapcheck_stdout.txt"
OUT = "results/exp37_gapcheck_extracted.json"


def main():
    with open(SRC) as f:
        text = f.read()

    # The printed JSON is the first top-level {...} block in the transcript
    # (everything before it is per-seed log lines; everything after is the
    # "saved -> ..." confirmation lines).
    start = text.index("{")
    # Find the matching closing brace by tracking depth.
    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise SystemExit(f"could not find a balanced JSON block in {SRC}")

    blob = text[start:end]
    data = json.loads(blob)

    # Sanity checks tying this extraction to what the paper claims.
    mis = data["secondary_mislabel"]
    assert set(mis) == {"loss", "grad_norm", "self_infl_dot", "self_infl_trak", "random"}, \
        f"unexpected mislabel metric keys: {sorted(mis)}"

    print("Run params: model={model} plant_frac(from n_train/k)={n_train}/{k_backdoor} "
          "epochs={epochs} n_seeds={n_seeds}".format(**data))
    print(f"per_seed_backdoor_learned={data['per_seed_backdoor_learned']} "
          f"(gap check {'PASSED' if any(data['per_seed_backdoor_learned']) else 'FAILED'} "
          f"-- backdoor AUCs in this run are flagged unreliable by the script's own warning "
          f"when this is False; mislabel AUCs are independent of backdoor learning and unaffected)")
    print("\nsecondary_mislabel (mislabel-detection AUCs, self-influence vs. free baselines):")
    for name, v in mis.items():
        print(f"  {name:15s} auc={v['auc']['mean']:.4f}  precision_at_k={v['precision_at_k']['mean']:.4f}")

    out = {
        "source": SRC,
        "source_provenance": "stdout transcript of notebooks/exp37_pythia_colab.ipynb cell 6 "
                              "(gap-check cell, n_seeds=1); the run's own results/exp37_gapcheck.json "
                              "and results/exp37_gapcheck_raw.npz were never copied off the ephemeral "
                              "Colab runtime, so this is reconstructed from the script's own "
                              "print(json.dumps(out)) call captured in the saved stdout, not retyped "
                              "from memory or recomputed from raw per-example arrays.",
        "run_params": {k: data[k] for k in
                       ("backend", "model", "n_seeds", "seeds", "n_train", "k_backdoor",
                        "k_distractor", "k_mislabel", "trigger_len", "epochs")},
        "backdoor_gap_check": {
            "per_seed_backdoor_learned": data["per_seed_backdoor_learned"],
            "status": "FAILED (gap +0.25 < 0.3 threshold) per the script's own warning -- "
                      "backdoor AUCs from this run are NOT used as evidence in the paper; "
                      "mislabel AUCs below are unaffected by this and are used.",
        },
        "secondary_mislabel": mis,
        "primary_backdoor_unreliable_do_not_cite": data["primary_backdoor"],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
