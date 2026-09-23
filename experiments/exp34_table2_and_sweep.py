"""exp34 -- Consolidated, resumable Table 2 (head-to-head) + tracked-parameters
sweep rerun, all on the SAME hardware as Table 1 (L4), in one process.

Why this exists: the Colab notebook (exp25_llm_inline_overhead_colab.ipynb)
hit the same "stale browser tab" problem twice -- cell source silently
reverted to an old --repeats 5 config when Colab auto-saved a tab that was
opened before a fix landed. Running everything as one `!python` invocation
sidesteps that entirely: there's no cell state to go stale, and a full rerun
here always uses whatever code is actually checked into the repo.

Design:
  - Runs all four jobs (Table 2 for GPT-2/Pythia-410M/Pythia-1B, plus the
    tracked-parameters sweep at Table 1's batch=16/seq=64 config) in one
    process, sequentially, on one GPU (matches "keep hardware consistent
    with Table 1" -- do NOT run this on Kaggle, which offers T4/P100, not
    L4; mixing GPUs across tables makes the percentages and post-hoc
    seconds incomparable).
  - Writes each job's result to results/exp34_<job>.json immediately after
    that job finishes (not buffered to the end), and to Google Drive if
    mounted, so a disconnect after job N still leaves jobs 1..N usable.
  - Resumable: on start, skips any job whose result file already exists.
    Delete the specific results/exp34_<job>.json to force a rerun of just
    that job.

Usage (Colab, GPU runtime):
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/Traceprop/experiments
    !python exp34_table2_and_sweep.py --drive_dir /content/drive/MyDrive/traceprop_runs

If the runtime disconnects mid-run, reconnect, re-mount Drive, `git pull`,
and just run the same command again -- completed jobs are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp26_posthoc_vs_inline as exp26
from exp25_llm_inline_overhead import run as exp25_run
from types import SimpleNamespace


TABLE2_JOBS = [
    ("table2_gpt2", dict(backend="hf", model="gpt2", device="cuda",
                          n_samples=512, batch=8, seq=128, rank=8, proj_dim=512,
                          track=1, repeats=20, trak_ckpts=5)),
    ("table2_pythia410m", dict(backend="hf", model="EleutherAI/pythia-410m", device="cuda",
                                n_samples=512, batch=8, seq=128, rank=8, proj_dim=512,
                                track=1, repeats=20, trak_ckpts=5)),
    ("table2_pythia1b", dict(backend="hf", model="EleutherAI/pythia-1b", device="cuda",
                              n_samples=384, batch=8, seq=128, rank=8, proj_dim=512,
                              track=1, repeats=20, trak_ckpts=5)),
]

# Matches Table 1's config (batch=16, seq=64) so the last-block point is
# directly comparable to Table 1's GPT-2 row, unlike the old sweep which
# used batch=8/seq=128 and was reported as non-comparable.
SWEEP_TRACKS = [1, 2, 6, 0]  # last-1, last-2, last-6, all layers
SWEEP_BASE = dict(backend="hf", model="gpt2", device="cuda", steps=200, warmup=10,
                   repeats=20, batch=16, seq=64, rank=8, proj_dim=512,
                   d=256, n_blocks=2, factored=False, kfac=16, dtype="fp32")


def result_path(job_name: str) -> str:
    return f"results/exp34_{job_name}.json"


def save(job_name: str, data: dict, drive_dir: str | None):
    os.makedirs("results", exist_ok=True)
    path = result_path(job_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[exp34] saved -> {path}")
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy(path, os.path.join(drive_dir, os.path.basename(path)))
        print(f"[exp34] backed up -> {drive_dir}/{os.path.basename(path)}")


def already_done(job_name: str) -> bool:
    return os.path.exists(result_path(job_name))


def run_table2_job(job_name: str, kwargs: dict, drive_dir: str | None):
    if already_done(job_name):
        print(f"[exp34] SKIP {job_name} (results/exp34_{job_name}.json already exists)")
        return
    print(f"[exp34] running {job_name}: {kwargs}")
    ns = SimpleNamespace(**kwargs)
    result = exp26.run(ns)
    save(job_name, result, drive_dir)


def run_sweep(drive_dir: str | None):
    job_name = "sweep"
    if already_done(job_name):
        print(f"[exp34] SKIP {job_name} (results/exp34_{job_name}.json already exists)")
        return
    sweep = []
    for t in SWEEP_TRACKS:
        print(f"[exp34] sweep: track={t}")
        ns = SimpleNamespace(**{**SWEEP_BASE, "track": t})
        r = exp25_run(ns)
        sweep.append({
            "track": t,
            "label": "all" if t == 0 else f"last-{t}",
            "n_tracked_layers": r["n_tracked_layers"],
            "per_sample_grad_dim": r["per_sample_grad_dim"],
            "throughput_overhead_pct": r["throughput_overhead_pct"],
            "throughput_overhead_std": r["throughput_overhead_std"],
            "base_step_ms": r["base_step_ms"],
        })
        # Save incrementally after EACH track too, in case the sweep itself
        # gets interrupted partway -- not just between jobs.
        save(f"sweep_partial_track{t}", {"completed_so_far": sweep}, drive_dir)
    save(job_name, {"batch": SWEEP_BASE["batch"], "seq": SWEEP_BASE["seq"],
                     "repeats": SWEEP_BASE["repeats"], "results": sweep}, drive_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive_dir", default=None,
                     help="e.g. /content/drive/MyDrive/traceprop_runs -- backs up each "
                          "result there immediately so a disconnect doesn't lose progress")
    ap.add_argument("--skip_table2", action="store_true")
    ap.add_argument("--skip_sweep", action="store_true")
    args = ap.parse_args()

    if not args.skip_table2:
        for job_name, kwargs in TABLE2_JOBS:
            run_table2_job(job_name, kwargs, args.drive_dir)

    if not args.skip_sweep:
        run_sweep(args.drive_dir)

    print("\n[exp34] all requested jobs complete (or already were). Summary:")
    for job_name, _ in TABLE2_JOBS:
        p = result_path(job_name)
        if os.path.exists(p):
            d = json.load(open(p))
            trak = [v for k, v in d.items() if k.startswith("speedup_vs_trak")][0]
            print(f"  {job_name:<20} flush={d['inline_flush_s']:.3f}s "
                  f"overhead={d['inline_flush_overhead_pct']:.2f}% "
                  f"posthoc={d['posthoc_pass_s']:.2f}s "
                  f"LoGRAx={d['speedup_vs_logra_1ckpt']:.1f} TRAKx={trak:.1f}")
    p = result_path("sweep")
    if os.path.exists(p):
        d = json.load(open(p))
        print("  sweep (batch16 seq64, matches Table 1):")
        for row in d["results"]:
            print(f"    {row['label']:<6} layers={row['n_tracked_layers']:<4} "
                  f"grad_dim={row['per_sample_grad_dim']:<8} "
                  f"throughput={row['throughput_overhead_pct']:.2f}±{row['throughput_overhead_std']:.2f}%")


if __name__ == "__main__":
    main()
