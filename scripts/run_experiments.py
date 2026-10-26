"""Run every (backbone, strategy, seed) active learning curve and the full-data upper bound.

Each finished (backbone, strategy) block is saved to results/parts/, so an interrupted run resumes
where it stopped. At the end everything is merged into results/curves.csv, results/full_data.csv and
results/picks_seed0.json.
"""
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from al_steel.data import split_indices  # noqa: E402
from al_steel.features import BACKBONES  # noqa: E402
from al_steel.loop import STRATEGIES, full_data_accuracy, run_active_learning  # noqa: E402

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
warnings.filterwarnings("ignore", category=UserWarning)

SEEDS = range(5)
CONFIG = {"init": 6, "query": 6, "rounds": 15, "mc_iter": 20, "epochs": 60}

if __name__ == "__main__":
    out = ROOT / "results"
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for bb in BACKBONES:
        d = np.load(ROOT / "features" / f"{bb}.npz")
        feats, labels = d["features"], d["labels"]
        train_idx, test_idx = split_indices(labels)
        full_part = parts / f"{bb}__full.csv"
        if not full_part.exists():
            pd.DataFrame([{"backbone": bb, "seed": seed,
                           "accuracy": full_data_accuracy(feats, labels, train_idx, test_idx, seed)}
                          for seed in SEEDS]).to_csv(full_part, index=False)
        for strategy in STRATEGIES:
            part = parts / f"{bb}__{strategy}.csv"
            if part.exists():
                print(f"skip {bb} {strategy} (done)", flush=True)
                continue
            t0, rows_all, picks_seed0 = time.time(), [], None
            for seed in SEEDS:
                rows, picks = run_active_learning(feats, labels, train_idx, test_idx, strategy, seed, **CONFIG)
                rows_all += [{"backbone": bb, **r} for r in rows]
                if seed == 0:
                    picks_seed0 = [[int(train_idx[i]) for i in p] for p in picks]
            df = pd.DataFrame(rows_all)
            (parts / f"{bb}__{strategy}.picks.json").write_text(json.dumps(picks_seed0))
            df.to_csv(part, index=False)
            last = df[df["round"] == CONFIG["rounds"]].accuracy
            print(f"{bb:14s} {strategy:9s} final acc {last.mean():.3f} ± {last.std():.3f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)

    pd.concat([pd.read_csv(p) for p in sorted(parts.glob("*__full.csv"))]).to_csv(out / "full_data.csv", index=False)
    curve_parts = [p for p in sorted(parts.glob("*.csv")) if not p.name.endswith("__full.csv")]
    pd.concat([pd.read_csv(p) for p in curve_parts]).to_csv(out / "curves.csv", index=False)
    picks = {p.name.split(".")[0].replace("__", "/"): json.loads(p.read_text()) for p in parts.glob("*.picks.json")}
    (out / "picks_seed0.json").write_text(json.dumps({"config": CONFIG, "picks": picks}))
    print("done", flush=True)
