"""Extract and cache frozen backbone features for every image (features/{backbone}.npz)."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from al_steel.data import load_neu  # noqa: E402
from al_steel.features import BACKBONES, extract_features  # noqa: E402

if __name__ == "__main__":
    out_dir = ROOT / "features"
    out_dir.mkdir(exist_ok=True)
    images, labels = load_neu()
    for name in sys.argv[1:] or BACKBONES:
        out = out_dir / f"{name}.npz"
        if out.exists():
            print(f"skip {name} (cached)")
            continue
        t0 = time.time()
        feats = extract_features(images, name)
        np.savez_compressed(out, features=feats, labels=labels)
        print(f"{name}: {feats.shape} in {time.time() - t0:.0f} s -> {out}")
