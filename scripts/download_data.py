"""Download the NEU-DET parquet file from Hugging Face and verify its SHA-256."""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from al_steel.data import DATA_PATH, DATA_SHA256, DATA_URL  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    digest = sha256(DATA_PATH)
    if digest != DATA_SHA256:
        sys.exit(f"Checksum mismatch for {DATA_PATH}: {digest}")
    print(f"OK {DATA_PATH} ({DATA_PATH.stat().st_size / 1e6:.1f} MB)")
