"""Loading and splitting the NEU-DET steel surface defect dataset."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_URL = "https://huggingface.co/datasets/newguyme/neu_det_caption/resolve/main/data/train-00000-of-00001.parquet"
DATA_SHA256 = "01cf9baad86e64f802b29e99f3773880a574f88957175392801e9da0aabd07ad"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "neu_det.parquet"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def load_neu(path=DATA_PATH):
    """Return (images as list of RGB PIL images, integer labels as np.ndarray)."""
    df = pd.read_parquet(path, columns=["image", "label", "label_str"])
    assert sorted(df.label_str.unique()) == sorted(CLASSES), "unexpected class names"
    images = [Image.open(io.BytesIO(b["bytes"])).convert("RGB") for b in df.image]
    return images, df.label.to_numpy()


def split_indices(labels, test_size=0.2, seed=0):
    """Fixed stratified train/test split. The test set is never queried by active learning."""
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    return np.sort(train_idx), np.sort(test_idx)
