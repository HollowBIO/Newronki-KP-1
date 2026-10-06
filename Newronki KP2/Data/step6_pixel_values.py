from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
SPLITS = {"train": DATA_DIR / "rps", "test": DATA_DIR / "rps-test-set"}

for split, split_dir in SPLITS.items():
    lo, hi = 255, 0
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    n_pixels = 0
    for path in split_dir.rglob("*.png"):
        with Image.open(path) as im:
            x = np.asarray(im.convert("RGB"), dtype=np.float64)
        lo, hi = min(lo, x.min()), max(hi, x.max())
        flat = x.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sq_sum += (flat ** 2).sum(axis=0)
        n_pixels += flat.shape[0]

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)
    print(f"{split}:")
    print(f"  raw range:          [{lo:.0f}, {hi:.0f}]")
    print(f"  raw mean (R, G, B): {np.round(mean, 1)}")
    print(f"  raw std  (R, G, B): {np.round(std, 1)}")
    print(f"  after /255 range:   [{lo / 255:.2f}, {hi / 255:.2f}]")
    print(f"  after /255 mean:    {np.round(mean / 255, 3)}")
