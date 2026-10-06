from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
SPLITS = {"train": DATA_DIR / "rps", "test": DATA_DIR / "rps-test-set"}

for split, split_dir in SPLITS.items():
    sizes, modes = Counter(), Counter()
    uses_transparency = 0
    for path in split_dir.rglob("*.png"):
        with Image.open(path) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
            # Converting to RGB drops alpha, so transparent pixels would show
            # whatever colour is stored under them
            if im.mode == "RGBA" and np.asarray(im)[..., 3].min() < 255:
                uses_transparency += 1

    print(f"{split}:")
    print(f"  sizes (w, h): {dict(sizes)}")
    print(f"  colour modes: {dict(modes)}")
    print(f"  images with any transparent pixels: {uses_transparency}")
