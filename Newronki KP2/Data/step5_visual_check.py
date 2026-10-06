import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
SPLITS = {"train": DATA_DIR / "rps", "test": DATA_DIR / "rps-test-set"}
PER_CLASS = 6

random.seed(42)

for split, split_dir in SPLITS.items():
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    fig, axes = plt.subplots(len(class_dirs), PER_CLASS, figsize=(2 * PER_CLASS, 2.2 * len(class_dirs)))

    for row, class_dir in enumerate(class_dirs):
        samples = random.sample(sorted(class_dir.glob("*.png")), PER_CLASS)
        for col, path in enumerate(samples):
            ax = axes[row, col]
            with Image.open(path) as im:
                ax.imshow(im.convert("RGB"))  # same alpha drop the loader does
            ax.set_title(path.name, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(class_dir.name, fontsize=12)

    fig.suptitle(f"Random {split} samples ({split_dir.name})")
    fig.tight_layout()
    out_path = DATA_DIR / f"samples_{split}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path.name}")
