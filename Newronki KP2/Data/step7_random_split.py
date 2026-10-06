import random
import shutil
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SOURCE_DIR = DATA_DIR / "rps"
SPLIT_DIRS = {"train": DATA_DIR / "train", "val": DATA_DIR / "val"}
VAL_FRACTION = 0.2
SEED = 42

for out_dir in SPLIT_DIRS.values():
    if out_dir.exists():
        shutil.rmtree(out_dir)

# Shuffle and split each class separately so train and val keep the 1/3 balance
rng = random.Random(SEED)
image_counts = Counter()
for class_dir in sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir()):
    files = sorted(class_dir.glob("*.png"))
    rng.shuffle(files)
    n_val = round(len(files) * VAL_FRACTION)

    for split, split_files in (("val", files[:n_val]), ("train", files[n_val:])):
        target_dir = SPLIT_DIRS[split] / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in split_files:
            shutil.copy2(path, target_dir / path.name)
        image_counts[split, class_dir.name] = len(split_files)

for split, out_dir in SPLIT_DIRS.items():
    total = sum(n for (s, _), n in image_counts.items() if s == split)
    print(f"{split} -> {out_dir.name}/  ({total} images)")
    for (s, class_name), n in sorted(image_counts.items()):
        if s == split:
            print(f"  {class_name:<9} {n:>4} images")
