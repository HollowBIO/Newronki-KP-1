import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
SPLITS = {"train": DATA_DIR / "rps", "test": DATA_DIR / "rps-test-set"}

# Hash the decoded pixels, not the file bytes, so the same picture saved
# with different compression still counts as a duplicate
by_hash = defaultdict(list)
for split, split_dir in SPLITS.items():
    for path in sorted(split_dir.rglob("*.png")):
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("RGB"))
        digest = hashlib.md5(pixels.tobytes()).hexdigest()
        by_hash[digest].append((split, path.parent.name, path.name))

groups = [g for g in by_hash.values() if len(g) > 1]
within_train = [g for g in groups if {s for s, _, _ in g} == {"train"}]
within_test = [g for g in groups if {s for s, _, _ in g} == {"test"}]
train_test = [g for g in groups if {s for s, _, _ in g} == {"train", "test"}]
label_conflicts = [g for g in groups if len({c for _, c, _ in g}) > 1]

print(f"unique images: {len(by_hash)}")
print(f"duplicate groups within train: {len(within_train)}")
print(f"duplicate groups within test:  {len(within_test)}")
print(f"duplicate groups across train/test (leakage): {len(train_test)}")
print(f"duplicates with different labels: {len(label_conflicts)}")

for g in train_test + label_conflicts:
    print("  " + " | ".join(f"{s}/{c}/{n}" for s, c, n in g))
