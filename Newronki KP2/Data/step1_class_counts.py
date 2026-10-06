from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SPLITS = {"train": DATA_DIR / "rps", "test": DATA_DIR / "rps-test-set"}

for split, split_dir in SPLITS.items():
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    counts = {d.name: sum(1 for f in d.iterdir() if f.is_file()) for d in class_dirs}
    total = sum(counts.values())

    print(f"{split} ({split_dir.name}):")
    for name, n in counts.items():
        print(f"  {name:<10} {n:>5}  ({n / total:.1%})")
    print(f"  {'total':<10} {total:>5}")

    extensions = Counter(f.suffix.lower() for d in class_dirs for f in d.iterdir() if f.is_file())
    print(f"  file types: {dict(extensions)}")

    stray = [p.name for p in split_dir.iterdir() if p.is_file()]
    if stray:
        print(f"  files outside class folders (ignored by loader): {stray}")
    print()
