from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
SPLITS = {"train": DATA_DIR / "rps", "test": DATA_DIR / "rps-test-set"}

for split, split_dir in SPLITS.items():
    files = sorted(f for f in split_dir.rglob("*") if f.is_file())
    broken = []
    for path in files:
        try:
            with Image.open(path) as im:
                im.load()  # full decode, verify() alone skips pixel data
        except Exception as e:
            broken.append((path, e))

    print(f"{split}: checked {len(files)} files, broken: {len(broken)}")
    for path, e in broken:
        print(f"  {path.relative_to(DATA_DIR)}: {e}")
