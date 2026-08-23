"""Import an authorized private data pack into data/private without committing it."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", help="Path to an authorized private data pack.")
    parser.add_argument("--dest", default="data/private")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        archive.extractall(dest)

    counts = {path.name: count_jsonl(path) for path in dest.rglob("*.jsonl")}
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("Private data imported locally. Keep data/private ignored by git.")


if __name__ == "__main__":
    main()