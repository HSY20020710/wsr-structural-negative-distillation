from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.parse_welding_cases import write_jsonl  # noqa: E402
from src.student.data import read_jsonl, record_id  # noqa: E402


PREFIXES = (
    "质量检查记录：\n",
    "船舶焊接质量案例：\n",
)


def shifted_copy(record: dict, prefix: str, variant: int) -> dict:
    shift = len(prefix)
    augmented = json.loads(json.dumps(record, ensure_ascii=False))
    augmented["case_id"] = f"{record_id(record)}_aug{variant}"
    augmented["source_case_id"] = record_id(record)
    augmented["augmentation"] = {
        "type": "entity_preserving_prefix",
        "variant": variant,
        "prefix": prefix.strip(),
    }
    augmented["text"] = prefix + str(record["text"])
    for entity in augmented.get("entities", []):
        if isinstance(entity.get("start"), int) and isinstance(entity.get("end"), int):
            entity["start"] += shift
            entity["end"] += shift
            if augmented["text"][entity["start"]:entity["end"]] != entity.get("text"):
                raise ValueError(
                    f"{record_id(record)}: augmentation broke entity offsets"
                )
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create entity-preserving text variants from labeled Train only."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists")
    records = read_jsonl(args.input)
    output = list(records)
    for record in records:
        for variant, prefix in enumerate(PREFIXES[: args.copies], start=1):
            output.append(shifted_copy(record, prefix, variant))
    write_jsonl(args.output, output)
    manifest = {
        "source_records": len(records),
        "copies_per_record": args.copies,
        "augmented_records": len(output) - len(records),
        "total_records": len(output),
        "policy": "Train only; entity text and labels unchanged; offsets shifted.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
