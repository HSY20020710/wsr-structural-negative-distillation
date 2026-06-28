from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.parse_welding_cases import parse_welding_cases, write_jsonl  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    char_ngrams,
    exact_deduplicate,
    group_near_duplicates,
    jaccard,
    normalize_for_dedup,
)
from src.student.data import read_jsonl, record_id  # noqa: E402


def text_group(record: dict) -> str:
    normalized = " ".join(str(record.get("text", "")).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def uniquify_case_ids(records: list[dict], protected_ids: set[str]) -> list[dict]:
    seen: dict[str, int] = {}
    output = []
    for record in records:
        base_id = record_id(record)
        seen[base_id] = seen.get(base_id, 0) + 1
        occurrence = seen[base_id]
        if occurrence == 1:
            output.append(record)
            continue
        if base_id in protected_ids:
            raise ValueError(
                f"Raw corpus contains duplicate protected test id {base_id}"
            )
        output.append(
            {
                **record,
                "case_id": f"{base_id}_dup{occurrence}",
                "source_case_id": base_id,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-free raw student train/dev candidates."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--test_gold", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dev_size", type=int, default=50)
    parser.add_argument("--near_duplicate_threshold", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_records = read_jsonl(args.test_gold)
    test_ids = {record_id(record) for record in test_records}
    records, parse_stats = parse_welding_cases(args.raw)
    records = uniquify_case_ids(records, test_ids)
    all_ids = {record_id(record) for record in records}
    missing_test_ids = sorted(test_ids - all_ids)
    if missing_test_ids:
        raise ValueError(
            f"{len(missing_test_ids)} test ids are absent from the raw corpus; "
            f"first={missing_test_ids[:5]}"
        )

    candidates = [
        record
        for record in records
        if record_id(record) not in test_ids
    ]
    test_ngrams = [
        char_ngrams(normalize_for_dedup(str(record.get("text", ""))))
        for record in test_records
    ]
    candidate_rows = []
    excluded_test_near_duplicates = []
    for record in candidates:
        dedup_text = normalize_for_dedup(str(record.get("text", "")))
        ngrams = char_ngrams(dedup_text)
        if any(
            jaccard(ngrams, test_tokens) >= args.near_duplicate_threshold
            for test_tokens in test_ngrams
        ):
            excluded_test_near_duplicates.append(record)
            continue
        candidate_rows.append(
            {
                **record,
                "record_id": record_id(record),
                "dedup_text": dedup_text,
                "source_format": record.get("source_format", "legacy"),
                "year": record.get("year"),
                "project": record.get("project", ""),
            }
        )
    unique_candidates, exact_duplicates = exact_deduplicate(candidate_rows)
    grouped_candidates = group_near_duplicates(
        unique_candidates, threshold=args.near_duplicate_threshold
    )
    groups: dict[str, list[dict]] = {}
    for record in grouped_candidates:
        groups.setdefault(record["group_id"], []).append(record)
    grouped = list(groups.values())
    random.Random(args.seed).shuffle(grouped)

    dev_groups = []
    train_groups = []
    dev_count = 0
    for group in grouped:
        if dev_count < args.dev_size:
            dev_groups.append(group)
            dev_count += len(group)
        else:
            train_groups.append(group)
    dev = [record for group in dev_groups for record in group]
    train = [record for group in train_groups for record in group]

    train_ids = {record_id(record) for record in train}
    dev_ids = {record_id(record) for record in dev}
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise AssertionError("Student split leakage detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "raw_train.jsonl", train)
    write_jsonl(args.output_dir / "raw_dev.jsonl", dev)
    write_jsonl(args.output_dir / "test.jsonl", test_records)
    manifest = {
        "raw_cases": len(records),
        "unique_raw_case_ids": len({record_id(record) for record in records}),
        "candidate_cases_before_dedup": len(candidates),
        "excluded_test_near_duplicates": len(excluded_test_near_duplicates),
        "exact_duplicates_removed": len(exact_duplicates),
        "near_duplicate_groups": len(groups),
        "train_candidates": len(train),
        "dev_candidates": len(dev),
        "fixed_test": len(test_records),
        "seed": args.seed,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "parse_stats": parse_stats,
        "required_next_files": {
            "teacher_train.jsonl": "Run the complete teacher system on raw_train.jsonl",
            "gold_train.jsonl": "Independently annotate a human-supervised subset",
            "dev.jsonl": "Human-label raw_dev.jsonl before model selection",
            "pseudo_train.jsonl": "Generate with the supervised student for self-training",
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
