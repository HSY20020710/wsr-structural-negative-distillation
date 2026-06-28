from __future__ import annotations

import hashlib
import html
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FIELD_RE = re.compile(r"^•\s*([^：]+)：(.*)$")
OLD_CASE_RE = re.compile(r"(?m)^案例\s*(\d+)\s*$")
NEW_CASE_RE = re.compile(r"(?m)^【(\d{4})年-案例(\d+)】\s*$")


def normalize_space(value: str) -> str:
    value = html.unescape(value).replace("\u3000", " ").replace("\xa0", " ")
    value = value.replace("&rdquo;", '"').replace("&ldquo;", '"')
    return re.sub(r"\s+", " ", value).strip()


def normalize_for_dedup(value: str) -> str:
    value = normalize_space(value).lower()
    value = re.sub(r"\d{4}[-年/.]\d{1,2}(?:[-月/.]\d{1,2}日?)?", "<date>", value)
    value = re.sub(r"\d+(?:\.\d+)?", "<num>", value)
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "", value)
    return value


def stable_id(prefix: str, value: str, length: int = 12) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def _extract_between(block: str, start: str, end: str | None) -> str:
    if start not in block:
        return ""
    value = block.split(start, 1)[1]
    if end and end in value:
        value = value.split(end, 1)[0]
    return normalize_space(value)


def parse_old_cases(text: str) -> list[dict]:
    matches = list(OLD_CASE_RE.finditer(text))
    records = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():end]
        if NEW_CASE_RE.search(block):
            block = block[: NEW_CASE_RE.search(block).start()]
        occurrence_time = _extract_between(block, "一、发生时间", "二、发生阶段")
        stage = _extract_between(block, "二、发生阶段", "三、问题现象")
        phenomenon = _extract_between(block, "三、问题现象", "四、问题原因分析")
        cause = _extract_between(block, "四、问题原因分析", None)
        if not phenomenon and not cause:
            continue
        case_number = match.group(1)
        full_text = "；".join(part for part in [phenomenon, cause] if part)
        year_match = re.search(r"(20\d{2})", occurrence_time)
        records.append(
            {
                "source_format": "legacy",
                "source_case_id": f"legacy-{case_number}",
                "year": int(year_match.group(1)) if year_match else None,
                "occurrence_time": occurrence_time,
                "stage": stage,
                "project_type": "",
                "project": "",
                "problem_category": "",
                "phenomenon": phenomenon,
                "cause": cause,
                "text": full_text,
                "responsible_department": "",
                "team": "",
                "contractor": "",
            }
        )
    return records


def parse_new_cases(text: str) -> list[dict]:
    matches = list(NEW_CASE_RE.finditer(text))
    records = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end].split("##########", 1)[0]
        fields: dict[str, str] = {}
        current_key = None
        for raw_line in block.splitlines():
            line = raw_line.strip()
            field_match = FIELD_RE.match(line)
            if field_match:
                current_key = field_match.group(1).strip()
                fields[current_key] = field_match.group(2).strip()
            elif line and current_key:
                fields[current_key] = f"{fields[current_key]}\n{line}".strip()
        fields = {key: normalize_space(value) for key, value in fields.items()}
        description = fields.get("问题简述", "")
        if not description:
            continue
        header_year, case_number = match.groups()
        occurrence_time = fields.get("发生时间", "")
        year_match = re.search(r"(20\d{2})", occurrence_time)
        records.append(
            {
                "source_format": "structured",
                "source_case_id": f"{header_year}-{case_number}",
                "year": int(year_match.group(1)) if year_match else int(header_year),
                "occurrence_time": occurrence_time,
                "stage": "",
                "project_type": fields.get("项目类型", ""),
                "project": fields.get("工程项目", ""),
                "problem_category": fields.get("问题分类", ""),
                "phenomenon": description,
                "cause": "",
                "text": description,
                "responsible_department": fields.get("责任部门", ""),
                "team": fields.get("科室/班组", ""),
                "contractor": fields.get("工程承包商", ""),
                "created_date": fields.get("创建日期", ""),
            }
        )
    return records


def parse_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    records = parse_old_cases(text) + parse_new_cases(text)
    for record in records:
        record["text"] = normalize_space(record["text"])
        record["record_id"] = stable_id(
            "case", f"{record['source_format']}|{record['source_case_id']}|{record['text']}"
        )
        record["dedup_text"] = normalize_for_dedup(record["text"])
    return records


def exact_deduplicate(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    duplicates: list[dict] = []
    owner_by_text: dict[str, str] = {}
    for record in records:
        key = record["dedup_text"]
        if key and key in owner_by_text:
            duplicate = dict(record)
            duplicate["duplicate_of"] = owner_by_text[key]
            duplicates.append(duplicate)
        else:
            owner_by_text[key] = record["record_id"]
            kept.append(record)
    return kept, duplicates


def char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[index:index + n] for index in range(len(text) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _blocking_keys(record: dict) -> set[str]:
    keys = set()
    project = normalize_for_dedup(record.get("project", ""))
    if project:
        keys.add(f"project:{project}")
    text = record["dedup_text"]
    for token in re.findall(r"[\u4e00-\u9fff]{4,}", record["text"])[:8]:
        keys.add(f"token:{token[:8]}")
    if not keys and text:
        keys.add(f"prefix:{text[:18]}")
    return keys


def group_near_duplicates(records: list[dict], threshold: float) -> list[dict]:
    ngrams = [char_ngrams(record["dedup_text"]) for record in records]
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for key in _blocking_keys(record):
            blocks[key].append(index)
    union_find = UnionFind(len(records))
    compared: set[tuple[int, int]] = set()
    for members in blocks.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                pair = (min(left, right), max(left, right))
                if pair in compared:
                    continue
                compared.add(pair)
                if jaccard(ngrams[left], ngrams[right]) >= threshold:
                    union_find.union(left, right)
    grouped = []
    root_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        root_members[union_find.find(index)].append(index)
    for indices in root_members.values():
        member_ids = sorted(records[index]["record_id"] for index in indices)
        group_id = stable_id("group", "|".join(member_ids))
        for index in indices:
            record = dict(records[index])
            record["group_id"] = group_id
            record["near_duplicate_group_size"] = len(indices)
            grouped.append(record)
    return grouped


@dataclass(frozen=True)
class SplitRatios:
    train: float
    dev: float
    test: float


def grouped_split(
    records: list[dict], ratios: SplitRatios, seed: int
) -> tuple[dict[str, list[dict]], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    result: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    strata: dict[tuple[str, str], list[tuple[str, list[dict]]]] = defaultdict(list)
    for group_id, members in groups.items():
        source = Counter(item["source_format"] for item in members).most_common(1)[0][0]
        years = [str(item["year"]) for item in members if item.get("year")]
        year = Counter(years).most_common(1)[0][0] if years else "unknown"
        strata[(source, year)].append((group_id, members))
    for stratum_index, group_items in enumerate(sorted(strata.values(), key=len, reverse=True)):
        randomizer = random.Random(seed + stratum_index)
        randomizer.shuffle(group_items)
        group_items.sort(key=lambda item: len(item[1]), reverse=True)
        stratum_size = sum(len(members) for _, members in group_items)
        target = {
            "train": stratum_size * ratios.train,
            "dev": stratum_size * ratios.dev,
            "test": stratum_size * ratios.test,
        }
        stratum_counts = {"train": 0, "dev": 0, "test": 0}
        for _, members in group_items:
            split = min(
                result,
                key=lambda name: (
                    stratum_counts[name] / max(target[name], 1),
                    stratum_counts[name],
                    name,
                ),
            )
            stratum_counts[split] += len(members)
            for record in members:
                item = dict(record)
                item["split"] = split
                result[split].append(item)
    report = {
        "total_records": len(records),
        "total_groups": len(groups),
        "seed": seed,
        "stratification": ["source_format", "year"],
        "ratios": ratios.__dict__,
        "splits": {
            name: {
                "records": len(items),
                "groups": len({item["group_id"] for item in items}),
                "source_formats": dict(Counter(item["source_format"] for item in items)),
                "years": dict(
                    sorted(Counter(str(item["year"]) for item in items if item["year"]).items())
                ),
            }
            for name, items in result.items()
        },
    }
    return result, report


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
