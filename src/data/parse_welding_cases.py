from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path


CASE_RE = re.compile(r"(?m)^\s*案例\s*(\d+)\s*$")
ANNUAL_CASE_RE = re.compile(r"(?m)^【(\d{4})年-案例(\d+)】\s*$")
ANNUAL_FIELD_RE = re.compile(r"^•\s*([^：:]+)[：:]\s*(.*)$")
WELDING_TERMS = (
    "焊",
    "坡口",
    "气孔",
    "裂纹",
    "夹渣",
    "未熔合",
    "未焊透",
    "咬边",
    "焊材",
    "焊条",
    "焊丝",
    "焊脚",
    "焊瘤",
    "引弧",
    "熄弧",
    "电弧",
    "弧板",
    "WPS",
    "探伤",
)
SECTION_PATTERNS = {
    "occurrence_time": re.compile(r"^\s*(?:一[、.]?)?\s*发生时间\s*$"),
    "occurrence_stage": re.compile(r"^\s*(?:二[、.]?)?\s*发生阶段\s*$"),
    "problem_phenomenon": re.compile(r"^\s*(?:三[、.]?)?\s*问题现象\s*$"),
    "cause_analysis": re.compile(
        r"^\s*(?:四[、.]?)?\s*问题原因(?:分析)?\s*$"
    ),
}
FIELD_ORDER = [
    "occurrence_time",
    "occurrence_stage",
    "problem_phenomenon",
    "cause_analysis",
]


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("\ufeff", "").replace("\u200b", "")
    value = value.replace("\u00a0", " ").replace("\u3000", " ")
    lines = []
    for line in value.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def build_teacher_text(record: dict) -> str:
    return "\n".join(
        [
            f"发生时间：{record['occurrence_time']}",
            f"发生阶段：{record['occurrence_stage']}",
            f"问题现象：{record['problem_phenomenon']}",
            f"问题原因分析：{record['cause_analysis']}",
        ]
    ).strip()


def _parse_case_block(case_number: str, block: str) -> dict:
    values = {field: [] for field in FIELD_ORDER}
    active_field = None
    for raw_line in block.splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        matched_field = next(
            (
                field
                for field, pattern in SECTION_PATTERNS.items()
                if pattern.fullmatch(line)
            ),
            None,
        )
        if matched_field:
            active_field = matched_field
            continue
        if active_field:
            values[active_field].append(line)

    record = {
        "case_id": f"case_{case_number}",
        "original_case_no": f"案例{case_number}",
        **{
            field: clean_text("\n".join(values[field])).replace("\n", " ")
            for field in FIELD_ORDER
        },
    }
    record["text"] = build_teacher_text(record)
    warnings = []
    if not record["problem_phenomenon"]:
        warnings.append("missing_problem_phenomenon")
    if not record["cause_analysis"]:
        warnings.append("missing_cause_analysis")
    record["warnings"] = warnings
    return record


def _parse_annual_cases(source: str) -> list[dict]:
    matches = list(ANNUAL_CASE_RE.finditer(source))
    records = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        block = source[match.end():end].split("##########", 1)[0]
        fields: dict[str, str] = {}
        active_field = None
        for raw_line in block.splitlines():
            line = clean_text(raw_line)
            field_match = ANNUAL_FIELD_RE.match(line)
            if field_match:
                active_field = field_match.group(1).strip()
                fields[active_field] = field_match.group(2).strip()
            elif line and active_field:
                fields[active_field] = clean_text(
                    f"{fields[active_field]} {line}"
                ).replace("\n", " ")
        description = fields.get("问题简述", "")
        if not description or not any(term.lower() in description.lower() for term in WELDING_TERMS):
            continue
        year, case_number = match.groups()
        record = {
            "case_id": f"annual_{year}_{case_number}",
            "original_case_no": f"【{year}年-案例{case_number}】",
            "source_format": "annual",
            "year": int(year),
            "occurrence_time": fields.get("发生时间", ""),
            "occurrence_stage": fields.get("项目类型", ""),
            "problem_phenomenon": description,
            "cause_analysis": "",
            "project": fields.get("工程项目", ""),
            "problem_category": fields.get("问题分类", ""),
            "responsible_department": fields.get("责任部门", ""),
            "team": fields.get("科室/班组", ""),
            "contractor": fields.get("工程承包商", ""),
            "warnings": ["missing_cause_analysis"],
        }
        record["text"] = build_teacher_text(record)
        records.append(record)
    return records


def parse_welding_cases(path: Path, limit: int = 0) -> tuple[list[dict], dict]:
    source = path.read_text(encoding="utf-8-sig")
    matches = list(CASE_RE.finditer(source))
    records = []
    empty_text_cases = []
    missing_phenomenon_cases = []
    missing_cause_cases = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        block = source[match.end():end]
        # The source file also contains a newer 【year-case】 dataset after legacy cases.
        newer_format = re.search(r"(?m)^【\d{4}年-案例\d+】", block)
        if newer_format:
            block = block[: newer_format.start()]
        record = _parse_case_block(match.group(1), block)
        meaningful = any(record[field] for field in FIELD_ORDER)
        if not meaningful:
            empty_text_cases.append(record["case_id"])
            continue
        if not record["problem_phenomenon"]:
            missing_phenomenon_cases.append(record["case_id"])
        if not record["cause_analysis"]:
            missing_cause_cases.append(record["case_id"])
        records.append(record)
        if limit > 0 and len(records) >= limit:
            break
    annual_records = _parse_annual_cases(source)
    if limit > 0:
        annual_records = annual_records[: max(0, limit - len(records))]
    records.extend(annual_records)
    stats = {
        "legacy_headers": len(matches),
        "annual_headers": len(list(ANNUAL_CASE_RE.finditer(source))),
        "annual_welding_cases": len(annual_records),
        "parsed_cases": len(records),
        "valid_cases": len(records),
        "empty_text_cases": empty_text_cases,
        "missing_phenomenon_cases": missing_phenomenon_cases,
        "missing_cause_cases": missing_cause_cases,
        "filter_policy": "Annual records require an explicit welding term.",
    }
    return records, stats


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "original_case_no",
        "occurrence_time",
        "occurrence_stage",
        "problem_phenomenon",
        "cause_analysis",
        "text",
        "warnings",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["warnings"] = "|".join(record.get("warnings", []))
            writer.writerow(row)
