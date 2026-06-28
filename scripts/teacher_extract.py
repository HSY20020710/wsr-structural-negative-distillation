from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ontology.gate import apply_wsr_gate  # noqa: E402
from src.teacher.parser import (  # noqa: E402
    extract_json_from_response,
    normalize_teacher_output,
)
from src.teacher.prompt_templates import (  # noqa: E402
    build_with_wsr_ontology_prompt,
    build_without_ontology_prompt,
)
from src.teacher.qwen_client import (  # noqa: E402
    QwenClientConfig,
    QwenGenerateClient,
)


MODES = {
    "without_ontology",
    "with_wsr_ontology",
    "with_wsr_ontology_gate",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("case_id"):
                completed.add(record["case_id"])
    return completed


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def empty_result(record: dict, args: argparse.Namespace) -> dict:
    return {
        **{
            key: record.get(key, "")
            for key in [
                "case_id",
                "original_case_no",
                "occurrence_time",
                "occurrence_stage",
                "problem_phenomenon",
                "cause_analysis",
                "text",
            ]
        },
        "mode": args.mode,
        "model": args.model,
        "raw_response": "",
        "raw_api_response": None,
        "parse_success": False,
        "defect_label": "",
        "event_label": "",
        "entities": [],
        "relations": [],
        "triples": [],
        "causal_chain": [],
        "teacher_probs": {"defect": {}, "event": {}, "triple_validity": {}},
        "rejected_relations": [],
        "gate_stats": {
            "predicted_relations_before_gate": 0,
            "predicted_relations_after_gate": 0,
            "rejected_total": 0,
            "rejected_wuli": 0,
            "rejected_shili": 0,
            "rejected_renli": 0,
            "rejected_multi_conflict": 0,
        },
        "errors": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.6-27B teacher extraction in one of three settings."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument(
        "--api_url",
        default="http://localhost:8000V1/api/generate",
    )
    parser.add_argument(
        "--protocol", choices=["auto", "ollama", "openai"], default="auto"
    )
    parser.add_argument(
        "--ontology", type=Path, default=ROOT / "configs" / "wsr_ontology.yaml"
    )
    parser.add_argument(
        "--source_predictions",
        type=Path,
        help="For gate mode, derive output from with_wsr_ontology predictions.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout_seconds", type=int, default=180)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=2.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    ontology = None
    if args.mode != "without_ontology":
        ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    records = read_jsonl(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output} already exists. Use --resume or choose a new output."
        )
    completed = load_completed_case_ids(args.output) if args.resume else set()
    if args.mode == "with_wsr_ontology_gate" and args.source_predictions:
        source_records = read_jsonl(args.source_predictions)
        source_by_id = {item["case_id"]: item for item in source_records}
        pending = [record for record in records if record["case_id"] not in completed]
        for index, record in enumerate(pending, start=1):
            source = source_by_id.get(record["case_id"])
            if not source:
                result = empty_result(record, args)
                result["errors"].append("missing_source_prediction")
            elif not source.get("parse_success"):
                result = deepcopy(source)
                result["mode"] = args.mode
                result["errors"] = list(result.get("errors", [])) + [
                    "source_prediction_parse_failed"
                ]
            else:
                result = apply_wsr_gate(deepcopy(source), ontology)
                result["mode"] = args.mode
            append_jsonl(args.output, result)
            print(
                f"[{index}/{len(pending)}] {record['case_id']} "
                f"parse_success={result['parse_success']} "
                f"entities={len(result['entities'])} "
                f"relations={len(result['relations'])} "
                f"rejected={len(result['rejected_relations'])} "
                f"errors={len(result['errors'])}"
            )
        return
    client = QwenGenerateClient(
        QwenClientConfig(
            api_url=args.api_url,
            model=args.model,
            protocol=args.protocol,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
            max_tokens=args.max_tokens,
        )
    )
    if client.url_warning:
        print(f"WARNING: {client.url_warning}")

    pending = [record for record in records if record["case_id"] not in completed]
    for index, record in enumerate(pending, start=1):
        result = empty_result(record, args)
        prompt = (
            build_without_ontology_prompt(record)
            if args.mode == "without_ontology"
            else build_with_wsr_ontology_prompt(record, ontology)
        )
        response = client.generate(prompt)
        result["raw_api_response"] = response.get("data")
        result["raw_response"] = response.get("response_text", "")
        if not response["success"]:
            result["errors"].append(f"request_failed:{response.get('error')}")
        else:
            parsed = extract_json_from_response(result["raw_response"])
            result["errors"].extend(parsed["errors"])
            if parsed["parse_success"]:
                normalized = normalize_teacher_output(
                    parsed["data"], record, ontology=ontology
                )
                result.update(
                    {
                        key: normalized[key]
                        for key in [
                            "defect_label",
                            "event_label",
                            "entities",
                            "relations",
                            "triples",
                            "causal_chain",
                            "teacher_probs",
                        ]
                    }
                )
                result["errors"].extend(normalized["errors"])
                result["parse_success"] = True
                if args.mode == "with_wsr_ontology_gate":
                    result = apply_wsr_gate(result, ontology)
        append_jsonl(args.output, result)
        print(
            f"[{index}/{len(pending)}] {record['case_id']} "
            f"parse_success={result['parse_success']} "
            f"entities={len(result['entities'])} "
            f"relations={len(result['relations'])} "
            f"rejected={len(result['rejected_relations'])} "
            f"errors={len(result['errors'])}"
        )


if __name__ == "__main__":
    main()
