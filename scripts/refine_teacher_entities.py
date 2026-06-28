from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.teacher.entity_refinement import (  # noqa: E402
    build_entity_refinement_prompt,
    filter_consensus_entities,
    parse_refined_entities,
    remap_relations,
)
from src.ontology.gate import apply_wsr_gate  # noqa: E402
from src.teacher.qwen_client import QwenClientConfig, QwenGenerateClient  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        item["case_id"]
        for item in read_jsonl(path)
        if item.get("case_id")
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine first-pass teacher entities with a second Qwen pass."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ontology", type=Path, default=ROOT / "configs" / "wsr_ontology.yaml"
    )
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument(
        "--mode",
        default="with_wsr_ontology_gate_enhanced",
        help="Mode recorded in each output row.",
    )
    parser.add_argument(
        "--api_url",
        default="http://localhost:8000/v1/chat/completions",
    )
    parser.add_argument("--protocol", choices=["auto", "ollama", "openai"], default="openai")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout_seconds", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output} already exists. Use --resume or choose another path."
        )
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    records = read_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    done = completed_ids(args.output) if args.resume else set()
    pending = [record for record in records if record["case_id"] not in done]
    client = QwenGenerateClient(
        QwenClientConfig(
            api_url=args.api_url,
            model=args.model,
            protocol=args.protocol,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
        )
    )
    for index, record in enumerate(pending, start=1):
        response = client.generate(build_entity_refinement_prompt(record, ontology))
        parsed = (
            parse_refined_entities(response.get("response_text", ""), record, ontology)
            if response["success"]
            else {
                "parse_success": False,
                "entities": [],
                "errors": [f"request_failed:{response.get('error')}"],
            }
        )
        refined_entities = (
            filter_consensus_entities(
                parsed["entities"], record.get("entities", [])
            )
            if parsed["parse_success"]
            else record.get("entities", [])
        )
        remapped_relations = remap_relations(
            record.get("entities", []),
            record.get("relations", []),
            refined_entities,
        )
        result = {
            **record,
            "mode": args.mode,
            "model": args.model,
            "pipeline_components": [
                "wsr_ontology",
                "entity_refinement_second_pass",
                "entity_consensus_filter",
                "relation_endpoint_remapping",
                "wsr_gate",
            ],
            "first_pass_entities": record.get("entities", []),
            "first_pass_relations": record.get("relations", []),
            "entities_before_consensus": parsed["entities"],
            "entities": refined_entities,
            "relations": remapped_relations,
            "triples": [],
            "parse_success": bool(record.get("parse_success")) and parsed["parse_success"],
            "entity_refinement_success": parsed["parse_success"],
            "entity_refinement_errors": parsed["errors"],
            "entity_refinement_raw_response": response.get("response_text", ""),
        }
        result.pop("relations_before_gate", None)
        result = apply_wsr_gate(result, ontology)
        append_jsonl(args.output, result)
        print(
            f"[{index}/{len(pending)}] {record['case_id']} "
            f"parse={result['entity_refinement_success']} "
            f"entities={len(result['entities'])} errors={len(parsed['errors'])}"
        )


if __name__ == "__main__":
    main()
