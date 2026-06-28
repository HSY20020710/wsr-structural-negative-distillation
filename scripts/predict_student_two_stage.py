from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ontology.gate import apply_wsr_gate  # noqa: E402
from src.student.data import (  # noqa: E402
    build_entity_system_prompt,
    build_system_prompt,
    build_relation_system_prompt,
    build_relation_user_prompt,
    build_user_prompt,
    read_jsonl,
)
from src.teacher.parser import (  # noqa: E402
    extract_json_from_response,
    normalize_teacher_output,
)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def extract_any_json(response: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate((response or "").strip()):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(response.strip()[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_relations(
    parsed: dict | None,
    entities: list[dict],
    ontology: dict,
) -> tuple[list[dict], list[str]]:
    entity_ids = {entity["id"] for entity in entities}
    allowed_types = {
        item for item in ontology.get("relation_types", [])
        if item != "NO_RELATION"
    }
    relations = []
    errors = []
    seen = set()
    for index, raw in enumerate((parsed or {}).get("relations") or [], start=1):
        if not isinstance(raw, dict):
            errors.append(f"relation_{index}_not_object")
            continue
        head = str(raw.get("head") or "")
        tail = str(raw.get("tail") or "")
        relation_type = str(raw.get("type") or "").upper()
        if head not in entity_ids or tail not in entity_ids:
            errors.append(f"relation_{index}_unknown_endpoint")
            continue
        if relation_type not in allowed_types:
            errors.append(f"relation_{index}_invalid_type:{relation_type}")
            continue
        key = (head, relation_type, tail)
        if key not in seen:
            seen.add(key)
            relations.append(
                {"head": head, "tail": tail, "type": relation_type}
            )
    return relations, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage entity then fixed-entity relation prediction."
    )
    parser.add_argument("--entity_model", type=Path)
    parser.add_argument(
        "--entity_predictions",
        type=Path,
        help=(
            "Optional fixed entity predictions. When set, the script skips "
            "entity_model generation and only runs the relation model."
        ),
    )
    parser.add_argument("--relation_model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ontology", type=Path, default=ROOT / "configs/wsr_ontology.yaml"
    )
    parser.add_argument("--max_entity_tokens", type=int, default=1152)
    parser.add_argument("--max_relation_tokens", type=int, default=768)
    parser.add_argument(
        "--entity_prompt",
        choices=["joint", "entity"],
        default="joint",
        help="Use joint for an existing single-stage Ours entity model.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if not args.entity_model and not args.entity_predictions:
        raise ValueError("Set either --entity_model or --entity_predictions")
    model_dirs = [args.relation_model]
    if args.entity_model:
        model_dirs.append(args.entity_model)
    for model_dir in model_dirs:
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Incomplete model directory: {model_dir}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-student.txt first") from exc

    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    records = read_jsonl(args.test)
    entity_predictions = {}
    if args.entity_predictions:
        entity_predictions = {
            str(item["case_id"]): item
            for item in read_jsonl(args.entity_predictions)
        }
    completed = (
        {str(item["case_id"]) for item in read_jsonl(args.output)}
        if args.resume and args.output.exists()
        else set()
    )
    records = [
        record for record in records
        if str(record["case_id"]) not in completed
    ]

    entity_tokenizer = None
    if args.entity_model:
        entity_tokenizer = AutoTokenizer.from_pretrained(
            args.entity_model, trust_remote_code=True
        )
    shared_model = (
        bool(args.entity_model)
        and args.entity_model.resolve() == args.relation_model.resolve()
    )
    relation_tokenizer = (
        entity_tokenizer
        if shared_model
        else AutoTokenizer.from_pretrained(
            args.relation_model, trust_remote_code=True
        )
    )
    for tokenizer in [item for item in (entity_tokenizer, relation_tokenizer) if item]:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    entity_model = None
    if args.entity_model:
        entity_model = AutoModelForCausalLM.from_pretrained(
            args.entity_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    relation_model = (
        entity_model
        if shared_model
        else AutoModelForCausalLM.from_pretrained(
            args.relation_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    )

    def generate(model, tokenizer, system: str, prompt: str, limit: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=limit,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            output[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    for index, record in enumerate(records, start=1):
        result = {
            "case_id": record["case_id"],
            "text": record["text"],
            "method": "ours",
            "pipeline": "two_stage",
            "entity_model_path": str(args.entity_model) if args.entity_model else "",
            "entity_predictions_path": (
                str(args.entity_predictions) if args.entity_predictions else ""
            ),
            "relation_model_path": str(args.relation_model),
            "entity_raw_response": "",
            "relation_raw_response": "",
            "parse_success": False,
            "defect_label": "",
            "event_label": "",
            "entities": [],
            "relations": [],
            "triples": [],
            "errors": [],
        }
        if args.entity_predictions:
            entity_result = entity_predictions.get(str(record["case_id"]))
            if not entity_result:
                result["errors"] = ["missing_entity_prediction"]
                append_jsonl(args.output, result)
                print(
                    f"[{len(completed) + index}/{len(completed) + len(records)}] "
                    f"{record['case_id']} parse=False entities=0 relations=0",
                    flush=True,
                )
                continue
            entity_result = {
                "defect_label": entity_result.get("defect_label", ""),
                "event_label": entity_result.get("event_label", ""),
                "entities": entity_result.get("entities", []) or [],
                "relations": [],
                "triples": [],
                "errors": entity_result.get("errors", []) or [],
            }
            entity_success = bool(entity_result["entities"])
        else:
            entity_response = generate(
                entity_model,
                entity_tokenizer,
                (
                    build_system_prompt(ontology)
                    if args.entity_prompt == "joint"
                    else build_entity_system_prompt(ontology)
                ),
                build_user_prompt(record),
                args.max_entity_tokens,
            )
            result["entity_raw_response"] = entity_response
            parsed_entity = extract_json_from_response(entity_response)
            result["errors"] = list(parsed_entity["errors"])
            entity_success = parsed_entity["parse_success"]
            entity_result = {}
            if entity_success:
                entity_result = normalize_teacher_output(
                    parsed_entity["data"], record, ontology=ontology
                )
            entity_result["relations"] = []
            entity_result["triples"] = []
        if entity_success:
            fixed_record = {**record, **entity_result}
            relation_response = generate(
                relation_model,
                relation_tokenizer,
                build_relation_system_prompt(ontology),
                build_relation_user_prompt(fixed_record),
                args.max_relation_tokens,
            )
            relation_json = extract_any_json(relation_response)
            relations, relation_errors = normalize_relations(
                relation_json, entity_result["entities"], ontology
            )
            result.update(entity_result)
            result["relations"] = relations
            result["relation_raw_response"] = relation_response
            result["parse_success"] = relation_json is not None
            result["errors"] = [
                *entity_result.get("errors", []),
                *relation_errors,
            ]
            result = apply_wsr_gate(result, ontology)
        append_jsonl(args.output, result)
        print(
            f"[{len(completed) + index}/{len(completed) + len(records)}] "
            f"{record['case_id']} parse={result['parse_success']} "
            f"entities={len(result['entities'])} "
            f"relations={len(result['relations'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
