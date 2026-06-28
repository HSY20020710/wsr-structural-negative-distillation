from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.parse_welding_cases import write_jsonl  # noqa: E402
from src.ontology.gate import apply_wsr_gate  # noqa: E402
from src.student.data import (  # noqa: E402
    build_system_prompt,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate filtered pseudo labels with the supervised student."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ontology", type=Path, default=ROOT / "configs/wsr_ontology.yaml"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reject_output",
        type=Path,
        help="Optional JSONL path for rejected raw generations.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} already exists")
    if not args.model.is_dir():
        raise FileNotFoundError(
            f"Student model directory does not exist: {args.model.resolve()}. "
            "Train qwen3_supervised first or pass its actual final directory."
        )
    required_model_files = [
        args.model / "config.json",
        args.model / "tokenizer_config.json",
    ]
    missing_model_files = [
        str(path) for path in required_model_files if not path.is_file()
    ]
    if missing_model_files:
        raise FileNotFoundError(
            f"Incomplete full-model directory {args.model.resolve()}; "
            f"missing: {missing_model_files}"
        )

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements-student.txt first") from exc

    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    records = read_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    completed_ids = (
        {str(record["case_id"]) for record in read_jsonl(args.output)}
        if args.resume and args.output.exists()
        else set()
    )
    records = [
        record for record in records if str(record["case_id"]) not in completed_ids
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    class CompleteJsonStoppingCriteria(StoppingCriteria):
        def __init__(self, prompt_width: int) -> None:
            self.prompt_width = prompt_width

        @staticmethod
        def is_complete_json(text: str) -> bool:
            start = text.find("{")
            if start < 0:
                return False
            depth = 0
            in_string = False
            escaped = False
            for char in text[start:]:
                if escaped:
                    escaped = False
                    continue
                if char == "\\" and in_string:
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return True
            return False

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return all(
                self.is_complete_json(
                    tokenizer.decode(
                        row[self.prompt_width :], skip_special_tokens=True
                    )
                )
                for row in input_ids
            )

    accepted = []
    failures = []
    total = len(records)
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        prompts = []
        for record in batch:
            messages = [
                {"role": "system", "content": build_system_prompt(ontology)},
                {"role": "user", "content": build_user_prompt(record)},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [CompleteJsonStoppingCriteria(encoded["input_ids"].shape[1])]
                ),
            )
        prompt_width = encoded["input_ids"].shape[1]
        for record, tokens in zip(batch, generated):
            response = tokenizer.decode(
                tokens[prompt_width:], skip_special_tokens=True
            )
            parsed = extract_json_from_response(response)
            if not parsed["parse_success"]:
                failure = {
                    "case_id": record["case_id"],
                    "reason": "parse_error",
                    "errors": parsed["errors"],
                    "raw_response": response,
                }
                failures.append(failure)
                if args.reject_output:
                    append_jsonl(args.reject_output, failure)
                print(
                    f"[{min(start + args.batch_size, total)}/{total}] "
                    f"{record['case_id']} rejected=parse_error",
                    flush=True,
                )
                continue
            normalized = normalize_teacher_output(
                parsed["data"], record, ontology=ontology
            )
            candidate = apply_wsr_gate(
                {
                    **record,
                    **normalized,
                    "parse_success": True,
                    "source": "supervised_student_pseudo",
                    "raw_response": response,
                },
                ontology,
            )
            if not candidate["entities"]:
                failure = {
                    "case_id": record["case_id"],
                    "reason": "no_valid_entities",
                    "errors": normalized.get("errors", []),
                    "parsed_data": parsed["data"],
                    "raw_response": response,
                }
                failures.append(failure)
                if args.reject_output:
                    append_jsonl(args.reject_output, failure)
                print(
                    f"[{min(start + args.batch_size, total)}/{total}] "
                    f"{record['case_id']} rejected=no_valid_entities",
                    flush=True,
                )
                continue
            accepted.append(candidate)
            append_jsonl(args.output, candidate)
            print(
                f"[{min(start + args.batch_size, total)}/{total}] "
                f"{record['case_id']} accepted "
                f"entities={len(candidate['entities'])} "
                f"relations={len(candidate['relations'])}",
                flush=True,
            )

    existing_count = len(completed_ids)
    report = {
        "input_records_this_run": len(records),
        "previously_completed": existing_count,
        "accepted_this_run": len(accepted),
        "accepted_total": existing_count + len(accepted),
        "rejected": len(failures),
        "acceptance_rate_this_run": (
            len(accepted) / len(records) if records else 0.0
        ),
        "failures": failures,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
