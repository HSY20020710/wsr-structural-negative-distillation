from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ontology.gate import apply_wsr_gate  # noqa: E402
from src.student.data import (  # noqa: E402
    SUPPORTED_METHODS,
    SUPPORTED_STAGES,
    SUPPORTED_TARGET_SCHEMAS,
    build_entity_system_prompt,
    build_system_prompt,
    build_text_schema_system_prompt,
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
        description="Run one trained student model on the fixed Test set."
    )
    prediction_methods = set(SUPPORTED_METHODS) | {"prompt_only"}
    parser.add_argument("--method", choices=sorted(prediction_methods), required=True)
    parser.add_argument("--stage", choices=sorted(SUPPORTED_STAGES), default="joint")
    parser.add_argument(
        "--target_schema",
        choices=sorted(SUPPORTED_TARGET_SCHEMAS),
        default="full",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ontology", type=Path, default=ROOT / "configs/wsr_ontology.yaml"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--benchmark_output",
        type=Path,
        help="Write parameter, peak-memory, and end-to-end latency statistics.",
    )
    parser.add_argument(
        "--benchmark_warmup_batches",
        type=int,
        default=0,
        help="Exclude the first N processed batches from latency statistics.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output} already exists; use --resume or remove it"
        )
    if not args.model.is_dir():
        raise FileNotFoundError(
            f"Student model directory does not exist: {args.model.resolve()}"
        )
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(
            f"Incomplete full-model directory: {args.model.resolve()}"
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
    test_records = read_jsonl(args.test)
    completed = (
        {str(item["case_id"]) for item in read_jsonl(args.output)}
        if args.resume and args.output.exists()
        else set()
    )
    pending = [
        record
        for record in test_records
        if str(record["case_id"]) not in completed
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
    apply_gate = args.method in {"kd_wsr_post", "ours"}
    total = len(pending)
    report_latencies = []

    for start in range(0, total, args.batch_size):
        batch_index = start // args.batch_size
        batch = pending[start : start + args.batch_size]
        prompts = []
        for record in batch:
            if args.target_schema == "text":
                system_prompt = build_text_schema_system_prompt(ontology)
                if args.stage == "entity":
                    system_prompt += "\n本阶段只抽取分类和实体，relations必须为空数组。"
            elif args.stage == "entity":
                system_prompt = build_entity_system_prompt(ontology)
            else:
                system_prompt = build_system_prompt(ontology)
            messages = [
                {"role": "system", "content": system_prompt},
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
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_started = time.perf_counter()
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
            result = {
                "case_id": record["case_id"],
                "text": record["text"],
                "method": args.method,
                "model_path": str(args.model),
                "raw_response": response,
                "parse_success": parsed["parse_success"],
                "defect_label": "",
                "event_label": "",
                "entities": [],
                "relations": [],
                "triples": [],
                "errors": list(parsed["errors"]),
            }
            if parsed["parse_success"]:
                normalized = normalize_teacher_output(
                    parsed["data"], record, ontology=ontology
                )
                result.update(normalized)
                result["parse_success"] = True
                if args.stage == "entity":
                    result["relations"] = []
                    result["triples"] = []
                if apply_gate:
                    result = apply_wsr_gate(result, ontology)
            append_jsonl(args.output, result)
            print(
                f"[{len(completed) + min(start + len(batch), total)}/"
                f"{len(test_records)}] {record['case_id']} "
                f"parse={result['parse_success']} "
                f"entities={len(result['entities'])} "
                f"relations={len(result['relations'])}",
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - batch_started
        if batch_index >= args.benchmark_warmup_batches:
            report_latencies.extend([batch_elapsed / len(batch)] * len(batch))

    if args.benchmark_output:
        if not report_latencies:
            raise ValueError(
                "No timed batches remain after --benchmark_warmup_batches"
            )
        sorted_latencies = sorted(report_latencies)
        p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
        benchmark = {
            "method": args.method,
            "model_path": str(args.model.resolve()),
            "test_path": str(args.test.resolve()),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "warmup_batches": args.benchmark_warmup_batches,
            "timed_reports": len(report_latencies),
            "parameters": parameter_count,
            "parameters_billion": parameter_count / 1e9,
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else None
            ),
            "latency_mean_seconds": statistics.mean(report_latencies),
            "latency_std_seconds": (
                statistics.stdev(report_latencies)
                if len(report_latencies) > 1
                else 0.0
            ),
            "latency_median_seconds": statistics.median(report_latencies),
            "latency_p95_seconds": sorted_latencies[p95_index],
            "includes_generation_parsing_gate_and_serialization": True,
        }
        args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_output.write_text(
            json.dumps(benchmark, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(benchmark, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

