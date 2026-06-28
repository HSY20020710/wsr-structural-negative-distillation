from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path


def version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the student-training server.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gold_train", type=Path, required=True)
    parser.add_argument("--teacher_train", type=Path, required=True)
    parser.add_argument("--pseudo_train", type=Path)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        args.config,
        args.gold_train,
        args.teacher_train,
        args.dev,
        args.test,
    ]
    if args.pseudo_train:
        paths.append(args.pseudo_train)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    config = json.loads(args.config.read_text(encoding="utf-8"))

    import torch

    report = {
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "accelerate": version("accelerate"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "memory_gb": round(
                    torch.cuda.get_device_properties(index).total_memory
                    / 1024**3,
                    2,
                ),
            }
            for index in range(torch.cuda.device_count())
        ],
        "student_model": config["student_model"],
        "torch_dtype": config["torch_dtype"],
        "training_strategy": config.get("training_strategy"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["cuda_available"]:
        raise RuntimeError("CUDA is not available")
    if config["torch_dtype"] == "bfloat16" and not all(
        gpu["bf16_supported"] for gpu in report["gpus"]
    ):
        raise RuntimeError(
            "The config requests bfloat16, but at least one GPU lacks bf16 support"
        )
    if config.get("training_strategy") != "full_parameter":
        raise RuntimeError(
            "student_training.json must set training_strategy=full_parameter"
        )


if __name__ == "__main__":
    main()
