from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = [
    ("qwen3_supervised", "Qwen3 supervised", "Online student"),
    ("standard_kd", "Qwen3 Standard KD", "Online student"),
    ("ours", "Qwen3 Ours", "Online student"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the student efficiency table.")
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--reports_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for method, model_name, role in METHODS:
        benchmark_path = args.benchmark_dir / f"{method}.json"
        metric_path = args.reports_dir / method / f"{method}.json"
        if not benchmark_path.is_file():
            raise FileNotFoundError(benchmark_path)
        if not metric_path.is_file():
            raise FileNotFoundError(metric_path)
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model_name,
                "role": role,
                "parameters_b": benchmark["parameters_billion"],
                "gpu_memory_gib": benchmark["peak_gpu_memory_gib"],
                "latency_mean": benchmark["latency_mean_seconds"],
                "latency_std": benchmark["latency_std_seconds"],
                "latency_p95": benchmark["latency_p95_seconds"],
                "triple_f1": metrics["triple_f1"],
                "cvr_all": metrics["cvr_all"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "efficiency_results.csv"
    headers = [
        "Model", "Online/offline role", "Parameters (B)", "GPU memory (GiB)",
        "Latency/report (s)", "P95 latency (s)", "Triple-F1", "CVR-All",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["model"], row["role"], row["parameters_b"],
                row["gpu_memory_gib"],
                f'{row["latency_mean"]:.4f} +/- {row["latency_std"]:.4f}',
                row["latency_p95"], row["triple_f1"], row["cvr_all"],
            ])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * 6) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            row["model"], row["role"], f'{row["parameters_b"]:.3f}',
            f'{row["gpu_memory_gib"]:.2f}',
            f'{row["latency_mean"]:.3f} +/- {row["latency_std"]:.3f}',
            f'{row["latency_p95"]:.3f}', f'{100 * row["triple_f1"]:.2f}%',
            f'{100 * row["cvr_all"]:.2f}%',
        ]) + " |")
    md_path = args.output_dir / "efficiency_results.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
