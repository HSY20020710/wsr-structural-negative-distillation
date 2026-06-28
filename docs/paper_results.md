# Paper Result Summary

This file records aggregate results reported in the paper. It is not a substitute for the private prediction files, training logs, or checkpoints.

## Main Reported Results

| Setting | Text Triple-F1 | CVR-All |
|---|---:|---:|
| Standard KD, Student-only | 47.49% | 31.21% |
| Ours, Student-only | 61.50% | 30.98% |
| Ours, Student+Gate | 71.63% | 0.00% |

## Efficiency Summary

| Model | Role | GPU memory | Latency/report |
|---|---|---:|---:|
| Qwen teacher | Offline teacher | 76.69 GiB | 63.893 s |
| Qwen3-1.7B student | Online student | 3.47 GiB | 9.952 s |

These aggregate results were generated from the private dataset and frozen experiment outputs using the scripts listed in `docs/reproduction.md`.
