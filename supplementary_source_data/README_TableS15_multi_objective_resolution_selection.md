# Table S15 — Multi-objective Resolution-selection Analysis

This file documents Table S15 source data.

The analysis combines main YOLO11n seed42 test accuracy with deployment and cost indicators:

- main test mAP50-95
- seed42 training time
- GFLOPs
- locked-weight total latency
- locked-weight peak reserved memory

Deployment indicators are read from:

supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv

Default weights:

- accuracy: 0.50
- training time efficiency: 0.20
- GFLOPs efficiency: 0.10
- latency efficiency: 0.10
- memory efficiency: 0.10

Accuracy is treated as a benefit metric. Training time, GFLOPs, latency, and memory are treated as cost metrics.

Interpretation:

Under this explicit task-objective weighting, 960 pixels is identified as the balanced knee-point candidate.
This does not mean that 960 is universally optimal.
The 1280-pixel setting remains the accuracy-oriented setting, whereas 800 pixels remains the lowest-cost setting.

Pareto analysis alone does not select a single resolution because the settings represent different task-dependent trade-offs.
Therefore, the weighted score is reported as an explicit resolution-selection rule rather than as an absolute ranking of detector quality.
