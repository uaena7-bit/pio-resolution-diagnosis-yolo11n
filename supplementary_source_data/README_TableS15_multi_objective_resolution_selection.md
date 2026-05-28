# Table S15 — Multi-objective resolution-selection analysis

This table uses main YOLO11n seed42 standard test mAP50-95, seed42 training time, GFLOPs, and 30-repeat locked-weight deployment latency and peak reserved CUDA memory from Table S14.

Default score:

```text
Score = 0.50*Accuracy_norm + 0.20*TrainingEfficiency_norm + 0.10*GFLOPsEfficiency_norm + 0.10*LatencyEfficiency_norm + 0.10*MemoryEfficiency_norm
```

Accuracy is normalized as a benefit metric: norm(x) = (x - x_min) / (x_max - x_min). Training time, GFLOPs, latency, and memory are normalized as cost-efficiency metrics: norm_cost_eff(x) = (x_max - x) / (x_max - x_min).

The score is not intended as a universal utility function. It provides a transparent example of task-objective-dependent resolution selection under one explicit weighting scheme. Different deployment objectives may select 800 or 1280 pixels.
