# Table S14 — YOLO11n locked-weight deployment benchmark

This file documents the 30-repeat locked-weight deployment benchmark used for Table S14 and the deployment columns in the main manuscript.

Final source-data file:

```text
supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv
```

Protocol: validation split, batch=2, workers=0, conf=0.001, NMS IoU=0.70, max_det=1000, FP16 GPU inference, five warm-up repeats, and thirty formal repeats.

The 30-repeat recheck replaces the earlier five-repeat latency summary. Median values across the 30 formal repeats are reported. Total latency is the median end-to-end time per image across formal repeats and may not equal the sum of individual stage medians because each stage median is computed independently. Peak allocated refers to the peak CUDA memory actively allocated by tensors; peak reserved refers to the peak CUDA memory reserved by the PyTorch caching allocator.

Raw repeat-level log:

```text
raw_logs/RawLog_TableS14_YOLO11n_deployment_recheck30_raw_repeats.csv
```
