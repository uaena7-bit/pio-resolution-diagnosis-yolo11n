# Table S14 — YOLO11n Locked-weight Deployment Benchmark

This file documents the source data for Table S14.

Final source-data file:

`supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv`

Raw repeat-level log:

`raw_logs/RawLog_TableS14_YOLO11n_deployment_benchmark_raw_repeats.csv`

## Protocol

The benchmark used validation split evaluation, batch size 2, workers=0, confidence threshold=0.001, NMS IoU=0.70, max_det=1000, GPU FP16 inference, one warm-up repeat, and five formal repeats.

The weights were explicitly locked to the seed-42 YOLO11n checkpoints for 800, 960, and 1280 pixels. The 960-pixel checkpoint used the non-b2-2 directory.

## Important notes

Total latency is the median end-to-end time per image across five formal repeats. It may not equal the sum of individual stage medians because each stage median is computed independently over the repeat distribution.

The 1280-pixel median total latency was slightly lower than the 960-pixel value in this batch-2 locked-weight benchmark. This reflects measurement variability across the five formal repeats and potential batch-processing efficiency effects rather than a systematic speed advantage of 1280 over 960.

Peak memory refers to validation/inference CUDA memory, not training peak VRAM.
