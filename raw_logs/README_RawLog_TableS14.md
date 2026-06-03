# Raw Log Description for Table S14

This file documents the raw repeat log used for the 30-repeat locked-weight YOLO11n deployment benchmark reported as Table S14.

Relevant source file:

- `RawLog_TableS14_YOLO11n_deployment_recheck30_raw_repeats.csv`

## Benchmark purpose

The raw log records repeated validation/inference timing measurements used to summarize deployment-oriented latency and CUDA memory pressure across input resolutions.

The tested resolutions are:

- 800 px
- 960 px
- 1280 px

## Benchmark protocol

The Table S14 benchmark used the following protocol:

- validation split
- batch size = 2
- workers = 0
- confidence threshold = 0.001
- NMS IoU = 0.70
- max_det = 1000
- GPU FP16 inference
- five warm-up repeats discarded
- thirty formal repeats retained for summary statistics

## Column interpretation

Column names may differ slightly depending on the exported benchmark script, but the expected fields are:

| Column type | Meaning |
|---|---|
| resolution, imgsz, or input_size | Input resolution used for inference |
| repeat, run, or trial | Formal repeat index after warm-up repeats were discarded |
| preprocess_ms | Preprocessing time per image in milliseconds |
| inference_ms | Model forward inference time per image in milliseconds |
| postprocess_ms | Postprocessing and NMS time per image in milliseconds |
| total_ms | End-to-end latency per image in milliseconds |
| peak_allocated_mib | Peak CUDA memory actively allocated by tensors |
| peak_reserved_mib | Peak CUDA memory reserved by the PyTorch caching allocator |
| params | Number of model parameters |
| gflops | Estimated computational cost in GFLOPs |
| val_map50_95 | Validation mAP50-95 for the locked model used in the benchmark |

## Summary rule

Table S14 reports medians and IQRs computed from the thirty formal repeats after warm-up repeats are discarded.

The total latency value is the median end-to-end time per image across formal repeats. It may not exactly equal the sum of the individual median preprocessing, inference, and postprocessing times because each component median is computed independently.

## Memory note

Peak allocated memory and peak reserved memory refer to validation/inference CUDA memory during the deployment benchmark. They should not be interpreted as training peak VRAM.

## Reproducibility note

Absolute latency values are hardware- and software-dependent. The raw log is provided to make the local benchmark transparent and to support the relative comparison among 800, 960, and 1280 px under the same local environment.
