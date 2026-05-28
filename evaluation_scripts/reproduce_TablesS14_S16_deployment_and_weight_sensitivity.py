#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
SUPP_DIR = PROJECT_ROOT / r"06_LOGS\supplementary_source_data"

RECHECK30_SUMMARY = SUPP_DIR / "TableS14_YOLO11n_deployment_benchmark_recheck30_locked_weights.csv"

OUT_S14 = SUPP_DIR / "TableS14_YOLO11n_deployment_benchmark_locked_weights.csv"
OUT_S15 = SUPP_DIR / "TableS15_multi_objective_resolution_selection.csv"
OUT_S16 = SUPP_DIR / "TableS16_multi_objective_weight_sensitivity.csv"

README_S14 = SUPP_DIR / "README_TableS14_YOLO11n_deployment_benchmark_locked_weights.md"
README_S15 = SUPP_DIR / "README_TableS15_multi_objective_resolution_selection.md"
README_S16 = SUPP_DIR / "README_TableS16_multi_objective_weight_sensitivity.md"

MAIN_TEST = {
    800: {"map50": 0.9643, "map50_95": 0.7199, "training_time_h": 4.48},
    960: {"map50": 0.9739, "map50_95": 0.7420, "training_time_h": 7.29},
    1280: {"map50": 0.9741, "map50_95": 0.7543, "training_time_h": 22.54},
}

PARAMS = 2590035
GFLOPS = {
    800: 10.06344,
    960: 14.491354,
    1280: 25.762406,
}


def read_recheck_summary() -> dict[int, dict]:
    if not RECHECK30_SUMMARY.exists():
        raise FileNotFoundError(f"Missing recheck30 summary: {RECHECK30_SUMMARY}")

    rows = {}
    with RECHECK30_SUMMARY.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res = int(row["resolution"])
            rows[res] = row

    expected = {800, 960, 1280}
    if set(rows) != expected:
        raise ValueError(f"Expected resolutions {expected}, got {set(rows)}")
    return rows


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def norm_benefit(values: dict[int, float]) -> dict[int, float]:
    vmin = min(values.values())
    vmax = max(values.values())
    if vmax == vmin:
        return {k: 1.0 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def norm_cost_eff(values: dict[int, float]) -> dict[int, float]:
    vmin = min(values.values())
    vmax = max(values.values())
    if vmax == vmin:
        return {k: 1.0 for k in values}
    return {k: (vmax - v) / (vmax - vmin) for k, v in values.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WRITE] {path}")


def write_table_s14(summary: dict[int, dict]) -> None:
    rows = []
    for res in [800, 960, 1280]:
        r = summary[res]
        rows.append({
            "resolution": res,
            "params": PARAMS,
            "gflops": f"{GFLOPS[res]:.6f}",
            "val_map50_95_median": f"{to_float(r, 'map50_95_median'):.6f}",
            "preprocess_median_ms_img": f"{to_float(r, 'preprocess_median_ms_img'):.6f}",
            "preprocess_iqr_ms_img": f"{to_float(r, 'preprocess_iqr_ms_img'):.6f}",
            "inference_median_ms_img": f"{to_float(r, 'inference_median_ms_img'):.6f}",
            "inference_iqr_ms_img": f"{to_float(r, 'inference_iqr_ms_img'):.6f}",
            "postprocess_median_ms_img": f"{to_float(r, 'postprocess_median_ms_img'):.6f}",
            "postprocess_iqr_ms_img": f"{to_float(r, 'postprocess_iqr_ms_img'):.6f}",
            "total_median_ms_img": f"{to_float(r, 'total_median_ms_img'):.6f}",
            "total_iqr_ms_img": f"{to_float(r, 'total_iqr_ms_img'):.6f}",
            "peak_allocated_median_mib": f"{to_float(r, 'peak_allocated_median_mib'):.6f}",
            "peak_reserved_median_mib": f"{to_float(r, 'peak_reserved_median_mib'):.6f}",
            "formal_repeats": int(r["formal_repeats"]),
            "warmup_repeats": int(r["warmup_repeats"]),
            "split": r["split"],
            "batch": int(r["batch"]),
            "workers": int(r["workers"]),
            "conf": f"{to_float(r, 'conf'):.3f}",
            "iou": f"{to_float(r, 'iou'):.2f}",
            "max_det": int(r["max_det"]),
            "half": r["half"],
        })
    write_csv(OUT_S14, rows)

    README_S14.write_text(
        "# Table S14 — YOLO11n locked-weight deployment benchmark\n\n"
        "This file documents the 30-repeat locked-weight deployment benchmark used for Table S14 and the deployment columns in the main manuscript.\n\n"
        "Final source-data file:\n\n"
        "```text\n"
        "supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv\n"
        "```\n\n"
        "Protocol: validation split, batch=2, workers=0, conf=0.001, NMS IoU=0.70, max_det=1000, FP16 GPU inference, five warm-up repeats, and thirty formal repeats.\n\n"
        "The 30-repeat recheck replaces the earlier five-repeat latency summary. Median values across the 30 formal repeats are reported. Total latency is the median end-to-end time per image across formal repeats and may not equal the sum of individual stage medians because each stage median is computed independently. Peak allocated refers to the peak CUDA memory actively allocated by tensors; peak reserved refers to the peak CUDA memory reserved by the PyTorch caching allocator.\n\n"
        "Raw repeat-level log:\n\n"
        "```text\n"
        "raw_logs/RawLog_TableS14_YOLO11n_deployment_recheck30_raw_repeats.csv\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"[WRITE] {README_S14}")


def compute_table_s15(summary: dict[int, dict]) -> list[dict]:
    accuracy = {res: MAIN_TEST[res]["map50_95"] for res in [800, 960, 1280]}
    training = {res: MAIN_TEST[res]["training_time_h"] for res in [800, 960, 1280]}
    gflops = {res: GFLOPS[res] for res in [800, 960, 1280]}
    latency = {res: to_float(summary[res], "total_median_ms_img") for res in [800, 960, 1280]}
    memory = {res: to_float(summary[res], "peak_reserved_median_mib") for res in [800, 960, 1280]}

    acc_n = norm_benefit(accuracy)
    train_n = norm_cost_eff(training)
    gf_n = norm_cost_eff(gflops)
    lat_n = norm_cost_eff(latency)
    mem_n = norm_cost_eff(memory)

    gain_total = accuracy[1280] - accuracy[800]
    rows = []

    for res in [800, 960, 1280]:
        score_default = (
            0.50 * acc_n[res]
            + 0.20 * train_n[res]
            + 0.10 * gf_n[res]
            + 0.10 * lat_n[res]
            + 0.10 * mem_n[res]
        )
        score_no_gflops = (
            0.50 * acc_n[res]
            + 0.20 * train_n[res]
            + 0.15 * lat_n[res]
            + 0.15 * mem_n[res]
        )

        captured = 0.0 if gain_total == 0 else (accuracy[res] - accuracy[800]) / gain_total

        rows.append({
            "resolution": res,
            "main_test_map50": f"{MAIN_TEST[res]['map50']:.6f}",
            "main_test_map50_95": f"{accuracy[res]:.6f}",
            "deployment_val_map50_95": f"{to_float(summary[res], 'map50_95_median'):.6f}",
            "training_time_h_seed42": f"{training[res]:.6f}",
            "gflops": f"{gflops[res]:.6f}",
            "total_latency_ms_img": f"{latency[res]:.6f}",
            "peak_reserved_mib": f"{memory[res]:.6f}",
            "ap_gain_from_800": f"{accuracy[res] - accuracy[800]:.6f}",
            "ap_gain_captured_fraction_of_800_to_1280": f"{captured:.6f}",
            "training_time_ratio_vs_800": f"{training[res] / training[800]:.6f}",
            "gflops_ratio_vs_800": f"{gflops[res] / gflops[800]:.6f}",
            "latency_ratio_vs_800": f"{latency[res] / latency[800]:.6f}",
            "peak_reserved_ratio_vs_800": f"{memory[res] / memory[800]:.6f}",
            "accuracy_norm": f"{acc_n[res]:.6f}",
            "training_efficiency_norm": f"{train_n[res]:.6f}",
            "gflops_efficiency_norm": f"{gf_n[res]:.6f}",
            "latency_efficiency_norm": f"{lat_n[res]:.6f}",
            "memory_efficiency_norm": f"{mem_n[res]:.6f}",
            "multi_objective_score_default": f"{score_default:.6f}",
            "multi_objective_score_no_gflops": f"{score_no_gflops:.6f}",
            "pareto_status": "non-dominated",
            "interpretation": {
                800: "lowest-cost baseline with the lowest accuracy",
                960: "balanced knee-point candidate under the default weighted objective",
                1280: "accuracy-oriented setting with highest GFLOPs and memory pressure",
            }[res],
        })

    ranked = sorted(rows, key=lambda r: float(r["multi_objective_score_default"]), reverse=True)
    rank = {r["resolution"]: i + 1 for i, r in enumerate(ranked)}
    for r in rows:
        r["score_rank_default"] = rank[r["resolution"]]

    ordered_rows = []
    order = [
        "resolution", "main_test_map50", "main_test_map50_95", "deployment_val_map50_95",
        "training_time_h_seed42", "gflops", "total_latency_ms_img", "peak_reserved_mib",
        "ap_gain_from_800", "ap_gain_captured_fraction_of_800_to_1280",
        "training_time_ratio_vs_800", "gflops_ratio_vs_800", "latency_ratio_vs_800",
        "peak_reserved_ratio_vs_800", "accuracy_norm", "training_efficiency_norm",
        "gflops_efficiency_norm", "latency_efficiency_norm", "memory_efficiency_norm",
        "multi_objective_score_default", "multi_objective_score_no_gflops",
        "score_rank_default", "pareto_status", "interpretation",
    ]
    for row in rows:
        ordered_rows.append({k: row[k] for k in order})
    return ordered_rows


def write_table_s15(summary: dict[int, dict]) -> None:
    rows = compute_table_s15(summary)
    write_csv(OUT_S15, rows)

    README_S15.write_text(
        "# Table S15 — Multi-objective resolution-selection analysis\n\n"
        "This table uses main YOLO11n seed42 standard test mAP50-95, seed42 training time, GFLOPs, and 30-repeat locked-weight deployment latency and peak reserved CUDA memory from Table S14.\n\n"
        "Default score:\n\n"
        "```text\n"
        "Score = 0.50*Accuracy_norm + 0.20*TrainingEfficiency_norm + 0.10*GFLOPsEfficiency_norm + 0.10*LatencyEfficiency_norm + 0.10*MemoryEfficiency_norm\n"
        "```\n\n"
        "Accuracy is normalized as a benefit metric: norm(x) = (x - x_min) / (x_max - x_min). Training time, GFLOPs, latency, and memory are normalized as cost-efficiency metrics: norm_cost_eff(x) = (x_max - x) / (x_max - x_min).\n\n"
        "The score is not intended as a universal utility function. It provides a transparent example of task-objective-dependent resolution selection under one explicit weighting scheme. Different deployment objectives may select 800 or 1280 pixels.\n",
        encoding="utf-8",
    )
    print(f"[WRITE] {README_S15}")


def write_table_s16(summary: dict[int, dict]) -> None:
    rows15 = compute_table_s15(summary)
    acc = {int(r["resolution"]): float(r["accuracy_norm"]) for r in rows15}
    train = {int(r["resolution"]): float(r["training_efficiency_norm"]) for r in rows15}
    gf = {int(r["resolution"]): float(r["gflops_efficiency_norm"]) for r in rows15}
    lat = {int(r["resolution"]): float(r["latency_efficiency_norm"]) for r in rows15}
    mem = {int(r["resolution"]): float(r["memory_efficiency_norm"]) for r in rows15}

    cost = {
        res: 0.4 * train[res] + 0.2 * gf[res] + 0.2 * lat[res] + 0.2 * mem[res]
        for res in [800, 960, 1280]
    }

    def crossing(i: int, j: int) -> float:
        denom = (acc[i] - cost[i]) - (acc[j] - cost[j])
        return (cost[j] - cost[i]) / denom

    alpha_800_960 = crossing(800, 960)
    alpha_960_1280 = crossing(960, 1280)

    rows = [
        {
            "accuracy_weight_alpha_range": f"0.000 <= alpha < {alpha_800_960:.3f}",
            "selected_resolution": 800,
            "interpretation": "cost-prioritized objective; lowest-cost baseline selected",
        },
        {
            "accuracy_weight_alpha_range": f"{alpha_800_960:.3f} <= alpha <= {alpha_960_1280:.3f}",
            "selected_resolution": 960,
            "interpretation": "balanced objective; intermediate resolution selected",
        },
        {
            "accuracy_weight_alpha_range": f"{alpha_960_1280:.3f} < alpha <= 1.000",
            "selected_resolution": 1280,
            "interpretation": "accuracy-prioritized objective; highest-accuracy setting selected",
        },
    ]

    for alpha in [0.00, 0.25, 0.50, 0.75, 1.00]:
        scores = {
            res: alpha * acc[res] + (1 - alpha) * cost[res]
            for res in [800, 960, 1280]
        }
        selected = max(scores, key=scores.get)
        rows.append({
            "accuracy_weight_alpha_range": f"sample alpha = {alpha:.2f}",
            "selected_resolution": selected,
            "interpretation": f"score800={scores[800]:.3f}; score960={scores[960]:.3f}; score1280={scores[1280]:.3f}",
        })

    write_csv(OUT_S16, rows)

    README_S16.write_text(
        "# Table S16 — Multi-objective weight-sensitivity analysis\n\n"
        "This table evaluates how the selected resolution changes as the accuracy weight alpha varies from 0 to 1.\n\n"
        "Definition:\n\n"
        "```text\n"
        "Score_alpha = alpha * Accuracy_norm + (1 - alpha) * CostEfficiency_mix\n"
        "```\n\n"
        "The cost-efficiency mixture keeps the internal cost ratio fixed as Training:GFLOPs:Latency:Memory = 0.4:0.2:0.2:0.2.\n\n"
        f"Using the 30-repeat latency recheck and the same normalized values as Table S15, 800 pixels is selected when alpha < {alpha_800_960:.3f}, 960 pixels is selected when {alpha_800_960:.3f} <= alpha <= {alpha_960_1280:.3f}, and 1280 pixels is selected when alpha > {alpha_960_1280:.3f}.\n\n"
        "This sensitivity analysis supports the interpretation that 960 pixels is not universally optimal; it is selected only under a moderate accuracy-prioritized objective range. Cost-prioritized settings select 800 pixels, whereas strongly accuracy-prioritized settings select 1280 pixels.\n",
        encoding="utf-8",
    )
    print(f"[WRITE] {README_S16}")


def main() -> None:
    print("=" * 100)
    print("Update Tables S14, S15, and S16 after 30-repeat deployment recheck")
    print("=" * 100)

    summary = read_recheck_summary()
    write_table_s14(summary)
    write_table_s15(summary)
    write_table_s16(summary)

    print("=" * 100)
    print("[DONE] Updated source-data files:")
    print(f"1. {OUT_S14}")
    print(f"2. {OUT_S15}")
    print(f"3. {OUT_S16}")
    print("=" * 100)


if __name__ == "__main__":
    main()
