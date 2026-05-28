#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Table S15: multi-objective resolution-selection analysis.

Inputs:
- TableS14_YOLO11n_deployment_benchmark_locked_weights.csv
- Main YOLO11n seed42 test accuracy and training time

Outputs:
- TableS15_multi_objective_resolution_selection.csv
- README_TableS15_multi_objective_resolution_selection.md
"""

from pathlib import Path
import csv


PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
SUPP_DIR = PROJECT_ROOT / "06_LOGS" / "supplementary_source_data"

TABLE_S14 = SUPP_DIR / "TableS14_YOLO11n_deployment_benchmark_locked_weights.csv"
OUT_CSV = SUPP_DIR / "TableS15_multi_objective_resolution_selection.csv"
OUT_README = SUPP_DIR / "README_TableS15_multi_objective_resolution_selection.md"

# Main YOLO11n seed42 test results.
MAIN_TEST_MAP50 = {
    800: 0.9643,
    960: 0.9739,
    1280: 0.9741,
}

MAIN_TEST_MAP50_95 = {
    800: 0.7199,
    960: 0.7420,
    1280: 0.7543,
}

# Main YOLO11n seed42 training time.
TRAINING_TIME_H = {
    800: 4.480,
    960: 7.290,
    1280: 22.540,
}


def read_table_s14(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing Table S14 file: {path}")

    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row["resolution"])] = row

    expected = {800, 960, 1280}
    if set(rows.keys()) != expected:
        raise ValueError(f"Expected resolutions {expected}, got {set(rows.keys())}")

    return rows


def benefit_norm(values):
    mn = min(values.values())
    mx = max(values.values())
    if mx == mn:
        return {k: 1.0 for k in values}
    return {k: (v - mn) / (mx - mn) for k, v in values.items()}


def cost_norm(values):
    mn = min(values.values())
    mx = max(values.values())
    if mx == mn:
        return {k: 1.0 for k in values}
    return {k: (mx - v) / (mx - mn) for k, v in values.items()}


def fmt(x):
    if isinstance(x, float):
        return f"{x:.6f}"
    return x


def main():
    deployment = read_table_s14(TABLE_S14)

    gflops = {r: float(deployment[r]["gflops"]) for r in deployment}
    total_latency = {r: float(deployment[r]["total_median_ms_img"]) for r in deployment}
    peak_reserved = {r: float(deployment[r]["peak_reserved_median_mib"]) for r in deployment}
    deployment_val_map = {r: float(deployment[r]["map50_95_median"]) for r in deployment}

    accuracy_n = benefit_norm(MAIN_TEST_MAP50_95)
    training_n = cost_norm(TRAINING_TIME_H)
    gflops_n = cost_norm(gflops)
    latency_n = cost_norm(total_latency)
    memory_n = cost_norm(peak_reserved)

    weights = {
        "accuracy": 0.50,
        "training": 0.20,
        "gflops": 0.10,
        "latency": 0.10,
        "memory": 0.10,
    }

    weights_no_gflops = {
        "accuracy": 0.50,
        "training": 0.20,
        "latency": 0.15,
        "memory": 0.15,
    }

    ap_range = MAIN_TEST_MAP50_95[1280] - MAIN_TEST_MAP50_95[800]

    rows = []
    for r in [800, 960, 1280]:
        score_default = (
            weights["accuracy"] * accuracy_n[r]
            + weights["training"] * training_n[r]
            + weights["gflops"] * gflops_n[r]
            + weights["latency"] * latency_n[r]
            + weights["memory"] * memory_n[r]
        )

        score_no_gflops = (
            weights_no_gflops["accuracy"] * accuracy_n[r]
            + weights_no_gflops["training"] * training_n[r]
            + weights_no_gflops["latency"] * latency_n[r]
            + weights_no_gflops["memory"] * memory_n[r]
        )

        ap_gain = MAIN_TEST_MAP50_95[r] - MAIN_TEST_MAP50_95[800]
        ap_gain_fraction = ap_gain / ap_range if ap_range > 0 else 0.0

        if r == 800:
            interpretation = "lowest-cost baseline with the lowest accuracy"
        elif r == 960:
            interpretation = "balanced knee-point candidate under the default weighted objective"
        else:
            interpretation = "accuracy-oriented setting with highest GFLOPs and memory pressure"

        rows.append({
            "resolution": r,
            "main_test_map50": MAIN_TEST_MAP50[r],
            "main_test_map50_95": MAIN_TEST_MAP50_95[r],
            "deployment_val_map50_95": deployment_val_map[r],
            "training_time_h_seed42": TRAINING_TIME_H[r],
            "gflops": gflops[r],
            "total_latency_ms_img": total_latency[r],
            "peak_reserved_mib": peak_reserved[r],
            "ap_gain_from_800": ap_gain,
            "ap_gain_captured_fraction_of_800_to_1280": ap_gain_fraction,
            "training_time_ratio_vs_800": TRAINING_TIME_H[r] / TRAINING_TIME_H[800],
            "gflops_ratio_vs_800": gflops[r] / gflops[800],
            "latency_ratio_vs_800": total_latency[r] / total_latency[800],
            "peak_reserved_ratio_vs_800": peak_reserved[r] / peak_reserved[800],
            "accuracy_norm": accuracy_n[r],
            "training_efficiency_norm": training_n[r],
            "gflops_efficiency_norm": gflops_n[r],
            "latency_efficiency_norm": latency_n[r],
            "memory_efficiency_norm": memory_n[r],
            "multi_objective_score_default": score_default,
            "multi_objective_score_no_gflops": score_no_gflops,
            "pareto_status": "non-dominated",
            "interpretation": interpretation,
        })

    ranked = sorted(rows, key=lambda x: x["multi_objective_score_default"], reverse=True)
    rank_map = {row["resolution"]: rank for rank, row in enumerate(ranked, start=1)}
    for row in rows:
        row["score_rank_default"] = rank_map[row["resolution"]]

    fieldnames = [
        "resolution",
        "main_test_map50",
        "main_test_map50_95",
        "deployment_val_map50_95",
        "training_time_h_seed42",
        "gflops",
        "total_latency_ms_img",
        "peak_reserved_mib",
        "ap_gain_from_800",
        "ap_gain_captured_fraction_of_800_to_1280",
        "training_time_ratio_vs_800",
        "gflops_ratio_vs_800",
        "latency_ratio_vs_800",
        "peak_reserved_ratio_vs_800",
        "accuracy_norm",
        "training_efficiency_norm",
        "gflops_efficiency_norm",
        "latency_efficiency_norm",
        "memory_efficiency_norm",
        "multi_objective_score_default",
        "multi_objective_score_no_gflops",
        "score_rank_default",
        "pareto_status",
        "interpretation",
    ]

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row[k]) for k in fieldnames})

    readme_lines = [
        "# Table S15 — Multi-objective Resolution-selection Analysis",
        "",
        "This file documents Table S15 source data.",
        "",
        "The analysis combines main YOLO11n seed42 test accuracy with deployment and cost indicators:",
        "",
        "- main test mAP50-95",
        "- seed42 training time",
        "- GFLOPs",
        "- locked-weight total latency",
        "- locked-weight peak reserved memory",
        "",
        "Deployment indicators are read from:",
        "",
        "supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv",
        "",
        "Default weights:",
        "",
        "- accuracy: 0.50",
        "- training time efficiency: 0.20",
        "- GFLOPs efficiency: 0.10",
        "- latency efficiency: 0.10",
        "- memory efficiency: 0.10",
        "",
        "Accuracy is treated as a benefit metric. Training time, GFLOPs, latency, and memory are treated as cost metrics.",
        "",
        "Interpretation:",
        "",
        "Under this explicit task-objective weighting, 960 pixels is identified as the balanced knee-point candidate.",
        "This does not mean that 960 is universally optimal.",
        "The 1280-pixel setting remains the accuracy-oriented setting, whereas 800 pixels remains the lowest-cost setting.",
        "",
        "Pareto analysis alone does not select a single resolution because the settings represent different task-dependent trade-offs.",
        "Therefore, the weighted score is reported as an explicit resolution-selection rule rather than as an absolute ranking of detector quality.",
        "",
    ]

    OUT_README.write_text("\n".join(readme_lines), encoding="utf-8")

    print("=" * 100)
    print("[DONE] Table S15 generated.")
    print(f"CSV   : {OUT_CSV}")
    print(f"README: {OUT_README}")
    print("=" * 100)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
