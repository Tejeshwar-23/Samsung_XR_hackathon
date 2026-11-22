"""
Stage 5: Full End-to-End Evaluation Pipeline for Samsung XR Hackathon 2026 Track 3.
Evaluates:
- 5 EuRoC Machine Hall degraded sequences (MH_01 through MH_05)
- 2 TUM-VI Room 1 variants (Canonical and Corrected)
Collects:
- Frames In / Frames Out
- Sanitized / Invalid Row Counts
- Bias Status & Bias Values
- ATE RMSE (m) via SE(3) umeyama alignment
- 1s RPE RMSE (m) (delta=200 frames at 200 Hz)
- Estimated & GT Path Lengths (m)
- Python Heap Peak (MB) (traced via tracemalloc)
- Compute Latency (ms/s) (both traced and untraced)
- TUM Trajectory Validation status
"""

import sys
import os
import time
import copy
import json
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd

from evo.core import trajectory, sync, metrics
from evo.tools import file_interface

from src.main import run_pipeline

SEQUENCES = [
    {
        "name": "MH_01_easy",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_01_easy_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/final_validation/MH_01_final.txt",
        "meta_path": "outputs/final_validation/MH_01_meta.json",
        "bias_ref": [-0.003206, 0.020793, 0.079723],
    },
    {
        "name": "MH_02_easy",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_02_easy_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/final_validation/MH_02_final.txt",
        "meta_path": "outputs/final_validation/MH_02_meta.json",
        "bias_ref": None,
    },
    {
        "name": "MH_03_medium",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_03_medium_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/final_validation/MH_03_final.txt",
        "meta_path": "outputs/final_validation/MH_03_meta.json",
        "bias_ref": None,
    },
    {
        "name": "MH_04_difficult",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_04_difficult_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/final_validation/MH_04_final.txt",
        "meta_path": "outputs/final_validation/MH_04_meta.json",
        "bias_ref": None,
    },
    {
        "name": "MH_05_difficult",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_05_difficult_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/final_validation/MH_05_final.txt",
        "meta_path": "outputs/final_validation/MH_05_meta.json",
        "bias_ref": None,
    },
    {
        "name": "TUM-VI Room 1",
        "variant": "Canonical",
        "input_csv": "outputs/TUM_VI_room1_degraded_canonical.csv",
        "gt_type": "tum",
        "gt_path": "outputs/tum_vi_room1_gt.tum",
        "out_tum": "outputs/final_validation/TUM_VI_room1_canonical_final.txt",
        "meta_path": "outputs/final_validation/TUM_VI_room1_canonical_meta.json",
        "bias_ref": None,
    },
    {
        "name": "TUM-VI Room 1",
        "variant": "Corrected",
        "input_csv": "outputs/TUM_VI_room1_degraded_corrected.csv",
        "gt_type": "tum",
        "gt_path": "outputs/tum_vi_room1_gt.tum",
        "out_tum": "outputs/final_validation/TUM_VI_room1_corrected_final.txt",
        "meta_path": "outputs/final_validation/TUM_VI_room1_corrected_meta.json",
        "bias_ref": None,
    },
]

def load_ground_truth(cfg):
    if cfg["gt_type"] == "euroc":
        df_gt = pd.read_csv(cfg["gt_path"])
        ts_gt = df_gt.iloc[:, 0].to_numpy() * 1e-9
        pos_gt = df_gt.iloc[:, 1:4].to_numpy()
        qw = df_gt.iloc[:, 4].to_numpy()
        qx = df_gt.iloc[:, 5].to_numpy()
        qy = df_gt.iloc[:, 6].to_numpy()
        qz = df_gt.iloc[:, 7].to_numpy()
        return trajectory.PoseTrajectory3D(pos_gt, np.column_stack([qx, qy, qz, qw]), ts_gt)
    else:
        return file_interface.read_tum_trajectory_file(cfg["gt_path"])

def measure_untraced_runtime(in_csv, out_temp):
    # Execute run_pipeline with tracemalloc temporarily disabled around the core or measure standard pipeline
    # Since run_pipeline internally starts tracemalloc, let's measure with tracemalloc stopped if possible,
    # or measure using a lean sub-call. In Python, tracemalloc.stop() stops tracing.
    # To measure truly untraced: we monkeypatch tracemalloc.start() to no-op for this test call!
    orig_start = tracemalloc.start
    orig_is_tracing = tracemalloc.is_tracing
    orig_get_traced = tracemalloc.get_traced_memory
    orig_stop = tracemalloc.stop

    try:
        tracemalloc.stop()  # ensure stopped
        tracemalloc.start = lambda *a, **kw: None
        tracemalloc.is_tracing = lambda: False
        tracemalloc.get_traced_memory = lambda: (0, 0)
        tracemalloc.stop = lambda: None

        t0 = time.perf_counter()
        telemetry = run_pipeline(
            input_path=in_csv,
            output_path=out_temp,
            log_level="ERROR",
        )
        t_total = time.perf_counter() - t0
        dur = float(telemetry["telemetry_duration_s"])
        ms_per_s = (t_total * 1000.0) / max(dur, 1e-6)
        return t_total, ms_per_s
    finally:
        tracemalloc.start = orig_start
        tracemalloc.is_tracing = orig_is_tracing
        tracemalloc.get_traced_memory = orig_get_traced
        tracemalloc.stop = orig_stop

def main():
    Path("outputs/final_validation").mkdir(parents=True, exist_ok=True)
    all_results = []

    print("=" * 80)
    print("STARTING STAGE 5 FULL EVALUATION (7 SEQUENCES)")
    print("=" * 80)

    for cfg in SEQUENCES:
        name = cfg["name"]
        variant = cfg["variant"]
        in_csv = cfg["input_csv"]
        out_tum = cfg["out_tum"]
        meta_json = cfg["meta_path"]

        print(f"\n--- Running: {name} ({variant}) ---")
        print(f"Input: {in_csv}")
        print(f"Output: {out_tum}")

        # 1. Run pipeline (generates TUM output + collects traced telemetry & validation)
        telemetry = run_pipeline(
            input_path=in_csv,
            output_path=out_tum,
            metadata_output=meta_json,
            log_level="WARNING",
        )

        # 2. Measure untraced latency
        untraced_time, untraced_ms_per_s = measure_untraced_runtime(in_csv, "scratch/temp_untraced_eval.txt")

        # 3. Offline ground truth evaluation
        gt_traj = load_ground_truth(cfg)
        est_traj = file_interface.read_tum_trajectory_file(out_tum)

        gt_sync, est_sync = sync.associate_trajectories(gt_traj, est_traj, max_diff=0.02)
        est_aligned = copy.deepcopy(est_sync)
        est_aligned.align(gt_sync, correct_scale=False)

        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((gt_sync, est_aligned))
        ate_rmse = float(ape.get_all_statistics()["rmse"])

        # RPE at 1 second: 200 Hz -> delta = 200 frames
        rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=200, delta_unit=metrics.Unit.frames, all_pairs=False)
        rpe.process_data((gt_sync, est_aligned))
        rpe_rmse = float(rpe.get_all_statistics()["rmse"])

        est_path_len = float(est_aligned.path_length)
        gt_path_len = float(gt_sync.path_length)

        # Bias error calculation
        bias_info = telemetry["bias_estimation"]
        bias_status = bias_info["status"]
        est_bias = np.array(bias_info["gyro_bias_rad_s"])
        if cfg["bias_ref"] is not None:
            ref_b = np.array(cfg["bias_ref"])
            bias_err_degs = float(np.linalg.norm(est_bias - ref_b) * 180.0 / np.pi)
            bias_err_str = f"{bias_err_degs:.3f}"
        else:
            bias_err_str = "N/A"

        val = telemetry["validation"]
        tum_status = "PASS" if val["valid"] else "FAIL"

        row_res = {
            "dataset": name,
            "variant": variant,
            "frames_in": telemetry["input_frames"],
            "frames_out": telemetry["output_tum_poses"],
            "bias_status": bias_status,
            "bias_err_deg_s": bias_err_str,
            "ate_rmse_m": ate_rmse,
            "rpe_rmse_m": rpe_rmse,
            "est_path_m": est_path_len,
            "gt_path_m": gt_path_len,
            "heap_mb": telemetry["python_heap_peak_mb"],
            "latency_untraced_ms_s": untraced_ms_per_s,
            "latency_traced_ms_s": telemetry["compute_latency_ms_per_s"],
            "tum_status": tum_status,
        }
        all_results.append(row_res)

        print(f"  Frames: In={row_res['frames_in']}, Out={row_res['frames_out']}")
        print(f"  Bias Status: {bias_status} | Error: {bias_err_str}")
        print(f"  ATE RMSE: {ate_rmse:.2f} m | 1s RPE RMSE: {rpe_rmse:.2f} m")
        print(f"  Path: Est={est_path_len:.1f} m, GT={gt_path_len:.1f} m")
        print(f"  Heap Peak: {row_res['heap_mb']:.2f} MB (Budget <= 8.0 MB)")
        print(f"  Untraced Latency: {untraced_ms_per_s:.2f} ms/s (Budget <= 20 ms/s)")
        print(f"  Traced Latency: {row_res['latency_traced_ms_s']:.2f} ms/s")
        print(f"  TUM Validation: {tum_status}")

    # Save summary JSON
    with open("outputs/final_validation/stage5_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 5 EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
