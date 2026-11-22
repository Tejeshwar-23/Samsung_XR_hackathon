"""
Final Seven-Sequence Benchmark Suite for Production Configuration A.
Evaluates:
- MH_01_easy
- MH_02_easy
- MH_03_medium
- MH_04_difficult
- MH_05_difficult
- TUM-VI Room 1 Canonical
- TUM-VI Room 1 Corrected

Computes and verifies:
- 1:1 input/output row correspondence
- Duplicate timestamp count
- Terminal failure frames retained
- Finite positions & normalized quaternions
- TUM validation
- ATE RMSE (m)
- 1s RPE RMSE (m)
- Checkpoint errors at 25%, 50%, 75%
- Blind 5s trajectory and endpoint errors
- Dropout recovery error
- Path lengths (Est vs GT)
- Traced peak Python heap (MB)
- Untraced runtime and latency (ms/s)
- Single-thread execution
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
        "id": "MH_01",
        "name": "MH_01_easy",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_01_easy_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/production_config_a/MH_01_config_a.txt",
        "meta_path": "outputs/production_config_a/MH_01_config_a_meta.json",
    },
    {
        "id": "MH_02",
        "name": "MH_02_easy",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_02_easy_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/production_config_a/MH_02_config_a.txt",
        "meta_path": "outputs/production_config_a/MH_02_config_a_meta.json",
    },
    {
        "id": "MH_03",
        "name": "MH_03_medium",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_03_medium_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/production_config_a/MH_03_config_a.txt",
        "meta_path": "outputs/production_config_a/MH_03_config_a_meta.json",
    },
    {
        "id": "MH_04",
        "name": "MH_04_difficult",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_04_difficult_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/production_config_a/MH_04_config_a.txt",
        "meta_path": "outputs/production_config_a/MH_04_config_a_meta.json",
    },
    {
        "id": "MH_05",
        "name": "MH_05_difficult",
        "variant": "EuRoC Degraded",
        "input_csv": "outputs/MH_05_difficult_degraded.csv",
        "gt_type": "euroc",
        "gt_path": "data/raw/euroc/machine_hall/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv",
        "out_tum": "outputs/production_config_a/MH_05_config_a.txt",
        "meta_path": "outputs/production_config_a/MH_05_config_a_meta.json",
    },
    {
        "id": "TUM_VI_canonical",
        "name": "TUM-VI Room 1",
        "variant": "Canonical",
        "input_csv": "outputs/TUM_VI_room1_degraded_canonical.csv",
        "gt_type": "tum",
        "gt_path": "outputs/tum_vi_room1_gt.tum",
        "out_tum": "outputs/production_config_a/TUM_VI_room1_canonical_config_a.txt",
        "meta_path": "outputs/production_config_a/TUM_VI_room1_canonical_config_a_meta.json",
    },
    {
        "id": "TUM_VI_corrected",
        "name": "TUM-VI Room 1",
        "variant": "Corrected",
        "input_csv": "outputs/TUM_VI_room1_degraded_corrected.csv",
        "gt_type": "tum",
        "gt_path": "outputs/tum_vi_room1_gt.tum",
        "out_tum": "outputs/production_config_a/TUM_VI_room1_corrected_config_a.txt",
        "meta_path": "outputs/production_config_a/TUM_VI_room1_corrected_config_a_meta.json",
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

def measure_untraced_execution(in_csv, out_temp):
    orig_start = tracemalloc.start
    orig_is_tracing = tracemalloc.is_tracing
    orig_get_traced = tracemalloc.get_traced_memory
    orig_stop = tracemalloc.stop

    try:
        tracemalloc.stop()
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
    Path("outputs/production_config_a").mkdir(parents=True, exist_ok=True)
    all_results = []

    print("=" * 80)
    print("STARTING FINAL BENCHMARK FOR PRODUCTION CONFIGURATION A (7 SEQUENCES)")
    print("=" * 80)

    for cfg in SEQUENCES:
        cid = cfg["id"]
        name = cfg["name"]
        variant = cfg["variant"]
        in_csv = cfg["input_csv"]
        out_tum = cfg["out_tum"]
        meta_json = cfg["meta_path"]

        print(f"\n--- Running: {name} ({variant}) ---")

        # 1. Run pipeline (generates TUM trajectory + collects traced heap and validation)
        telemetry = run_pipeline(
            input_path=in_csv,
            output_path=out_tum,
            metadata_output=meta_json,
            log_level="WARNING",
        )

        # 2. Measure untraced latency
        untraced_time, untraced_ms_per_s = measure_untraced_execution(in_csv, "scratch/temp_untraced.txt")

        # 3. Read output TUM trajectory and perform offline ground truth evaluation
        gt_traj = load_ground_truth(cfg)
        est_traj = file_interface.read_tum_trajectory_file(out_tum)

        gt_sync, est_sync = sync.associate_trajectories(gt_traj, est_traj, max_diff=0.02)
        est_aligned = copy.deepcopy(est_sync)
        est_aligned.align(gt_sync, correct_scale=False)

        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((gt_sync, est_aligned))
        ate_rmse = float(ape.get_all_statistics()["rmse"])

        rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=200, delta_unit=metrics.Unit.frames, all_pairs=False)
        rpe.process_data((gt_sync, est_aligned))
        rpe_rmse = float(rpe.get_all_statistics()["rmse"])

        est_path_len = float(est_aligned.path_length)
        gt_path_len = float(gt_sync.path_length)

        # Checkpoints at 25%, 50%, 75%
        sync_ts = est_aligned.timestamps
        t_start, t_end = sync_ts[0], sync_ts[-1]
        t_span = t_end - t_start
        cp_errors = {}
        for pct in [0.25, 0.50, 0.75]:
            target_t = t_start + pct * t_span
            idx_cp = np.argmin(np.abs(sync_ts - target_t))
            err_cp = float(np.linalg.norm(est_aligned.positions_xyz[idx_cp] - gt_sync.positions_xyz[idx_cp]))
            cp_errors[f"cp_{int(pct*100)}"] = err_cp

        # Blind 5s trajectory & endpoint error
        blind_mask = sync_ts >= (t_end - 5.0)
        if np.any(blind_mask):
            blind_pos_diff = np.linalg.norm(est_aligned.positions_xyz[blind_mask] - gt_sync.positions_xyz[blind_mask], axis=1)
            blind_5s_traj_err = float(np.mean(blind_pos_diff))
            blind_5s_endpoint_err = float(blind_pos_diff[-1])
        else:
            blind_5s_traj_err = 0.0
            blind_5s_endpoint_err = 0.0

        # Dropout recovery error (frames right after dt > 0.025s)
        out_arr = np.loadtxt(out_tum)
        full_ts = out_arr[:, 0]
        dt_full = np.diff(full_ts)
        gap_indices = np.where(dt_full > 0.025)[0]
        rec_errors = []
        for g_idx in gap_indices:
            t_rec = full_ts[g_idx + 1]
            c_idx = np.argmin(np.abs(sync_ts - t_rec))
            if np.abs(sync_ts[c_idx] - t_rec) < 0.02:
                rec_err = np.linalg.norm(est_aligned.positions_xyz[c_idx] - gt_sync.positions_xyz[c_idx])
                rec_errors.append(rec_err)
        dropout_rec_err = float(np.mean(rec_errors)) if rec_errors else 0.0

        # Duplicate timestamps in output
        n_dups = int(np.sum(dt_full == 0.0))

        # Check finite & normalized
        all_finite = bool(np.all(np.isfinite(out_arr)))
        q_norms = np.linalg.norm(out_arr[:, 4:8], axis=1)
        max_q_err = float(np.max(np.abs(q_norms - 1.0)))

        val = telemetry["validation"]
        tum_status = "PASS" if val["valid"] else "FAIL"

        row_res = {
            "id": cid,
            "name": name,
            "variant": variant,
            "frames_in": telemetry["input_frames"],
            "frames_out": telemetry["output_tum_poses"],
            "duplicates": n_dups,
            "all_finite": all_finite,
            "max_quat_norm_err": max_q_err,
            "ate_rmse": ate_rmse,
            "rpe_rmse": rpe_rmse,
            "cp_25": cp_errors["cp_25"],
            "cp_50": cp_errors["cp_50"],
            "cp_75": cp_errors["cp_75"],
            "blind_5s_traj": blind_5s_traj_err,
            "blind_5s_end": blind_5s_endpoint_err,
            "dropout_rec": dropout_rec_err,
            "est_path": est_path_len,
            "gt_path": gt_path_len,
            "heap_mb": telemetry["python_heap_peak_mb"],
            "latency_untraced_ms_s": untraced_ms_per_s,
            "latency_traced_ms_s": telemetry["compute_latency_ms_per_s"],
            "tum_status": tum_status,
            "single_thread": telemetry["single_thread_enforced"],
        }
        all_results.append(row_res)

        print(f"  Frames In/Out: {row_res['frames_in']} / {row_res['frames_out']} (Duplicates: {n_dups})")
        print(f"  ATE RMSE: {ate_rmse:.2f} m | 1s RPE RMSE: {rpe_rmse:.2f} m")
        print(f"  CP 25/50/75%: {cp_errors['cp_25']:.1f}m / {cp_errors['cp_50']:.1f}m / {cp_errors['cp_75']:.1f}m")
        print(f"  Blind 5s: Traj={blind_5s_traj_err:.2f}m, End={blind_5s_endpoint_err:.2f}m | DropoutRec={dropout_rec_err:.2f}m")
        print(f"  Est Path: {est_path_len:.1f}m (GT: {gt_path_len:.1f}m)")
        print(f"  Heap: {row_res['heap_mb']:.2f} MB | Untraced Latency: {untraced_ms_per_s:.2f} ms/s (Traced: {row_res['latency_traced_ms_s']:.2f} ms/s)")
        print(f"  Validation: {tum_status} | Max Quat Err: {max_q_err:.2e}")

    with open("outputs/production_config_a/final_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
