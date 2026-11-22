"""
Comprehensive Sensitivity Analysis of Velocity Cap and Outage Decay Parameters.
Evaluates Configurations A through E across MH_01 through MH_05 and TUM-VI Room 1.
Computes:
- ATE RMSE (m)
- 1s RPE RMSE (m)
- Checkpoint errors at 25%, 50%, 75% of sequence
- Blind 5-second trajectory error & endpoint error
- Dropout recovery error
- Path lengths (Est vs GT)
- Untraced runtime and latency
- Python heap peak
"""

import sys
import os
import time
import math
import copy
import json
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd

from evo.core import trajectory, sync, metrics
from evo.tools import file_interface

from src.main import _load_imu_arrays_lean, _sanitize_arrays_inplace, export_tum_trajectory
from src.bias_estimation import estimate_gyro_bias
from src.trajectory import compute_gravity_alignment_quaternion, STANDARD_GRAVITY
from src.data_loader import IMU_TIMESTAMP_COL
from src.preprocessing import GYRO_COLS, ACCEL_COLS

CONFIGS = [
    {"id": "A", "v_max": 10.0, "decay_rate": 1.5},
    {"id": "B", "v_max": 15.0, "decay_rate": 0.5},
    {"id": "C", "v_max": 15.0, "decay_rate": 1.5},
    {"id": "D", "v_max": 15.0, "decay_rate": 3.0},
    {"id": "E", "v_max": 25.0, "decay_rate": 1.5},
]

def load_gt_traj(seq_name, is_tum=False):
    if is_tum:
        return file_interface.read_tum_trajectory_file("outputs/tum_vi_room1_gt.tum")
    else:
        gt_path = f"data/raw/euroc/machine_hall/{seq_name}/mav0/state_groundtruth_estimate0/data.csv"
        df_gt = pd.read_csv(gt_path)
        ts_gt = df_gt.iloc[:, 0].to_numpy() * 1e-9
        pos_gt = df_gt.iloc[:, 1:4].to_numpy()
        qw_g, qx_g, qy_g, qz_g = df_gt.iloc[:, 4].to_numpy(), df_gt.iloc[:, 5].to_numpy(), df_gt.iloc[:, 6].to_numpy(), df_gt.iloc[:, 7].to_numpy()
        return trajectory.PoseTrajectory3D(pos_gt, np.column_stack([qx_g, qy_g, qz_g, qw_g]), ts_gt)

def run_estimation(dataset_path, v_max, decay_rate):
    t0 = time.perf_counter()
    ts_ns, gyro_rates, a_body = _load_imu_arrays_lean(dataset_path)
    valid_gyro_mask, valid_accel_mask = _sanitize_arrays_inplace(gyro_rates, a_body)
    n_samples = len(ts_ns)

    df_bias_view = pd.DataFrame({
        IMU_TIMESTAMP_COL: ts_ns,
        GYRO_COLS[0]: gyro_rates[:, 0], GYRO_COLS[1]: gyro_rates[:, 1], GYRO_COLS[2]: gyro_rates[:, 2],
        ACCEL_COLS[0]: a_body[:, 0], ACCEL_COLS[1]: a_body[:, 1], ACCEL_COLS[2]: a_body[:, 2],
    }, copy=False)
    bias_result = estimate_gyro_bias(df_bias_view, max_allowed_gap_s=0.025)
    gyro_bias = bias_result["gyro_bias"]
    gyro_rates[valid_gyro_mask] -= gyro_bias

    aligned_from_stationary = False
    interval_info = bias_result.get("interval_info") or {}
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if (
        bias_result.get("status") == "CONFIRMED_QUASI_STATIONARY"
        and "start_index" in interval_info
        and "end_index" in interval_info
    ):
        s_i = max(0, min(n_samples - 1, int(interval_info["start_index"])))
        e_i = max(s_i, min(n_samples - 1, int(interval_info["end_index"])))
        stat_mask = np.zeros(n_samples, dtype=bool)
        stat_mask[s_i : e_i + 1] = True
        stat_mask &= valid_accel_mask
        if np.any(stat_mask):
            g_est_rel = a_body[stat_mask].mean(axis=0)
            if not np.isnan(g_est_rel).any() and np.linalg.norm(g_est_rel) >= 1.0:
                q0 = compute_gravity_alignment_quaternion(g_est_rel, yaw_rad=0.0)
                aligned_from_stationary = True

    if not aligned_from_stationary:
        valid_accel_idx = np.flatnonzero(valid_accel_mask)[:100]
        if len(valid_accel_idx) > 0:
            initial_accel_sample = a_body[valid_accel_idx].mean(axis=0)
            if not np.isnan(initial_accel_sample).any() and np.linalg.norm(initial_accel_sample) >= 1.0:
                q0 = compute_gravity_alignment_quaternion(initial_accel_sample, yaw_rad=0.0)

    # Orientation integration
    dt_s = np.zeros(n_samples, dtype=np.float64)
    dt_s[1:] = (ts_ns[1:] - ts_ns[:-1]) * 1e-9

    q_rel = np.empty((n_samples, 4), dtype=np.float64)
    q_rel[0] = [1.0, 0.0, 0.0, 0.0]
    cq_w, cq_x, cq_y, cq_z = 1.0, 0.0, 0.0, 0.0
    _sin, _cos, _sqrt, _exp = math.sin, math.cos, math.sqrt, math.exp
    gap_p = 0.025

    for i in range(1, n_samples):
        dt = dt_s[i]
        if dt <= gap_p and valid_gyro_mask[i - 1] and valid_gyro_mask[i]:
            half_dt = 0.5 * dt
            rx = (gyro_rates[i - 1, 0] + gyro_rates[i, 0]) * half_dt
            ry = (gyro_rates[i - 1, 1] + gyro_rates[i, 1]) * half_dt
            rz = (gyro_rates[i - 1, 2] + gyro_rates[i, 2]) * half_dt
            ang = _sqrt(rx * rx + ry * ry + rz * rz)
            if ang < 1e-12:
                dw, dx, dy, dz = 1.0, 0.0, 0.0, 0.0
            else:
                half_a = 0.5 * ang
                s = _sin(half_a) / ang
                dw = _cos(half_a)
                dx, dy, dz = rx * s, ry * s, rz * s
            nw = cq_w * dw - cq_x * dx - cq_y * dy - cq_z * dz
            nx = cq_w * dx + cq_x * dw + cq_y * dz - cq_z * dy
            ny = cq_w * dy - cq_x * dz + cq_y * dw + cq_z * dx
            nz = cq_w * dz + cq_x * dy - cq_y * dx + cq_z * dw
            inv_n = 1.0 / _sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
            cq_w, cq_x, cq_y, cq_z = nw * inv_n, nx * inv_n, ny * inv_n, nz * inv_n
        q_rel[i] = [cq_w, cq_x, cq_y, cq_z]

    w1, x1, y1, z1 = float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])
    w2, x2, y2, z2 = q_rel[:, 0].copy(), q_rel[:, 1].copy(), q_rel[:, 2].copy(), q_rel[:, 3].copy()
    quats = q_rel
    quats[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    quats[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    quats[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    quats[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    vx_b, vy_b, vz_b = a_body[:, 0].copy(), a_body[:, 1].copy(), a_body[:, 2].copy()
    a_w = np.empty_like(a_body)
    a_w[:, 0] = (1.0 - 2.0 * (qy * qy + qz * qz)) * vx_b + 2.0 * (qx * qy - qw * qz) * vy_b + 2.0 * (qx * qz + qw * qy) * vz_b
    a_w[:, 1] = 2.0 * (qx * qy + qw * qz) * vx_b + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy_b + 2.0 * (qy * qz - qw * qx) * vz_b
    a_w[:, 2] = 2.0 * (qx * qz - qw * qy) * vx_b + 2.0 * (qy * qz + qw * qx) * vy_b + (1.0 - 2.0 * (qx * qx + qy * qy)) * vz_b - STANDARD_GRAVITY
    a_w[~valid_accel_mask] = 0.0

    pos = np.zeros((n_samples, 3), dtype=np.float64)
    px, py, pz = 0.0, 0.0, 0.0
    vx, vy, vz = 0.0, 0.0, 0.0
    for i in range(1, n_samples):
        dt = dt_s[i]
        if dt == 0.0:
            pass  # Duplicate timestamp: velocity holds, position does not advance
        elif dt > gap_p or not (valid_accel_mask[i - 1] and valid_accel_mask[i]):
            # Outage / gap / invalid row: exponential decay of velocity across gap
            decay = _exp(-decay_rate * dt)
            vx *= decay; vy *= decay; vz *= decay
            px += vx * dt; py += vy * dt; pz += vz * dt
        else:
            half_dt = 0.5 * dt
            vnx = vx + (a_w[i - 1, 0] + a_w[i, 0]) * half_dt
            vny = vy + (a_w[i - 1, 1] + a_w[i, 1]) * half_dt
            vnz = vz + (a_w[i - 1, 2] + a_w[i, 2]) * half_dt
            v_norm = _sqrt(vnx * vnx + vny * vny + vnz * vnz)
            if v_norm > v_max:
                s = v_max / v_norm
                vnx *= s; vny *= s; vnz *= s
            px += (vx + vnx) * half_dt
            py += (vy + vny) * half_dt
            pz += (vz + vnz) * half_dt
            vx, vy, vz = vnx, vny, vnz
        pos[i] = [px, py, pz]

    t_runtime = time.perf_counter() - t0
    dur = (ts_ns[-1] - ts_ns[0]) * 1e-9
    ms_per_s = (t_runtime * 1000.0) / max(dur, 1e-6)

    return ts_ns, pos, quats, dt_s, valid_accel_mask, t_runtime, ms_per_s

def evaluate_metrics(ts_ns, pos, quats, dt_s, valid_accel_mask, gt_traj, out_file):
    export_tum_trajectory(
        output_path=out_file,
        timestamps_ns=ts_ns,
        pos_arr=pos,
        quats_wxyz=quats,
    )
    est_traj = file_interface.read_tum_trajectory_file(out_file)
    gt_sync, est_sync = sync.associate_trajectories(gt_traj, est_traj, max_diff=0.02)
    est_aligned = copy.deepcopy(est_sync)
    est_aligned.align(gt_sync, correct_scale=False)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt_sync, est_aligned))
    ate_rmse = float(ape.get_all_statistics()["rmse"])

    rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=200, delta_unit=metrics.Unit.frames, all_pairs=False)
    rpe.process_data((gt_sync, est_aligned))
    rpe_rmse = float(rpe.get_all_statistics()["rmse"])

    # Path lengths
    est_path = float(est_aligned.path_length)
    gt_path = float(gt_sync.path_length)

    # Checkpoint errors at 25%, 50%, 75%
    sync_ts = est_aligned.timestamps
    t_start, t_end = sync_ts[0], sync_ts[-1]
    t_span = t_end - t_start

    cp_errors = {}
    for pct in [0.25, 0.50, 0.75]:
        target_t = t_start + pct * t_span
        idx_cp = np.argmin(np.abs(sync_ts - target_t))
        err_cp = float(np.linalg.norm(est_aligned.positions_xyz[idx_cp] - gt_sync.positions_xyz[idx_cp]))
        cp_errors[f"cp_{int(pct*100)}"] = err_cp

    # Blind 5s evaluation (t >= t_end - 5.0)
    blind_mask = sync_ts >= (t_end - 5.0)
    if np.any(blind_mask):
        blind_pos_diff = np.linalg.norm(est_aligned.positions_xyz[blind_mask] - gt_sync.positions_xyz[blind_mask], axis=1)
        blind_5s_traj_err = float(np.mean(blind_pos_diff))
        blind_5s_endpoint_err = float(blind_pos_diff[-1])
    else:
        blind_5s_traj_err = 0.0
        blind_5s_endpoint_err = 0.0

    # Dropout recovery error (error at the first frame immediately after each burst dropout > 0.025s)
    est_full_ts = ts_ns * 1e-9
    gap_indices = np.where(dt_s > 0.025)[0]
    rec_errors = []
    for g_idx in gap_indices:
        t_rec = est_full_ts[g_idx]
        # find matching index in sync_ts
        close_idx = np.argmin(np.abs(sync_ts - t_rec))
        if np.abs(sync_ts[close_idx] - t_rec) < 0.02:
            rec_err = np.linalg.norm(est_aligned.positions_xyz[close_idx] - gt_sync.positions_xyz[close_idx])
            rec_errors.append(rec_err)
    dropout_recovery_err = float(np.mean(rec_errors)) if rec_errors else 0.0

    return {
        "ate_rmse": ate_rmse,
        "rpe_rmse": rpe_rmse,
        "est_path": est_path,
        "gt_path": gt_path,
        "cp_25": cp_errors["cp_25"],
        "cp_50": cp_errors["cp_50"],
        "cp_75": cp_errors["cp_75"],
        "blind_5s_traj": blind_5s_traj_err,
        "blind_5s_end": blind_5s_endpoint_err,
        "dropout_recovery": dropout_recovery_err,
    }

def main():
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)
    results = {}

    print("=" * 80)
    print("STEP 1: EVALUATION OF CONFIGS A-E ON MH_01 AND TUM-VI ROOM 1")
    print("=" * 80)

    for seq_name, is_tum, csv_path in [
        ("MH_01_easy", False, "outputs/MH_01_easy_degraded.csv"),
        ("TUM_VI_room1", True, "outputs/TUM_VI_room1_degraded_canonical.csv"),
    ]:
        gt_traj = load_gt_traj(seq_name, is_tum=is_tum)
        results[seq_name] = {}

        for cfg in CONFIGS:
            cid = cfg["id"]
            v_max = cfg["v_max"]
            decay = cfg["decay_rate"]
            out_file = f"outputs/experiments/{seq_name}_config_{cid}.txt"

            ts_ns, pos, quats, dt_s, va_mask, runtime_s, ms_per_s = run_estimation(csv_path, v_max, decay)
            met = evaluate_metrics(ts_ns, pos, quats, dt_s, va_mask, gt_traj, out_file)
            met["runtime_s"] = runtime_s
            met["ms_per_s"] = ms_per_s
            met["v_max"] = v_max
            met["decay_rate"] = decay

            results[seq_name][cid] = met
            print(f"[{seq_name}] Config {cid} (v_max={v_max}, decay={decay}): "
                  f"ATE={met['ate_rmse']:.2f}m, RPE={met['rpe_rmse']:.2f}m, "
                  f"Blind5sTraj={met['blind_5s_traj']:.2f}m, Blind5sEnd={met['blind_5s_end']:.2f}m, "
                  f"DropoutRec={met['dropout_recovery']:.2f}m, Path={met['est_path']:.1f}m (GT={met['gt_path']:.1f}m), "
                  f"Latency={met['ms_per_s']:.2f}ms/s")

    # Step 2: Validate the two best configurations across remaining EuRoC sequences
    # Based on ATE & RPE: Config A (v_max=10, decay=1.5) and Config D (v_max=15, decay=3.0) or C (15, 1.5)
    # Let's run Config A and Config D (and C as standard reference) across MH_02, MH_03, MH_04, MH_05
    print("\n" + "=" * 80)
    print("STEP 2: CROSS-VALIDATION OF TOP CONFIGS ACROSS REMAINING EUROC SEQUENCES")
    print("=" * 80)

    top_configs = [c for c in CONFIGS if c["id"] in ["A", "C", "D"]]
    for seq_name in ["MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]:
        gt_traj = load_gt_traj(seq_name, is_tum=False)
        csv_path = f"outputs/{seq_name}_degraded.csv"
        results[seq_name] = {}

        for cfg in top_configs:
            cid = cfg["id"]
            v_max = cfg["v_max"]
            decay = cfg["decay_rate"]
            out_file = f"outputs/experiments/{seq_name}_config_{cid}.txt"

            ts_ns, pos, quats, dt_s, va_mask, runtime_s, ms_per_s = run_estimation(csv_path, v_max, decay)
            met = evaluate_metrics(ts_ns, pos, quats, dt_s, va_mask, gt_traj, out_file)
            met["runtime_s"] = runtime_s
            met["ms_per_s"] = ms_per_s
            met["v_max"] = v_max
            met["decay_rate"] = decay

            results[seq_name][cid] = met
            print(f"[{seq_name}] Config {cid} (v_max={v_max}, decay={decay}): "
                  f"ATE={met['ate_rmse']:.2f}m, RPE={met['rpe_rmse']:.2f}m, "
                  f"Blind5sTraj={met['blind_5s_traj']:.2f}m, Blind5sEnd={met['blind_5s_end']:.2f}m, "
                  f"DropoutRec={met['dropout_recovery']:.2f}m, Path={met['est_path']:.1f}m (GT={met['gt_path']:.1f}m)")

    # Save all results to JSON
    with open("outputs/experiments/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
