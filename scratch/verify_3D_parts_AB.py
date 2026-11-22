"""
Stage 3D-Verification — Parts A and B
READ-ONLY audit: re-runs the ZUPT integration using the SAME src/ modules
used in the actual Stage 3D experiment, then inspects vel/pos arrays and
the saved output file. No files are created or modified.
"""
import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from data_loader import load_imu_data, load_groundtruth_data, IMU_TIMESTAMP_COL
from preprocessing import GYRO_COLS, ACCEL_COLS
from bias_estimation import detect_quasi_stationary_intervals, estimate_gyro_bias
from trajectory import (
    compute_gravity_alignment_quaternion,
    rotate_and_compensate_gravity,
    integrate_trajectory,
    STANDARD_GRAVITY,
)
from degradation import sanitize_imu_measurements

IMU_PATH = "data/raw/euroc/machine_hall/MH_02_easy/mav0/imu0/data.csv"
GT_PATH  = "data/raw/euroc/machine_hall/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"
ZUPT_OUT = "outputs/exp_MH_02_easy_zupt.txt"

MICRO_GAP_THRESHOLD_S = 0.025

print("=" * 70)
print("Loading and preparing IMU data (identical to Stage 3D pipeline)...")
print("=" * 70)

df_raw = load_imu_data(IMU_PATH)
df_degraded = df_raw.copy()  # S4-01 baseline = no degradation
df_sanitized, san_meta = sanitize_imu_measurements(df_degraded)
print(f"IMU samples N = {len(df_sanitized)}")

# --- Gyro bias (same call as main.py) ---
bias_result  = estimate_gyro_bias(df_sanitized, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
gyro_bias    = bias_result["gyro_bias"]
print(f"Gyro bias [rad/s]: {gyro_bias}")

df_proc = df_sanitized.copy()
valid_gyro_mask = df_proc["valid_gyro"].to_numpy(dtype=bool)
for col_idx, col in enumerate(GYRO_COLS):
    df_proc.loc[valid_gyro_mask, col] = df_proc.loc[valid_gyro_mask, col] - gyro_bias[col_idx]

# --- Initial gravity alignment ---
initial_accel_sample = df_proc.loc[df_proc["valid_accel"], ACCEL_COLS].iloc[:100].mean().to_numpy()
q0 = compute_gravity_alignment_quaternion(initial_accel_sample, yaw_rad=0.0)
print(f"Initial quaternion q0 [w,x,y,z]: {q0}")

# --- Detect stationary intervals (ZUPT detector — same call as Stage 3D) ---
det = detect_quasi_stationary_intervals(df_proc, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
intervals = det["detected_intervals"]  # list of 5-tuples (start_idx, end_idx, t_s, t_e, dur_s)
print(f"\nDetected stationary intervals (count={len(intervals)}):")
for iv in intervals:
    print(f"  idx [{iv[0]}, {iv[1]}]  t=[{iv[2]*1e-9:.3f}, {iv[3]*1e-9:.3f}] s  dur={iv[4]:.3f} s")

# --- Orientation integration (same loop as main.py lines 374-390) ---
n_samples = len(df_proc)
ts_ns = df_proc[IMU_TIMESTAMP_COL].to_numpy(dtype=np.int64)
dt_s_arr = np.zeros(n_samples, dtype=np.float64)
if n_samples > 1:
    dt_s_arr[1:] = (ts_ns[1:] - ts_ns[:-1]).astype(np.float64) * 1e-9

gyro_rates = df_proc[GYRO_COLS].to_numpy(dtype=np.float64)

def _rotvec_to_dq_wxyz(rv):
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * angle
    s = np.sin(half) / angle
    return np.array([np.cos(half), rv[0]*s, rv[1]*s, rv[2]*s])

def _qmul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

quats_wxyz = np.empty((n_samples, 4), dtype=np.float64)
curr_q = q0.copy()
quats_wxyz[0] = curr_q

for i in range(1, n_samples):
    is_macro_gap = dt_s_arr[i] > MICRO_GAP_THRESHOLD_S
    both_gyro_valid = bool(valid_gyro_mask[i-1]) and bool(valid_gyro_mask[i])
    if both_gyro_valid and not is_macro_gap:
        w_mid = 0.5 * (gyro_rates[i-1] + gyro_rates[i])
        dq = _rotvec_to_dq_wxyz(w_mid * dt_s_arr[i])
        curr_q = _qmul_wxyz(curr_q, dq)
        curr_q /= np.linalg.norm(curr_q)
    quats_wxyz[i] = curr_q

# --- ZUPT mask ---
zupt_mask = np.zeros(n_samples, dtype=bool)
if intervals:
    s_idx, e_idx = intervals[0][0], intervals[0][1]
    zupt_mask[s_idx:e_idx+1] = True
    print(f"\nZUPT mask applied: [{s_idx}, {e_idx}]  ({zupt_mask.sum()} samples)")
else:
    print("\nWARNING: No stationary interval detected — ZUPT mask is empty!")
    s_idx, e_idx = None, None

# --- Trajectory integration with ZUPT (via integrate_trajectory) ---
df_traj = integrate_trajectory(
    df_imu=df_proc,
    quaternions_wxyz=quats_wxyz,
    enable_zupt=True,
    zupt_mask=zupt_mask,
    gap_threshold_s=MICRO_GAP_THRESHOLD_S,
)
pos_arr  = df_traj[["p_x", "p_y", "p_z"]].to_numpy()
vel_arr  = df_traj[["v_x", "v_y", "v_z"]].to_numpy()

print("\n" + "=" * 70)
print("PART A: Velocity at index 5179 (immediately BEFORE ZUPT at 5180)")
print("=" * 70)
v5179 = vel_arr[5179]
print(f"  vel[5179]  X = {v5179[0]:+.6f} m/s")
print(f"  vel[5179]  Y = {v5179[1]:+.6f} m/s")
print(f"  vel[5179]  Z = {v5179[2]:+.6f} m/s")
print(f"  vel[5179] |v| = {np.linalg.norm(v5179):.6f} m/s")

# --- Compare with GT velocity at the same timestamp ---
print("\nLoading GT for velocity comparison at index 5179...")
gt_df = load_groundtruth_data(GT_PATH)
gt_ts  = gt_df["timestamp"].to_numpy(dtype=np.int64)
gt_vx  = gt_df["v_RS_R_x [m s^-1]"].to_numpy(dtype=np.float64)
gt_vy  = gt_df["v_RS_R_y [m s^-1]"].to_numpy(dtype=np.float64)
gt_vz  = gt_df["v_RS_R_z [m s^-1]"].to_numpy(dtype=np.float64)

target_ts = ts_ns[5179]
nearest_gt_idx = int(np.argmin(np.abs(gt_ts - target_ts)))
dt_match_s = (gt_ts[nearest_gt_idx] - target_ts) * 1e-9
gt_vel = np.array([gt_vx[nearest_gt_idx], gt_vy[nearest_gt_idx], gt_vz[nearest_gt_idx]])

print(f"  IMU ts[5179]  = {target_ts} ns  ({target_ts*1e-9:.4f} s)")
print(f"  Nearest GT idx = {nearest_gt_idx}, ts = {gt_ts[nearest_gt_idx]} ns ({gt_ts[nearest_gt_idx]*1e-9:.4f} s), dt = {dt_match_s*1000:.2f} ms")
print(f"  GT vel at t≈25.9s:  X={gt_vel[0]:+.6f}  Y={gt_vel[1]:+.6f}  Z={gt_vel[2]:+.6f} m/s  |v|={np.linalg.norm(gt_vel):.4f} m/s")
err = v5179 - gt_vel
print(f"  Velocity error (est - GT):  X={err[0]:+.6f}  Y={err[1]:+.6f}  Z={err[2]:+.6f}  |e|={np.linalg.norm(err):.4f} m/s")

# --- Cumulative accel breakdown 0 → 5179 (using computed a_lin_world) ---
a_lin_world = df_traj[["a_lin_x", "a_lin_y", "a_lin_z"]].to_numpy()
cum_a_lin = np.zeros(3)
for i in range(1, 5180):
    dt = dt_s_arr[i]
    # trapezoidal contribution from segment [i-1, i]
    cum_a_lin += 0.5 * (a_lin_world[i-1] + a_lin_world[i]) * dt
print(f"\n  Cumulative trapezoidal a_lin_world integral [0→5179]:")
print(f"    X={cum_a_lin[0]:+.4f}  Y={cum_a_lin[1]:+.4f}  Z={cum_a_lin[2]:+.4f}  |net|={np.linalg.norm(cum_a_lin):.4f} m/s")
print(f"  (Should match vel[5179] closely; discrepancy reveals ZUPT boundary effect if any)")

# Verify vel[5179] matches pure no-ZUPT velocity (ZUPT only starts at 5180)
print(f"\n  Verification: ZUPT mask[5179]={zupt_mask[5179]}  (expected False)")
print(f"  Verification: ZUPT mask[5180]={zupt_mask[5180]}  (expected True)")
print(f"  Verification: vel[5180] = {vel_arr[5180]} (expected [0,0,0])")
print(f"  Verification: pos[5180] = {pos_arr[5180]}")
print(f"  Verification: pos[5179] = {pos_arr[5179]}")
print(f"  pos[5180] == pos[5179]? {np.allclose(pos_arr[5180], pos_arr[5179], atol=1e-12)}")

print("\n--- Sample velocities around ZUPT boundary ---")
for idx in [5175, 5176, 5177, 5178, 5179, 5180, 5181, 5979, 5980, 5981]:
    print(f"  vel[{idx:5d}] = [{vel_arr[idx,0]:+.4f}, {vel_arr[idx,1]:+.4f}, {vel_arr[idx,2]:+.4f}]  |v|={np.linalg.norm(vel_arr[idx]):.4f}  ZUPT={zupt_mask[idx]}")

print("\n" + "=" * 70)
print("PART B: Verify ZUPT output file")
print("=" * 70)
import os
size_bytes = os.path.getsize(ZUPT_OUT)
print(f"File: {ZUPT_OUT}  size={size_bytes:,} bytes ({size_bytes/1024:.1f} KB)")

poses = []
with open(ZUPT_OUT) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        poses.append([float(x) for x in parts])
poses = np.array(poses)
print(f"Row count in ZUPT output: {len(poses)} (expected 30,400)")
print(f"Columns per row: {poses.shape[1]} (expected 8: ts tx ty tz qx qy qz qw)")

# TUM format: timestamp tx ty tz qx qy qz qw
q_out = poses[:, 4:8]   # qx qy qz qw in TUM
norms = np.linalg.norm(q_out, axis=1)
print(f"\nQuaternion norm stats:  min={norms.min():.8f}  max={norms.max():.8f}  mean={norms.mean():.8f}")
finite = np.all(np.isfinite(poses), axis=1)
print(f"All-finite rows: {finite.sum()} / {len(poses)}")
print(f"Non-finite rows: {(~finite).sum()}")

# Timestamp column check
out_ts = poses[:, 0]
print(f"\nOutput timestamp range: {out_ts[0]:.9f} → {out_ts[-1]:.9f} s")
print(f"Timestamps monotonic:   {bool(np.all(np.diff(out_ts) >= 0))}")

# ZUPT boundary checks using in-memory arrays
print("\n--- ZUPT boundary checks (from in-memory integration) ---")
print(f"B1. vel[5179] norm = {np.linalg.norm(vel_arr[5179]):.6f} m/s  (pre-ZUPT, should be non-zero)")
print(f"B2. vel[5180] = {vel_arr[5180]}  (first ZUPT, expected [0,0,0])")
print(f"B3. vel zeroed at 5180: {np.allclose(vel_arr[5180], 0, atol=1e-14)}")
print(f"B4. pos[5180] == pos[5179]: {np.allclose(pos_arr[5180], pos_arr[5179], atol=1e-14)}")

# Transition OUT: integrate_trajectory at ZUPT boundary uses trapezoidal v_next,
# but ZUPT zeroes v_next → pos update is 0.5*(v[5979]+v_next)*dt = 0.5*(0+0)*dt = 0
# So pos[5980] = pos[5979] + 0.5*(vel[5979]+vel[5980])*dt
dt_5980 = dt_s_arr[5980]
expected_pos_5980 = pos_arr[5979] + 0.5 * (vel_arr[5979] + vel_arr[5980]) * dt_5980
print(f"\nB5. Transition out of ZUPT at index 5980:")
print(f"    vel[5979] = {vel_arr[5979]}  ZUPT={zupt_mask[5979]}")
print(f"    vel[5980] = {vel_arr[5980]}  ZUPT={zupt_mask[5980]}")
print(f"    pos[5979] = {pos_arr[5979]}")
print(f"    pos[5980] = {pos_arr[5980]}")
print(f"    expected pos[5980] (trapezoidal) = {expected_pos_5980}")
print(f"    Transition correct: {np.allclose(pos_arr[5980], expected_pos_5980, atol=1e-10)}")

# Verify pre-ZUPT is identical to pure baseline
print("\n--- B6. Pre-ZUPT baseline identity check (idx 0 → 5179) ---")
df_base = integrate_trajectory(
    df_imu=df_proc,
    quaternions_wxyz=quats_wxyz,
    enable_zupt=False,
    gap_threshold_s=MICRO_GAP_THRESHOLD_S,
)
pos_base = df_base[["p_x", "p_y", "p_z"]].to_numpy()
vel_base = df_base[["v_x", "v_y", "v_z"]].to_numpy()

max_pos_diff = np.max(np.abs(pos_arr[:5180] - pos_base[:5180]))
max_vel_diff = np.max(np.abs(vel_arr[:5180] - vel_base[:5180]))
print(f"    Max |pos_zupt - pos_base|[:5180] = {max_pos_diff:.2e} m (expected ~0)")
print(f"    Max |vel_zupt - vel_base|[:5180] = {max_vel_diff:.2e} m/s (expected ~0)")
print(f"    Pre-ZUPT identical: {max_pos_diff < 1e-10 and max_vel_diff < 1e-10}")

print(f"\nB7. Number of detected intervals: {len(intervals)} (expected 1)")
if intervals:
    s, e = intervals[0][0], intervals[0][1]
    print(f"    Interval indices: [{s}, {e}]  ZUPT samples: {e-s+1}")
    print(f"    Expected [5180, 5979] → 800 samples")
    print(f"    Match: start={s==5180}, end={e==5979}")

print("\nB8. Sample count verification:")
print(f"    Total ZUPT samples (mask True): {zupt_mask.sum()}")
print(f"    Expected: 800 (indices 5180–5979 inclusive)")

print("\n--- Parts A and B COMPLETE ---")
