"""
Stage 3D-Verification — Parts A through D (READ-ONLY)
Replicates Stage 3D integration exactly, then audits every claimed result.
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
BASE_OUT = "outputs/MH_02_easy_est.txt"
MICRO_GAP_THRESHOLD_S = 0.025

def sep(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ============================================================
# SHARED SETUP (identical to Stage 3D / main.py pipeline)
# ============================================================
sep("SHARED SETUP: IMU + GT loading, bias, orientation, trajectory")

df_raw = load_imu_data(IMU_PATH)
df_degraded = df_raw.copy()
df_sanitized, san_meta = sanitize_imu_measurements(df_degraded)
N = len(df_sanitized)
print(f"IMU samples N = {N}")

bias_result = estimate_gyro_bias(df_sanitized, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
gyro_bias   = bias_result["gyro_bias"]
print(f"Gyro bias [rad/s]: {gyro_bias}")

df_proc = df_sanitized.copy()
valid_gyro_mask = df_proc["valid_gyro"].to_numpy(dtype=bool)
for ci, col in enumerate(GYRO_COLS):
    df_proc.loc[valid_gyro_mask, col] = df_proc.loc[valid_gyro_mask, col] - gyro_bias[ci]

initial_accel_sample = df_proc.loc[df_proc["valid_accel"], ACCEL_COLS].iloc[:100].mean().to_numpy()
q0 = compute_gravity_alignment_quaternion(initial_accel_sample, yaw_rad=0.0)
print(f"q0 [w,x,y,z] = {q0}")

det = detect_quasi_stationary_intervals(df_proc, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
intervals = det["detected_intervals"]
print(f"Detected intervals: {len(intervals)}")
for iv in intervals:
    print(f"  idx [{iv[0]}, {iv[1]}]  t=[{iv[0]*5e-3:.3f}, {iv[1]*5e-3:.3f}]  dur={iv[4]:.3f}s")

# --- Orientation integration (same as main.py) ---
ts_ns = df_proc[IMU_TIMESTAMP_COL].to_numpy(dtype=np.int64)
dt_s_arr = np.zeros(N, dtype=np.float64)
if N > 1:
    dt_s_arr[1:] = (ts_ns[1:] - ts_ns[:-1]).astype(np.float64) * 1e-9
gyro_rates = df_proc[GYRO_COLS].to_numpy(dtype=np.float64)

def _rotvec_to_dq(rv):
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * angle; s = np.sin(half) / angle
    return np.array([np.cos(half), rv[0]*s, rv[1]*s, rv[2]*s])

def _qmul(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])

quats = np.empty((N, 4), dtype=np.float64)
q = q0.copy(); quats[0] = q
for i in range(1, N):
    is_macro = dt_s_arr[i] > MICRO_GAP_THRESHOLD_S
    both_ok = bool(valid_gyro_mask[i-1]) and bool(valid_gyro_mask[i])
    if both_ok and not is_macro:
        w_mid = 0.5 * (gyro_rates[i-1] + gyro_rates[i])
        dq = _rotvec_to_dq(w_mid * dt_s_arr[i])
        q = _qmul(q, dq); q /= np.linalg.norm(q)
    quats[i] = q

# --- ZUPT mask ---
zupt_mask = np.zeros(N, dtype=bool)
ZUPT_S, ZUPT_E = None, None
if intervals:
    ZUPT_S, ZUPT_E = intervals[0][0], intervals[0][1]
    zupt_mask[ZUPT_S:ZUPT_E+1] = True
print(f"ZUPT mask: [{ZUPT_S}, {ZUPT_E}], count={zupt_mask.sum()}")

# --- ZUPT trajectory ---
df_zupt = integrate_trajectory(df_proc, quats, enable_zupt=True,
                               zupt_mask=zupt_mask,
                               gap_threshold_s=MICRO_GAP_THRESHOLD_S)
pos_z = df_zupt[["p_x","p_y","p_z"]].to_numpy()
vel_z = df_zupt[["v_x","v_y","v_z"]].to_numpy()

# --- Baseline trajectory (no ZUPT) ---
df_base = integrate_trajectory(df_proc, quats, enable_zupt=False,
                               gap_threshold_s=MICRO_GAP_THRESHOLD_S)
pos_b = df_base[["p_x","p_y","p_z"]].to_numpy()
vel_b = df_base[["v_x","v_y","v_z"]].to_numpy()

# --- GT ---
gt_df = load_groundtruth_data(GT_PATH)
gt_ts  = gt_df["timestamp"].to_numpy(dtype=np.int64)
gt_pos = gt_df[["p_RS_R_x","p_RS_R_y","p_RS_R_z"]].to_numpy(dtype=np.float64)
gt_vel = gt_df[["v_RS_R_x","v_RS_R_y","v_RS_R_z"]].to_numpy(dtype=np.float64)
gt_q   = gt_df[["q_RS_w","q_RS_x","q_RS_y","q_RS_z"]].to_numpy(dtype=np.float64)
gt_ba  = gt_df[["b_a_RS_S_x","b_a_RS_S_y","b_a_RS_S_z"]].to_numpy(dtype=np.float64)
print(f"GT samples: {len(gt_ts)}")

# ============================================================
# PART A: Velocity at index 5179 (immediately before ZUPT)
# ============================================================
sep("PART A: Velocity at index 5179 (last pre-ZUPT sample)")

v5179 = vel_z[5179]   # identical to baseline here (pre-ZUPT)
v5179_base = vel_b[5179]
print(f"vel_zupt[5179]  = [{v5179[0]:+.6f}, {v5179[1]:+.6f}, {v5179[2]:+.6f}]  |v|={np.linalg.norm(v5179):.6f} m/s")
print(f"vel_base[5179]  = [{v5179_base[0]:+.6f}, {v5179_base[1]:+.6f}, {v5179_base[2]:+.6f}]  |v|={np.linalg.norm(v5179_base):.6f} m/s")
print(f"Match (pre-ZUPT identical): {np.allclose(v5179, v5179_base, atol=1e-14)}")

target_ts = ts_ns[5179]
nearest_gt_idx = int(np.argmin(np.abs(gt_ts - target_ts)))
dt_match_ms = (gt_ts[nearest_gt_idx] - target_ts) * 1e-6
gt_v5179 = gt_vel[nearest_gt_idx]
print(f"\nIMU ts[5179]        = {target_ts} ns  ({target_ts*1e-9:.4f} s)")
print(f"Nearest GT idx      = {nearest_gt_idx}  dt = {dt_match_ms:.2f} ms")
print(f"GT vel at ~25.9s    = [{gt_v5179[0]:+.6f}, {gt_v5179[1]:+.6f}, {gt_v5179[2]:+.6f}]  |v|={np.linalg.norm(gt_v5179):.4f} m/s")
err_vel = v5179 - gt_v5179
print(f"Velocity error (est-GT) = [{err_vel[0]:+.6f}, {err_vel[1]:+.6f}, {err_vel[2]:+.6f}]  |e|={np.linalg.norm(err_vel):.4f} m/s")

# Verify the "25.1 m/s" claim
print(f"\nClaim was 25.1 m/s.  Actual |vel[5179]| = {np.linalg.norm(v5179):.4f} m/s")
print(f"Verdict: {'SUPPORTED' if abs(np.linalg.norm(v5179) - 25.1) < 0.5 else 'CONTRADICTED'} by data")

# Cumulative accel breakdown
a_lin = df_zupt[["a_lin_x","a_lin_y","a_lin_z"]].to_numpy()
cum = np.zeros(3)
for i in range(1, 5180):
    cum += 0.5 * (a_lin[i-1] + a_lin[i]) * dt_s_arr[i]
print(f"\nCumulative a_lin_world integral [0→5179] (trapezoidal):")
print(f"  X={cum[0]:+.4f}  Y={cum[1]:+.4f}  Z={cum[2]:+.4f}  |net|={np.linalg.norm(cum):.4f} m/s")

print("\nSample velocities around ZUPT boundary:")
for idx in [5178, 5179, 5180, 5181, 5979, 5980, 5981]:
    print(f"  vel[{idx:5d}] = [{vel_z[idx,0]:+.4f}, {vel_z[idx,1]:+.4f}, {vel_z[idx,2]:+.4f}]"
          f"  |v|={np.linalg.norm(vel_z[idx]):.4f}  ZUPT={zupt_mask[idx]}")

# ============================================================
# PART B: Verify ZUPT output file
# ============================================================
sep("PART B: Verify saved ZUPT output file")

size_bytes = os.path.getsize(ZUPT_OUT)
print(f"File: {ZUPT_OUT}  size={size_bytes:,} bytes")

poses_z = []
with open(ZUPT_OUT) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        poses_z.append([float(x) for x in line.split()])
poses_z = np.array(poses_z)

print(f"Row count: {len(poses_z)} (expected 30,400)")
print(f"Columns:   {poses_z.shape[1]} (expected 8: ts tx ty tz qx qy qz qw)")

# TUM format: ts tx ty tz qx qy qz qw
q_out = poses_z[:, 4:8]   # qx qy qz qw
norms = np.linalg.norm(q_out, axis=1)
print(f"\nQuaternion norm  min={norms.min():.8f}  max={norms.max():.8f}  mean={norms.mean():.8f}")
finite = np.all(np.isfinite(poses_z), axis=1)
print(f"All-finite rows: {finite.sum()} / {len(poses_z)}")

print(f"\nB1. vel[5180] = {vel_z[5180]}  → zeroed: {np.allclose(vel_z[5180], 0, atol=1e-14)}")
print(f"B2. vel[5979] = {vel_z[5979]}  → zeroed (last ZUPT): {np.allclose(vel_z[5979], 0, atol=1e-14)}")
print(f"B3. pos[5180] == pos[5179]: {np.allclose(pos_z[5180], pos_z[5179], atol=1e-14)}")

dt_5980 = dt_s_arr[5980]
expected_5980 = pos_z[5979] + 0.5*(vel_z[5979]+vel_z[5980])*dt_5980
print(f"B4. pos[5980] transition (trapezoidal): expected={expected_5980}, actual={pos_z[5980]}")
print(f"    correct: {np.allclose(pos_z[5980], expected_5980, atol=1e-10)}")

max_pos_pre = np.max(np.abs(pos_z[:5180] - pos_b[:5180]))
print(f"B5. Max |pos_zupt - pos_base|[:5180] = {max_pos_pre:.2e} m  (identical pre-ZUPT)")

print(f"B6. ZUPT interval [{ZUPT_S}, {ZUPT_E}], count={zupt_mask.sum()}")
print(f"    Expected [5180, 5979] = 800 samples → match: {ZUPT_S==5180 and ZUPT_E==5979 and zupt_mask.sum()==800}")

# ============================================================
# PART C: Reconcile evaluation metrics
# ============================================================
sep("PART C: Reconcile SE(3) ATE, RPE, and 1705.3 m vs 1738.52 m")

def umeyama_se3(p_est, p_gt):
    """SE(3) Umeyama (no scale). Returns R, t, R@p_est+t."""
    mu_e = p_est.mean(0); mu_g = p_gt.mean(0)
    E = p_est - mu_e; G = p_gt - mu_g
    H = (G.T @ E) / len(p_est)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R_opt = U @ D @ Vt
    t_opt = mu_g - R_opt @ mu_e
    p_aligned = (R_opt @ p_est.T).T + t_opt
    return R_opt, t_opt, p_aligned

# Build valid mask (IMU timestamps that have nearby GT)
MAX_INTERP_GAP_NS = int(0.1 * 1e9)  # 100 ms
valid_mask = np.zeros(N, dtype=bool)
gt_pos_interp = np.full((N, 3), np.nan)

for i in range(N):
    t = ts_ns[i]
    idx_lo = int(np.searchsorted(gt_ts, t, side='right')) - 1
    if idx_lo < 0 or idx_lo >= len(gt_ts)-1:
        continue
    idx_hi = idx_lo + 1
    if (gt_ts[idx_hi] - gt_ts[idx_lo]) > MAX_INTERP_GAP_NS:
        continue
    alpha = float(t - gt_ts[idx_lo]) / float(gt_ts[idx_hi] - gt_ts[idx_lo])
    gt_pos_interp[i] = (1-alpha)*gt_pos[idx_lo] + alpha*gt_pos[idx_hi]
    valid_mask[i] = True

n_valid = int(valid_mask.sum())
print(f"Valid IMU samples with GT interpolation: {n_valid}")

ts_v  = ts_ns[valid_mask]
p_z_v = pos_z[valid_mask]
p_b_v = pos_b[valid_mask]
p_gt_v = gt_pos_interp[valid_mask]

# SE(3) Umeyama ATE for ZUPT
R_z, t_z, p_z_al = umeyama_se3(p_z_v, p_gt_v)
det_Rz = np.linalg.det(R_z)
err_z = p_z_al - p_gt_v
ate_z = float(np.sqrt(np.mean(np.sum(err_z**2, axis=1))))
print(f"\nZUPT trajectory SE(3) ATE RMSE = {ate_z:.4f} m  (det(R)={det_Rz:.6f})")

# SE(3) Umeyama ATE for Baseline
R_b, t_b, p_b_al = umeyama_se3(p_b_v, p_gt_v)
det_Rb = np.linalg.det(R_b)
err_b = p_b_al - p_gt_v
ate_b = float(np.sqrt(np.mean(np.sum(err_b**2, axis=1))))
print(f"Baseline trajectory SE(3) ATE RMSE = {ate_b:.4f} m  (det(R)={det_Rb:.6f})")

improvement_pct = (ate_b - ate_z) / ate_b * 100
print(f"ATE improvement: {ate_b:.2f} → {ate_z:.2f} m  ({improvement_pct:.1f}%)")
print(f"Claimed improvement: 17.6% (2948→2430 m) — {'CLOSE' if abs(improvement_pct-17.6)<5 else 'DIFFERS'}")

# Unaligned endpoint analysis (reconcile 1705.3 vs 1738.52 m claims)
print("\n--- Endpoint difference analysis ---")
norm_base_last = np.linalg.norm(pos_b[-1])
norm_zupt_last = np.linalg.norm(pos_z[-1])
norm_diff      = norm_base_last - norm_zupt_last
print(f"||pos_base[-1]|| = {norm_base_last:.4f} m")
print(f"||pos_zupt[-1]|| = {norm_zupt_last:.4f} m")
print(f"Norm difference (||base||-||zupt||) = {norm_diff:.4f} m  ← claimed as 1705.3 m?")

# endpoint error vs GT (last valid GT timestamp)
gt_last_pos = gt_pos[-1]
err_base_last = np.linalg.norm(pos_b[-1] - gt_last_pos)  # but trajectories not aligned
err_zupt_last = np.linalg.norm(pos_z[-1] - gt_last_pos)
err_base_last_al = np.linalg.norm(pos_b[-1] - p_b_al[-1] + p_b_al[-1] - p_gt_v[-1])  # approximation

# Actually compute using last valid aligned point
err_base_aligned_end = np.linalg.norm(p_b_al[-1] - p_gt_v[-1])
err_zupt_aligned_end = np.linalg.norm(p_z_al[-1] - p_gt_v[-1])
print(f"\nAligned endpoint error  base = {err_base_aligned_end:.4f} m")
print(f"Aligned endpoint error  zupt = {err_zupt_aligned_end:.4f} m")
print(f"Aligned endpoint error diff  = {err_base_aligned_end - err_zupt_aligned_end:.4f} m  ← claimed 1738.52 m?")

# RPE 1-second
def compute_rpe_1s(ts_ns, pos, dt_target_s=1.0):
    rpe_list = []
    dt_ns = int(dt_target_s * 1e9)
    i = 0
    while i < len(ts_ns):
        j = int(np.searchsorted(ts_ns, ts_ns[i] + dt_ns, side='left'))
        if j >= len(ts_ns):
            break
        d_est = np.linalg.norm(pos[j] - pos[i])
        rpe_list.append(d_est)
        i = j
    return np.array(rpe_list)

rpe_z = compute_rpe_1s(ts_v, p_z_v)
rpe_b = compute_rpe_1s(ts_v, p_b_v)
print(f"\nRPE 1s  baseline RMSE = {float(np.sqrt(np.mean(rpe_b**2))):.4f} m/s")
print(f"RPE 1s  ZUPT     RMSE = {float(np.sqrt(np.mean(rpe_z**2))):.4f} m/s")
print(f"Claimed baseline RPE = 87.57 m/s, ZUPT RPE = 77.29 m/s")

# ============================================================
# PART D: Per-timestamp error profile at 9 checkpoints
# ============================================================
sep("PART D: Per-timestamp error profile at 9 checkpoints")

def nearest_imu_idx(t_target_s):
    return int(np.argmin(np.abs(ts_ns - int(t_target_s * 1e9))))

def gt_interp_at(t_ns):
    idx_lo = int(np.searchsorted(gt_ts, t_ns, side='right')) - 1
    if idx_lo < 0: idx_lo = 0
    if idx_lo >= len(gt_ts)-1: idx_lo = len(gt_ts)-2
    alpha = float(t_ns - gt_ts[idx_lo]) / max(float(gt_ts[idx_lo+1] - gt_ts[idx_lo]), 1)
    alpha = np.clip(alpha, 0, 1)
    return (1-alpha)*gt_pos[idx_lo] + alpha*gt_pos[idx_lo+1]

checkpoints_s = [
    ("t=0.905s (first IMU)",  ts_ns[0] * 1e-9),
    ("t≈10s",                 10.0),
    ("t≈20s",                 20.0 + ts_ns[0]*1e-9),
    ("t≈25.9s (last pre-ZUPT)", ts_ns[5179]*1e-9),
    ("t≈29.9s (last ZUPT)",   ts_ns[5979]*1e-9),
    ("t≈30.0s (first post-ZUPT)", ts_ns[5980]*1e-9),
    ("t≈60s",                 60.0 + ts_ns[0]*1e-9),
    ("t≈90s",                 90.0 + ts_ns[0]*1e-9),
    ("last valid GT overlap",  gt_ts[-1] * 1e-9),
]

print(f"{'Label':35s} {'idx':>6} {'|pos_base|':>12} {'|pos_zupt|':>12} "
      f"{'err_base':>10} {'err_zupt':>10} {'diff_err':>10}")
print("-" * 100)

for label, t_target_s in checkpoints_s:
    t_target_ns = int(t_target_s * 1e9)
    idx = int(np.argmin(np.abs(ts_ns - t_target_ns)))
    t_actual_s = ts_ns[idx] * 1e-9

    # Align positions using the global Umeyama transforms computed above
    p_b_aligned_i = R_b @ pos_b[idx] + t_b
    p_z_aligned_i = R_z @ pos_z[idx] + t_z
    p_gt_i = gt_interp_at(ts_ns[idx])

    err_b_i = np.linalg.norm(p_b_aligned_i - p_gt_i)
    err_z_i = np.linalg.norm(p_z_aligned_i - p_gt_i)
    diff_i  = err_b_i - err_z_i

    print(f"{label:35s} {idx:6d} "
          f"{np.linalg.norm(pos_b[idx]):12.3f} {np.linalg.norm(pos_z[idx]):12.3f} "
          f"{err_b_i:10.3f} {err_z_i:10.3f} {diff_i:+10.3f}")

# ============================================================
# PART E summary will be in the report artifact
# ============================================================
sep("KEY SCALAR SUMMARY (for Part E report)")

print(f"N (IMU samples):                   {N}")
print(f"N_valid (GT overlap):              {n_valid}")
print(f"ZUPT interval:                     [{ZUPT_S}, {ZUPT_E}]  ({zupt_mask.sum()} samples)")
print(f"vel[5179] norm (actual):           {np.linalg.norm(vel_z[5179]):.4f} m/s")
print(f"GT vel norm at t=25.9s:            {np.linalg.norm(gt_v5179):.4f} m/s")
print(f"Velocity error at 5179:            {np.linalg.norm(err_vel):.4f} m/s")
print(f"Pre-ZUPT identical (pos max diff): {np.max(np.abs(pos_z[:5180]-pos_b[:5180])):.2e} m")
print(f"vel[5180] norm (ZUPT zeroed):      {np.linalg.norm(vel_z[5180]):.2e} m/s")
print(f"pos[5180] == pos[5179]:            {np.allclose(pos_z[5180], pos_z[5179], atol=1e-14)}")
print(f"Baseline SE(3) ATE RMSE:           {ate_b:.4f} m")
print(f"ZUPT     SE(3) ATE RMSE:           {ate_z:.4f} m")
print(f"ATE improvement:                   {improvement_pct:.2f}%")
print(f"RPE 1s baseline:                   {float(np.sqrt(np.mean(rpe_b**2))):.4f} m/s")
print(f"RPE 1s ZUPT:                       {float(np.sqrt(np.mean(rpe_z**2))):.4f} m/s")
print(f"det(R_b)={det_Rb:.6f}  det(R_z)={det_Rz:.6f}  (expected +1, no scale)")
print(f"\nAll Parts A-D complete.")
