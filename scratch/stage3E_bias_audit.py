"""
Stage 3E — Accelerometer Bias Observability Audit (READ-ONLY)
MH_02_easy: GT bias analysis, IMU stationarity, observability theory, IMU-only estimation.
No files modified. All writes to stdout only.
"""
import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from data_loader import load_imu_data, load_groundtruth_data, IMU_TIMESTAMP_COL
from preprocessing import GYRO_COLS, ACCEL_COLS
from bias_estimation import detect_quasi_stationary_intervals, estimate_gyro_bias
from trajectory import compute_gravity_alignment_quaternion, STANDARD_GRAVITY
from degradation import sanitize_imu_measurements

IMU_PATH = "data/raw/euroc/machine_hall/MH_02_easy/mav0/imu0/data.csv"
GT_PATH  = "data/raw/euroc/machine_hall/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"
MICRO_GAP_THRESHOLD_S = 0.025

def sep(title, width=72):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)

# ================================================================
# LOAD DATA
# ================================================================
sep("LOADING DATA")
df_raw = load_imu_data(IMU_PATH)
df_sanitized, _ = sanitize_imu_measurements(df_raw.copy())
N = len(df_sanitized)
print(f"IMU samples: {N}")

gt_df = load_groundtruth_data(GT_PATH)
gt_ts  = gt_df["timestamp"].to_numpy(np.int64)
gt_px  = gt_df["p_RS_R_x"].to_numpy(np.float64)
gt_py  = gt_df["p_RS_R_y"].to_numpy(np.float64)
gt_pz  = gt_df["p_RS_R_z"].to_numpy(np.float64)
gt_vx  = gt_df["v_RS_R_x"].to_numpy(np.float64)
gt_vy  = gt_df["v_RS_R_y"].to_numpy(np.float64)
gt_vz  = gt_df["v_RS_R_z"].to_numpy(np.float64)
gt_qw  = gt_df["q_RS_w"].to_numpy(np.float64)
gt_qx  = gt_df["q_RS_x"].to_numpy(np.float64)
gt_qy  = gt_df["q_RS_y"].to_numpy(np.float64)
gt_qz  = gt_df["q_RS_z"].to_numpy(np.float64)
gt_bax = gt_df["b_a_RS_S_x"].to_numpy(np.float64)
gt_bay = gt_df["b_a_RS_S_y"].to_numpy(np.float64)
gt_baz = gt_df["b_a_RS_S_z"].to_numpy(np.float64)
gt_bwx = gt_df["b_w_RS_S_x"].to_numpy(np.float64)
gt_bwy = gt_df["b_w_RS_S_y"].to_numpy(np.float64)
gt_bwz = gt_df["b_w_RS_S_z"].to_numpy(np.float64)
print(f"GT samples:  {len(gt_ts)}")

imu_ts = df_sanitized[IMU_TIMESTAMP_COL].to_numpy(np.int64)
imu_ax = df_sanitized[ACCEL_COLS[0]].to_numpy(np.float64)
imu_ay = df_sanitized[ACCEL_COLS[1]].to_numpy(np.float64)
imu_az = df_sanitized[ACCEL_COLS[2]].to_numpy(np.float64)
imu_a  = np.column_stack([imu_ax, imu_ay, imu_az])
imu_a_mag = np.linalg.norm(imu_a, axis=1)

# ================================================================
# 1. GT ACCEL BIAS — FULL SEQUENCE ANALYSIS
# ================================================================
sep("1. GT ACCELEROMETER BIAS — FULL SEQUENCE")

gt_ba = np.column_stack([gt_bax, gt_bay, gt_baz])
gt_ba_norm = np.linalg.norm(gt_ba, axis=1)

print(f"GT accel bias (body frame, m/s^2)  — statistics over {len(gt_ts)} samples:")
print(f"  b_a_x: mean={gt_bax.mean():+.6f}  std={gt_bax.std():.6f}  min={gt_bax.min():+.6f}  max={gt_bax.max():+.6f}")
print(f"  b_a_y: mean={gt_bay.mean():+.6f}  std={gt_bay.std():.6f}  min={gt_bay.min():+.6f}  max={gt_bay.max():+.6f}")
print(f"  b_a_z: mean={gt_baz.mean():+.6f}  std={gt_baz.std():.6f}  min={gt_baz.min():+.6f}  max={gt_baz.max():+.6f}")
print(f"  ||b_a||: mean={gt_ba_norm.mean():.6f}  std={gt_ba_norm.std():.6f}  max={gt_ba_norm.max():.6f}")

# How much does bias change over time?
ba_range = gt_ba.max(0) - gt_ba.min(0)
print(f"\nBias range over sequence:  x={ba_range[0]:.6f}  y={ba_range[1]:.6f}  z={ba_range[2]:.6f} m/s^2")

# Change from first to last
ba_first = gt_ba[0]; ba_last = gt_ba[-1]
ba_drift = ba_last - ba_first
print(f"Bias drift (last-first):   x={ba_drift[0]:+.6f}  y={ba_drift[1]:+.6f}  z={ba_drift[2]:+.6f} m/s^2")
print(f"Bias drift ||: {np.linalg.norm(ba_drift):.6f} m/s^2  (~{np.linalg.norm(ba_drift)/np.linalg.norm(ba_first)*100:.1f}% of initial magnitude)")

# Time-varying: check at 10 intervals
n_seg = 10
seg_size = len(gt_ts) // n_seg
print(f"\nBias at {n_seg} evenly-spaced segments:")
print(f"  {'Segment':>10}  {'t_start(s)':>12}  {'b_a_x':>10}  {'b_a_y':>10}  {'b_a_z':>10}  {'||b_a||':>10}")
for k in range(n_seg):
    si = k * seg_size
    ei = min(si + seg_size, len(gt_ts))
    t_s = gt_ts[si] * 1e-9
    bx = gt_bax[si:ei].mean(); by = gt_bay[si:ei].mean(); bz = gt_baz[si:ei].mean()
    bn = np.sqrt(bx**2+by**2+bz**2)
    print(f"  {k:>10}  {t_s:>12.3f}  {bx:>10.6f}  {by:>10.6f}  {bz:>10.6f}  {bn:>10.6f}")

# EuRoC convention: accelerometer measurement model
# f_S = R_SW^T (a_W - g_W) + b_a + noise
# where g_W = [0, 0, -9.80665] (Z-up), a_W = vehicle acceleration in world frame
# At rest: a_W = 0 → f_S = -R_SW^T * g_W + b_a = R_WS^T * [0,0,9.80665] + b_a
print(f"\nEuRoC measurement model: f_S = R_WS^T (a_W - g_W) + b_a + noise")
print(f"  g_W = [0, 0, -9.80665] m/s^2 (Z-up world frame)")
print(f"  At rest: a_W = 0 → f_S = -R_WS^T*g_W + b_a = [gravity component in body] + b_a")
print(f"  Sign convention: b_a is ADDITIVE to the specific force measurement.")

# ================================================================
# 2. GT BIAS vs RAW IMU DURING STATIONARY INTERVAL
# ================================================================
sep("2. GT BIAS vs RAW IMU — STATIONARY INTERVAL [5180, 5979]")

# Stationary interval confirmed by Stage 3B/3C
STAT_S_IDX = 5180
STAT_E_IDX = 5979
stat_ts_start = imu_ts[STAT_S_IDX]
stat_ts_end   = imu_ts[STAT_E_IDX]
print(f"Stationary IMU indices: [{STAT_S_IDX}, {STAT_E_IDX}]")
print(f"Time range: {stat_ts_start*1e-9:.4f} → {stat_ts_end*1e-9:.4f} s  ({(stat_ts_end-stat_ts_start)*1e-9:.3f} s)")

# IMU measurements during stationary interval
a_stat = imu_a[STAT_S_IDX:STAT_E_IDX+1]
a_stat_mean = a_stat.mean(0)
a_stat_std  = a_stat.std(0)
a_stat_mag  = np.linalg.norm(a_stat, axis=1)
print(f"\nRaw IMU accel during stationary interval:")
print(f"  mean: [{a_stat_mean[0]:+.6f}, {a_stat_mean[1]:+.6f}, {a_stat_mean[2]:+.6f}] m/s^2")
print(f"  std:  [{a_stat_std[0]:.6f}, {a_stat_std[1]:.6f}, {a_stat_std[2]:.6f}] m/s^2")
print(f"  ||mean||: {np.linalg.norm(a_stat_mean):.6f} m/s^2  (nominal g = {STANDARD_GRAVITY:.5f})")

# GT bias at the stationary interval (GT timestamps → nearest)
gt_stat_mask = (gt_ts >= stat_ts_start) & (gt_ts <= stat_ts_end)
n_gt_stat = gt_stat_mask.sum()
print(f"\nGT samples inside stationary interval: {n_gt_stat}")

gt_ba_stat = gt_ba[gt_stat_mask]
gt_ba_stat_mean = gt_ba_stat.mean(0)
print(f"GT b_a during stationary interval (mean):")
print(f"  b_a_x = {gt_ba_stat_mean[0]:+.8f} m/s^2")
print(f"  b_a_y = {gt_ba_stat_mean[1]:+.8f} m/s^2")
print(f"  b_a_z = {gt_ba_stat_mean[2]:+.8f} m/s^2")
print(f"  ||b_a|| = {np.linalg.norm(gt_ba_stat_mean):.8f} m/s^2")

# GT orientation at stationary interval → rotate bias to world frame
gt_q_stat_mean_xyzw = np.array([
    gt_qx[gt_stat_mask].mean(),
    gt_qy[gt_stat_mask].mean(),
    gt_qz[gt_stat_mask].mean(),
    gt_qw[gt_stat_mask].mean(),
])
gt_q_stat_mean_xyzw /= np.linalg.norm(gt_q_stat_mean_xyzw)
R_WS = Rotation.from_quat(gt_q_stat_mean_xyzw).as_matrix()  # body→world
print(f"\nGT mean orientation R_WS (body-to-world) at stationary interval:")
print(f"  Roll/pitch/yaw: {Rotation.from_quat(gt_q_stat_mean_xyzw).as_euler('xyz', degrees=True)} deg")

# Project bias onto world Z (gravity direction)
g_world = np.array([0.0, 0.0, -STANDARD_GRAVITY])
g_body  = R_WS.T @ g_world   # gravity in body frame at stationary pose
g_body_unit = g_body / np.linalg.norm(g_body)
ba_parallel = np.dot(gt_ba_stat_mean, g_body_unit) * g_body_unit  # component along gravity
ba_perp     = gt_ba_stat_mean - ba_parallel
print(f"\nGT bias decomposition in BODY frame at stationary interval:")
print(f"  Gravity direction in body frame (unit): {g_body_unit}")
print(f"  Bias component PARALLEL to gravity:  {ba_parallel}  ||={np.linalg.norm(ba_parallel):.6f} m/s^2")
print(f"  Bias component PERPENDICULAR to g:   {ba_perp}  ||={np.linalg.norm(ba_perp):.6f} m/s^2")
print(f"  Fraction parallel: {np.linalg.norm(ba_parallel)/np.linalg.norm(gt_ba_stat_mean)*100:.1f}%")
print(f"  Fraction perp:     {np.linalg.norm(ba_perp)/np.linalg.norm(gt_ba_stat_mean)*100:.1f}%")

# What would IMU-only estimation using norm deviation give?
norm_dev = np.linalg.norm(a_stat_mean) - STANDARD_GRAVITY
# This is the scalar bias along the measured specific-force direction
imu_a_stat_unit = a_stat_mean / np.linalg.norm(a_stat_mean)
ba_imu_1d = norm_dev * imu_a_stat_unit
print(f"\nIMU-ONLY 1D bias estimate (norm deviation method):")
print(f"  ||a_stat_mean|| - g = {np.linalg.norm(a_stat_mean):.6f} - {STANDARD_GRAVITY:.5f} = {norm_dev:+.6f} m/s^2")
print(f"  Projected bias vector: {ba_imu_1d}  ||={np.linalg.norm(ba_imu_1d):.6f} m/s^2")
print(f"  Error vs GT:          {ba_imu_1d - gt_ba_stat_mean}  ||={np.linalg.norm(ba_imu_1d - gt_ba_stat_mean):.6f} m/s^2")
print(f"  Recovered fraction:   {np.dot(ba_imu_1d, gt_ba_stat_mean)/np.linalg.norm(gt_ba_stat_mean)**2*100:.1f}% of GT bias norm")

# ================================================================
# 3. GRAVITY ALIGNMENT QUATERNION vs GT ORIENTATION
# ================================================================
sep("3. GRAVITY ALIGNMENT ERROR vs GT ORIENTATION")

initial_accel = df_sanitized.loc[df_sanitized["valid_accel"], ACCEL_COLS].iloc[:100].mean().to_numpy()
q0_wxyz = compute_gravity_alignment_quaternion(initial_accel, yaw_rad=0.0)
print(f"Pipeline initial q0 [w,x,y,z]: {q0_wxyz}")

# GT orientation at first IMU timestamp
nearest_gt_first = int(np.argmin(np.abs(gt_ts - imu_ts[0])))
gt_q0_xyzw = np.array([gt_qx[nearest_gt_first], gt_qy[nearest_gt_first],
                        gt_qz[nearest_gt_first], gt_qw[nearest_gt_first]])
gt_R0 = Rotation.from_quat(gt_q0_xyzw)
gt_euler0 = gt_R0.as_euler('xyz', degrees=True)
print(f"GT orientation at t=0 [w,x,y,z]: {gt_qw[nearest_gt_first]:.6f}, {gt_qx[nearest_gt_first]:.6f}, {gt_qy[nearest_gt_first]:.6f}, {gt_qz[nearest_gt_first]:.6f}")
print(f"GT Euler [roll, pitch, yaw] deg: {gt_euler0}")

est_q0_xyzw = np.array([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])
est_R0 = Rotation.from_quat(est_q0_xyzw)
est_euler0 = est_R0.as_euler('xyz', degrees=True)
print(f"Est Euler [roll, pitch, yaw] deg: {est_euler0}")

# Angular difference
dR = gt_R0.inv() * est_R0
dangle = np.degrees(np.linalg.norm(dR.as_rotvec()))
print(f"\nOrientation error at t=0: {dangle:.4f} degrees")
print(f"  (Yaw is unobservable from gravity alone — estimator defaults to 0 deg)")
print(f"  GT yaw at t=0: {gt_euler0[2]:.4f} deg  — estimator yaw error = {est_euler0[2] - gt_euler0[2]:.4f} deg")

# Consequence of yaw error on gravity compensation
# If yaw is wrong by delta_yaw, the rotated gravity direction in body frame is wrong,
# introducing a horizontal acceleration bias proportional to g*sin(delta_yaw)
delta_yaw_rad = np.radians(est_euler0[2] - gt_euler0[2])
h_bias_from_yaw = STANDARD_GRAVITY * np.abs(np.sin(delta_yaw_rad))
print(f"\nHorizontal acceleration error from yaw misalignment: {h_bias_from_yaw:.4f} m/s^2")
print(f"  g * sin(delta_yaw={np.degrees(delta_yaw_rad):.2f} deg) = {h_bias_from_yaw:.4f} m/s^2")

# Roll/pitch error from imperfect gravity alignment
# IMU mean accel over first 100 samples vs GT mean
mean_a_initial = initial_accel
print(f"\nInitial 100-sample mean accel [m/s^2]: {mean_a_initial}")
print(f"||mean_a_initial|| = {np.linalg.norm(mean_a_initial):.6f} m/s^2  (vs g = {STANDARD_GRAVITY:.5f})")

# Specific force in world frame using q0
R_q0 = Rotation.from_quat(est_q0_xyzw).as_matrix()
f_world = R_q0 @ mean_a_initial
print(f"f_world via q0 (should be [0,0,g]):  {f_world}")
print(f"World Z residual: {f_world[2] - STANDARD_GRAVITY:+.6f} m/s^2  (bias-induced)")
print(f"World XY residual: ||[{f_world[0]:.6f}, {f_world[1]:.6f}]|| = {np.sqrt(f_world[0]**2+f_world[1]**2):.6f} m/s^2")

# ================================================================
# 4. DYNAMIC FLIGHT: IMU accel vs GT bias over full sequence
# ================================================================
sep("4. DYNAMIC FLIGHT — IMU vs GT MEASUREMENTS")

# GT ground-truth specific force: f_S = R_WS^T (a_W - g_W) + b_a
# Reconstruct what the ideal (zero-bias) IMU would read:
# f_S_true = R_WS^T (a_W - g_W)  [no bias]
# Therefore:  imu_meas ≈ f_S_true + b_a + noise
# → imu_meas - b_a ≈ f_S_true = R_WS^T (a_W - g_W)

# Interpolate GT to IMU timestamps (subsample for speed)
stride = 10
imu_ts_sub = imu_ts[::stride]
imu_a_sub  = imu_a[::stride]

gt_bax_interp = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_bax)
gt_bay_interp = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_bay)
gt_baz_interp = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_baz)
gt_qw_interp  = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_qw)
gt_qx_interp  = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_qx)
gt_qy_interp  = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_qy)
gt_qz_interp  = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_qz)
gt_vx_interp  = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_vx)
gt_vy_interp  = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_vy)
gt_vz_interp  = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_vz)
gt_ba_interp  = np.column_stack([gt_bax_interp, gt_bay_interp, gt_baz_interp])

# Bias-corrected specific force residual vs expected
residual_body = imu_a_sub - gt_ba_interp  # should equal f_S_true (gravity+dynamics in body)
residual_body_norm = np.linalg.norm(residual_body, axis=1)

# Without bias correction
raw_norm_sub = np.linalg.norm(imu_a_sub, axis=1)
print(f"IMU accel magnitude (subsampled, stride={stride}):")
print(f"  Raw ||a|| mean={raw_norm_sub.mean():.4f} std={raw_norm_sub.std():.4f} min={raw_norm_sub.min():.4f} max={raw_norm_sub.max():.4f}")
print(f"  After GT-bias-correction: mean={residual_body_norm.mean():.4f} std={residual_body_norm.std():.4f}")
print(f"  Nominal g = {STANDARD_GRAVITY:.5f} m/s^2")

# Velocity magnitude → indicates dynamic motion
vel_sub = np.sqrt(gt_vx_interp**2 + gt_vy_interp**2 + gt_vz_interp**2)
print(f"\nGT speed during sequence:")
print(f"  min={vel_sub.min():.4f}  max={vel_sub.max():.4f}  mean={vel_sub.mean():.4f} m/s")

# Classify quasi-static vs dynamic segments
STATIC_VEL_THRESH = 0.05   # m/s
STATIC_ACCEL_THRESH = 0.3  # m/s^2 (dynamic accel mag - g)
is_static_vel   = vel_sub < STATIC_VEL_THRESH
is_static_accel = np.abs(raw_norm_sub - STANDARD_GRAVITY) < STATIC_ACCEL_THRESH

print(f"\nSegment classification (stride={stride}):")
print(f"  Quasi-static (speed<{STATIC_VEL_THRESH} m/s):                {is_static_vel.sum()} samples ({is_static_vel.mean()*100:.1f}%)")
print(f"  Low dynamic accel (||a||-g < {STATIC_ACCEL_THRESH} m/s^2):   {is_static_accel.sum()} samples ({is_static_accel.mean()*100:.1f}%)")
both_static = is_static_vel & is_static_accel
print(f"  Both conditions (suitable for bias estimation):  {both_static.sum()} samples ({both_static.mean()*100:.1f}%)")

# Bias during dynamic vs static (can bias be estimated during flight?)
print(f"\nGT bias during static segments:")
ba_static = gt_ba_interp[both_static]
if len(ba_static) > 0:
    print(f"  b_a mean: {ba_static.mean(0)}")
    print(f"  b_a std:  {ba_static.std(0)}")
    
print(f"\nGT bias during dynamic segments:")
ba_dynamic = gt_ba_interp[~both_static]
if len(ba_dynamic) > 0:
    print(f"  b_a mean: {ba_dynamic.mean(0)}")
    print(f"  b_a std:  {ba_dynamic.std(0)}")

# ================================================================
# 5. OBSERVABILITY ANALYSIS
# ================================================================
sep("5. OBSERVABILITY ANALYSIS (mathematical)")

print("""
IMU measurement model (EuRoC strapdown):
  f_S[k] = R_WS[k]^T * (a_W[k] - g_W) + b_a + eta_a

where:
  f_S[k]  = measured specific force in body frame [3 x 1]
  R_WS[k] = body-to-world rotation (3D orientation) [3 x 3]
  a_W[k]  = vehicle acceleration in world frame [3 x 1]
  g_W     = [0, 0, -g] in Z-up world frame = [0, 0, -9.80665] m/s^2
  b_a     = accelerometer bias (body frame, quasi-static) [3 x 1]
  eta_a   = zero-mean noise

Rearranging:
  b_a = f_S[k] - R_WS[k]^T * (a_W[k] - g_W)
  b_a = f_S[k] - R_WS[k]^T * a_W[k] + R_WS[k]^T * g_W

OBSERVABILITY CASE 1: Stationary (a_W = 0, known R_WS)
  b_a = f_S - R_WS^T * g_W
  → 3 equations, 3 unknowns → FULLY OBSERVABLE in principle.
  BUT: R_WS must be known precisely. If R_WS is unknown (as in pure IMU-DR),
  then orientation error and bias are ALIASED.

OBSERVABILITY CASE 2: Stationary with UNKNOWN R_WS (IMU-only)
  The specific force f_S encodes both the gravity direction (→ gives roll, pitch)
  and the bias residual from the true specific force magnitude.
  → Only 1D bias (along gravity direction) is observable from ||f_S|| - g.
  → Roll/pitch errors and horizontal bias are ALIASED.
  → Yaw is completely unobservable (gravity is yaw-invariant).

OBSERVABILITY CASE 3: Dynamic flight (a_W ≠ 0)
  b_a = f_S - R_WS^T * (a_W - g_W)
  If a_W is not independently known, we have TWO unknowns (b_a, a_W) for 3 equations.
  → Bias and vehicle acceleration are ALIASED unless external position/velocity used.
""")

# Numerical check: how well can we recover b_a from stationary IMU alone?
print("Numerical observability check at stationary interval:")
print(f"  IMU-only observable: scalar norm deviation = {norm_dev:+.6f} m/s^2")
print(f"  GT 3-axis bias norm  = {np.linalg.norm(gt_ba_stat_mean):.6f} m/s^2")
print(f"  IMU-only recovered   = {np.linalg.norm(ba_imu_1d):.6f} m/s^2  ({np.linalg.norm(ba_imu_1d)/np.linalg.norm(gt_ba_stat_mean)*100:.1f}% of norm)")
print()
print("  Unobservable component (perpendicular to gravity):")
print(f"    ||b_a_perp|| = {np.linalg.norm(ba_perp):.6f} m/s^2")
print(f"    b_a_perp     = {ba_perp}")
print()

# What COULD be recovered if orientation were known exactly?
# Then b_a = f_S - R_WS^T * g_W  (all 3 axes observable)
g_body_true = -R_WS.T @ np.array([0.0, 0.0, STANDARD_GRAVITY])
# Note: g_W = [0,0,-g] → R_WS^T * g_W = [-R_WS^T*[0,0,g]]
ba_recovered_3d = a_stat_mean - g_body_true
print(f"  With KNOWN orientation (R_WS from GT):")
print(f"    g in body frame: {g_body_true}")
print(f"    b_a recovered 3-axis: {ba_recovered_3d}")
print(f"    GT b_a (reference):   {gt_ba_stat_mean}")
print(f"    Recovery error: {ba_recovered_3d - gt_ba_stat_mean}  ||={np.linalg.norm(ba_recovered_3d - gt_ba_stat_mean):.6f} m/s^2")
print(f"    Explanation: residual = IMU noise std ≈ {np.linalg.norm(a_stat_std):.4f} m/s^2")

# ================================================================
# 6. WHAT IS AVAILABLE TO THE PRODUCTION ESTIMATOR?
# ================================================================
sep("6. PRODUCTION ESTIMATOR — AVAILABLE INFORMATION")

print("""
Available WITHOUT ground truth (IMU-only, production pipeline):
  a) f_S[k]  : raw 3-axis accelerometer measurement  (all 3 axes)
  b) w_S[k]  : raw 3-axis gyroscope measurement      (all 3 axes)
  c) ts[k]   : timestamps                             (all samples)
  d) STATIONARY INTERVAL: detected via bias_estimation.detect_quasi_stationary_intervals()
     → Window [5180, 5979] confirmed detected
  e) GYRO BIAS: estimated from stationary interval (current pipeline does this ✓)
  f) GRAVITY DIRECTION: from mean accel during stationary interval
     → Gives roll + pitch (not yaw)
  g) SPECIFIC FORCE MAGNITUDE: ||f_S|| during stationary interval
     → Gives scalar bias component along gravity direction only

NOT available without ground truth:
  h) True vehicle acceleration a_W[k]  (aliased with bias during flight)
  i) True orientation R_WS[k]          (accumulates error from gyro integration)
  j) Yaw angle psi                     (completely unobservable from IMU alone)
  k) Perpendicular (horizontal) bias components (aliased with tilt/yaw error)
""")

# Quantify what 1D bias correction would do
print("1D bias correction effect estimate:")
# Drift from uncompensated parallel bias component
ba_par_norm = np.linalg.norm(ba_parallel)
T = (imu_ts[-1] - imu_ts[0]) * 1e-9
drift_parallel = 0.5 * ba_par_norm * T**2
print(f"  ||b_a_parallel|| = {ba_par_norm:.6f} m/s^2")
print(f"  Sequence duration T = {T:.2f} s")
print(f"  Quadratic drift from PARALLEL bias: 0.5 * {ba_par_norm:.6f} * {T:.2f}^2 = {drift_parallel:.2f} m")

# Drift from uncompensated perpendicular bias
ba_perp_norm = np.linalg.norm(ba_perp)
drift_perp = 0.5 * ba_perp_norm * T**2
print(f"  ||b_a_perp|| = {ba_perp_norm:.6f} m/s^2")
print(f"  Quadratic drift from PERP bias: 0.5 * {ba_perp_norm:.6f} * {T:.2f}^2 = {drift_perp:.2f} m")
print(f"  Total drift from both bias components: {drift_parallel+drift_perp:.2f} m  (over {T:.1f} s)")
print(f"  Note: actual ATE baseline = 2948.42 m (includes orientation error compounding)")

# ================================================================
# 7. IMU-ONLY BIAS ESTIMATION STRATEGY
# ================================================================
sep("7. PROPOSED IMU-ONLY BIAS ESTIMATION STRATEGY")

print("""
=== PROPOSED STRATEGY: Stationary-interval-based 1D bias correction ===

Assumptions:
  A1. At least one quasi-stationary interval is detectable (confirmed for MH_02).
  A2. During stationarity: a_W ≈ 0, v_W ≈ 0.
  A3. Accelerometer noise is zero-mean Gaussian (EuRoC spec: ~2 mg/sqrt(Hz)).
  A4. Gravity magnitude is known: g = 9.80665 m/s^2 (FAQ STANDARD_GRAVITY).
  A5. Bias is quasi-static (slowly varying): valid across the full 152-s sequence.

Observable (1D bias component along gravity):
  b_a_parallel = (||f_S_mean|| - g) * û_S
  where û_S = f_S_mean / ||f_S_mean||  (measured specific force direction in body)

  Current norm deviation: {:.6f} m/s^2  → bias component along gravity direction

Unobservable without additional pose information:
  b_a_perp = b_a - b_a_parallel  (horizontal, {:.6f} m/s^2)
  Yaw-dependent horizontal components of bias require multi-pose excitation.

Correction procedure for production estimator (IMU-only, NO GT):
  1. Detect stationary interval using existing detect_quasi_stationary_intervals().
  2. Compute mean accel f_S_mean from that interval.
  3. Compute: b_a_1d = (||f_S_mean|| - STANDARD_GRAVITY) * (f_S_mean / ||f_S_mean||)
  4. Subtract b_a_1d from every accelerometer sample before integration.
  5. Keep gravity magnitude constant at STANDARD_GRAVITY (not measured ||f_S||).

Implementation note (exists in trajectory.py):
  → estimate_accel_bias_gravity_aligned() already implements exactly this.
  → It is NOT currently called in the production main.py pipeline.
  → Adding this single call requires explicit authorization.
""".format(norm_dev, np.linalg.norm(ba_perp)))

# ================================================================
# 8. ACCURACY BENEFIT ESTIMATE
# ================================================================
sep("8. ESTIMATED ACCURACY BENEFIT")

# 1D bias correction removes the gravity-parallel component
# The residual unobservable perp component causes drift:
ba_full = gt_ba_stat_mean

# Scenario A: NO bias correction (current)
drift_A = 0.5 * np.linalg.norm(ba_full) * T**2
print(f"Scenario A (current, no bias correction):")
print(f"  Total bias norm:        {np.linalg.norm(ba_full):.6f} m/s^2")
print(f"  Estimated position drift: ~{drift_A:.1f} m over {T:.1f} s")
print(f"  Actual SE(3) ATE RMSE:   2948.42 m")

# Scenario B: 1D bias correction applied
# After correction: residual bias = b_a_perp (unobservable component only)
ba_residual_B = ba_perp
drift_B = 0.5 * np.linalg.norm(ba_residual_B) * T**2
print(f"\nScenario B (1D bias correction from stationary interval):")
print(f"  Residual (perp) bias:     {np.linalg.norm(ba_residual_B):.6f} m/s^2")
print(f"  Estimated residual drift: ~{drift_B:.1f} m over {T:.1f} s")
print(f"  Expected ATE reduction:   ~{(drift_A-drift_B)/drift_A*100:.0f}% (order-of-magnitude estimate)")
print(f"  Note: does NOT account for orientation error compounding — actual improvement may differ")

# Scenario C: FULL 3-axis bias correction (requires known orientation = GT)
print(f"\nScenario C (full 3-axis correction, requires known GT orientation):")
print(f"  Residual bias:   ~IMU noise floor ({np.linalg.norm(a_stat_std):.4f} m/s^2)")
print(f"  Estimated drift: ~{0.5*np.linalg.norm(a_stat_std)*T**2:.1f} m over {T:.1f} s")
print(f"  NOT achievable with IMU-only estimator (yaw + horizontal bias unobservable)")

print(f"\nKey constraint:")
print(f"  Yaw error ({est_euler0[2]-gt_euler0[2]:.2f} deg) causes horizontal gravity-cancel error:")
print(f"  ~{h_bias_from_yaw:.4f} m/s^2 horizontal acceleration bias (indistinguishable from true accel bias)")
print(f"  → Limits practical benefit of any IMU-only horizontal bias correction")

print("\n--- Stage 3E Analysis Complete ---")
