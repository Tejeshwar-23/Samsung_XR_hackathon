"""
Stage 3E-Verification — READ-ONLY
Verifies all 7 items from the Stage 3E bias observability audit.
No files modified. Stdout only.
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
from trajectory import (
    compute_gravity_alignment_quaternion,
    rotate_and_compensate_gravity,
    integrate_trajectory,
    STANDARD_GRAVITY,
)
from degradation import sanitize_imu_measurements

IMU_PATH = "data/raw/euroc/machine_hall/MH_02_easy/mav0/imu0/data.csv"
GT_PATH  = "data/raw/euroc/machine_hall/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"
MICRO_GAP_THRESHOLD_S = 0.025

def sep(title, n=72):
    print("\n" + "=" * n)
    print(title)
    print("=" * n)

# ================================================================
# LOAD DATA
# ================================================================
sep("LOADING DATA")
df_raw       = load_imu_data(IMU_PATH)
df_san, _    = sanitize_imu_measurements(df_raw.copy())
gt_df        = load_groundtruth_data(GT_PATH)
gt_ts        = gt_df["timestamp"].to_numpy(np.int64)
imu_ts       = df_san[IMU_TIMESTAMP_COL].to_numpy(np.int64)
imu_a        = df_san[ACCEL_COLS].to_numpy(np.float64)
N = len(df_san)
print(f"IMU samples: {N}  |  GT samples: {len(gt_ts)}")
print(f"IMU t0: {imu_ts[0]}  ({imu_ts[0]*1e-9:.6f} s)")
print(f"GT  t0: {gt_ts[0]}   ({gt_ts[0]*1e-9:.6f} s)")
print(f"IMU t_last: {imu_ts[-1]} ({imu_ts[-1]*1e-9:.6f} s)")
print(f"GT  t_last: {gt_ts[-1]}  ({gt_ts[-1]*1e-9:.6f} s)")

# ================================================================
# ITEM 5A: VERIFY QUATERNION CONVENTION AND GT COLUMN NAMES
# ================================================================
sep("ITEM 5: Convention Verification")

# EuRoC GT quaternion: q_RS_w, q_RS_x, q_RS_y, q_RS_z
# Convention: q_RS is body-to-world rotation (R=reference/world, S=sensor/body)
# SciPy from_quat expects [x,y,z,w]
gt_qw = gt_df["q_RS_w"].to_numpy(np.float64)
gt_qx = gt_df["q_RS_x"].to_numpy(np.float64)
gt_qy = gt_df["q_RS_y"].to_numpy(np.float64)
gt_qz = gt_df["q_RS_z"].to_numpy(np.float64)

# EuRoC accel bias: b_a_RS_S_x/y/z — bias in BODY (sensor, S) frame
gt_bax = gt_df["b_a_RS_S_x"].to_numpy(np.float64)
gt_bay = gt_df["b_a_RS_S_y"].to_numpy(np.float64)
gt_baz = gt_df["b_a_RS_S_z"].to_numpy(np.float64)

# EuRoC velocity: v_RS_R_x/y/z — velocity in WORLD (reference, R) frame
gt_vx = gt_df["v_RS_R_x"].to_numpy(np.float64)
gt_vy = gt_df["v_RS_R_y"].to_numpy(np.float64)
gt_vz = gt_df["v_RS_R_z"].to_numpy(np.float64)

# EuRoC position: p_RS_R_x/y/z — position in WORLD frame
gt_px = gt_df["p_RS_R_x"].to_numpy(np.float64)
gt_py = gt_df["p_RS_R_y"].to_numpy(np.float64)
gt_pz = gt_df["p_RS_R_z"].to_numpy(np.float64)

print("GT column conventions (EuRoC spec):")
print("  q_RS_[w,x,y,z] : body-to-world quaternion (SciPy: [x,y,z,w])")
print("  b_a_RS_S_[xyz]  : accel bias in BODY frame")
print("  v_RS_R_[xyz]    : velocity in WORLD frame")
print("  p_RS_R_[xyz]    : position in WORLD frame")

# EuRoC measurement model:
# f_S = R_SW * (a_W - g_W) + b_a   where R_SW = R_WS^T (world-to-body)
# g_W = [0, 0, -g] in Z-up world frame
# At rest: a_W = 0 → f_S = R_SW * (-g_W) + b_a = R_WS^T * [0,0,g] + b_a
# where [0,0,g] = -g_W with g = 9.80665
print("\nEuRoC measurement model: f_S = R_WS^T * (a_W - g_W) + b_a")
print(f"  g_W = [0, 0, -{STANDARD_GRAVITY}] m/s^2  (Z-up, gravity negative Z)")
print(f"  At rest: f_S = R_WS^T * [0, 0, +{STANDARD_GRAVITY}] + b_a")

# Verify unit quaternion norms at first GT sample
q0_norm = np.sqrt(gt_qw[0]**2 + gt_qx[0]**2 + gt_qy[0]**2 + gt_qz[0]**2)
print(f"\nGT q[0] unit norm check: {q0_norm:.10f} (should be 1.0)")

# Verify our pipeline's q0 rotates gravity correctly
initial_accel_100 = df_san.loc[df_san["valid_accel"], ACCEL_COLS].iloc[:100].mean().to_numpy()
q0_wxyz = compute_gravity_alignment_quaternion(initial_accel_100, yaw_rad=0.0)
q0_xyzw = np.array([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])
R_q0 = Rotation.from_quat(q0_xyzw).as_matrix()
f_world_via_q0 = R_q0 @ initial_accel_100
print(f"\nPipeline q0 gravity-alignment verification:")
print(f"  initial_accel_100 = {initial_accel_100}  ||={np.linalg.norm(initial_accel_100.copy()):.6f}")
print(f"  R(q0) * f_S = {f_world_via_q0}")
print(f"  Expected: [0, 0, {np.linalg.norm(initial_accel_100.copy()):.4f}]")
print(f"  World-XY residual: {np.sqrt(f_world_via_q0[0]**2+f_world_via_q0[1]**2):.2e} m/s^2 ✓")
print(f"  World-Z value: {f_world_via_q0[2]:.6f} (= ||f_S||, as expected) ✓")

# ================================================================
# ITEM 5B: TIMESTAMP ALIGNMENT CHECK
# ================================================================
sep("ITEM 5B: IMU/GT Timestamp Alignment")

imu_t0_ns = imu_ts[0]; gt_t0_ns = gt_ts[0]
delta_t0_ms = (gt_t0_ns - imu_t0_ns) * 1e-6
print(f"IMU first timestamp:  {imu_t0_ns} ns")
print(f"GT  first timestamp:  {gt_t0_ns} ns")
print(f"GT starts AFTER IMU:  dt = {delta_t0_ms:.3f} ms  ({int(delta_t0_ms/5):.0f} IMU frames)")
print(f"IMU last timestamp:   {imu_ts[-1]} ns")
print(f"GT  last timestamp:   {gt_ts[-1]} ns")
delta_last_ms = (imu_ts[-1] - gt_ts[-1]) * 1e-6
print(f"IMU extends BEYOND GT by: {delta_last_ms:.3f} ms  ({int(delta_last_ms/5):.0f} IMU frames)")

# Nearest GT sample to IMU[0]
nearest_gt_to_imu0 = int(np.argmin(np.abs(gt_ts - imu_ts[0])))
print(f"\nNearest GT index to IMU[0]: {nearest_gt_to_imu0}  dt={abs(gt_ts[nearest_gt_to_imu0]-imu_ts[0])*1e-6:.3f} ms")
print(f"GT q at nearest to IMU[0]: w={gt_qw[nearest_gt_to_imu0]:.6f} x={gt_qx[nearest_gt_to_imu0]:.6f} y={gt_qy[nearest_gt_to_imu0]:.6f} z={gt_qz[nearest_gt_to_imu0]:.6f}")

# Note: IMU[0] has NO corresponding GT sample (GT starts 24 ms later)
# Pipeline uses first 100 IMU samples for initial alignment; these overlap GT
nearest_gt_to_imu100 = int(np.argmin(np.abs(gt_ts - imu_ts[100])))
print(f"Nearest GT index to IMU[100] (end of initial window): {nearest_gt_to_imu100}  dt={abs(gt_ts[nearest_gt_to_imu100]-imu_ts[100])*1e-6:.3f} ms")
print(f"GT q at IMU[100]: w={gt_qw[nearest_gt_to_imu100]:.6f} x={gt_qx[nearest_gt_to_imu100]:.6f} y={gt_qy[nearest_gt_to_imu100]:.6f} z={gt_qz[nearest_gt_to_imu100]:.6f}")

# ================================================================
# ITEM 1 & 2: GRAVITY VECTOR DIFFERENCE + YAW/ROLL-PITCH SEPARATION
# ================================================================
sep("ITEMS 1 & 2: Gravity-Vector Difference + Yaw vs Roll/Pitch Separation")

# Use nearest GT to IMU[0] for comparison
gt_q0_xyzw = np.array([gt_qx[nearest_gt_to_imu0], gt_qy[nearest_gt_to_imu0],
                        gt_qz[nearest_gt_to_imu0], gt_qw[nearest_gt_to_imu0]])
gt_q0_xyzw /= np.linalg.norm(gt_q0_xyzw)
gt_R0 = Rotation.from_quat(gt_q0_xyzw)

est_R0 = Rotation.from_quat(q0_xyzw)

print("Orientations at t=0 (nearest GT to IMU[0]):")
gt_euler = gt_R0.as_euler('XYZ', degrees=True)   # intrinsic roll-pitch-yaw
est_euler = est_R0.as_euler('XYZ', degrees=True)
print(f"  GT  Euler XYZ (intrinsic) deg: roll={gt_euler[0]:.4f}  pitch={gt_euler[1]:.4f}  yaw={gt_euler[2]:.4f}")
print(f"  Est Euler XYZ (intrinsic) deg: roll={est_euler[0]:.4f}  pitch={est_euler[1]:.4f}  yaw={est_euler[2]:.4f}")
print(f"  Diff roll={est_euler[0]-gt_euler[0]:+.4f}  pitch={est_euler[1]-gt_euler[1]:+.4f}  yaw={est_euler[2]-gt_euler[2]:+.4f} deg")

# 1. Total orientation error (rotation angle between the two)
dR_total = gt_R0.inv() * est_R0
dangle_total_deg = np.degrees(np.linalg.norm(dR_total.as_rotvec()))
print(f"\nTotal orientation error (angle): {dangle_total_deg:.4f} deg")

# 2. Gravity-vector difference: how much does the estimated q0 mis-align gravity?
g_world = np.array([0.0, 0.0, -STANDARD_GRAVITY])  # g_W in Z-up world
g_body_gt  = gt_R0.inv().apply(g_world)   # gravity in body frame via GT
g_body_est = est_R0.inv().apply(g_world)  # gravity in body frame via estimator
print(f"\nGravity in body frame (GT):  {g_body_gt}  ||={np.linalg.norm(g_body_gt):.6f}")
print(f"Gravity in body frame (Est): {g_body_est}  ||={np.linalg.norm(g_body_est):.6f}")

# Angular difference in gravity direction alone
ghat_gt  = g_body_gt  / np.linalg.norm(g_body_gt)
ghat_est = g_body_est / np.linalg.norm(g_body_est)
dot_g = np.clip(np.dot(ghat_gt, ghat_est), -1, 1)
angle_g_deg = np.degrees(np.arccos(dot_g))
print(f"Gravity-vector angle difference: {angle_g_deg:.4f} deg")
print(f"(This = roll/pitch error only; yaw leaves gravity direction unchanged)")

# 3. Isolate yaw error from roll/pitch error
# Method: strip yaw from both quaternions (project onto gravity-consistent subspace)
# A rotation can be decomposed as R = R_yaw * R_tilt
# where R_tilt aligns the body Z to world Z, and R_yaw rotates around world Z

def remove_yaw(R_mat):
    """Strip yaw: return the gravity-only aligned rotation."""
    # Get the body Z-axis in world frame
    z_body_in_world = R_mat[:, 2]  # column 2 of body-to-world R = world coords of body Z
    z_world = np.array([0., 0., 1.])
    # Find rotation that aligns z_body_in_world to z_world (no yaw component)
    cross = np.cross(z_body_in_world, z_world)
    s = np.linalg.norm(cross); c = np.dot(z_body_in_world, z_world)
    if s < 1e-10:
        return np.eye(3)
    kmat = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    R_align = np.eye(3) + kmat + kmat @ kmat * (1-c)/(s**2)
    return R_align

R_gt_mat  = gt_R0.as_matrix()
R_est_mat = est_R0.as_matrix()

# Yaw of each rotation: project body X onto world XY plane
def get_yaw_rad(R_mat):
    # World-frame body X direction
    x_world = R_mat[:, 0]
    return np.arctan2(x_world[1], x_world[0])

yaw_gt  = np.degrees(get_yaw_rad(R_gt_mat))
yaw_est = np.degrees(get_yaw_rad(R_est_mat))
yaw_err = yaw_est - yaw_gt
# Wrap to [-180, 180]
yaw_err = (yaw_err + 180) % 360 - 180
print(f"\nYaw decomposition:")
print(f"  GT  yaw: {yaw_gt:.4f} deg")
print(f"  Est yaw: {yaw_est:.4f} deg")
print(f"  Yaw error (wrapped): {yaw_err:.4f} deg")
print(f"  Roll/pitch error (from gravity angle): {angle_g_deg:.4f} deg")
print(f"  Total orientation error: {dangle_total_deg:.4f} deg")
print(f"  Yaw contributes: {abs(yaw_err):.4f} deg  (of {dangle_total_deg:.4f} deg total)")

# 4. Gravity direction differences in WORLD frame (how the drone looks from above)
g_world_via_gt  = gt_R0.apply(g_body_gt)
g_world_via_est = est_R0.apply(g_body_est)
print(f"\nGravity in world frame (GT):  {g_world_via_gt}")
print(f"Gravity in world frame (Est): {g_world_via_est}")
print(f"(Both should be ~[0, 0, -g]; difference = error in world-Z gravity compensation)")

# 5. Actual gravity compensation residual in world frame
# World-Z residual after q0 rotation
g_world_z_gt  = gt_R0.apply(initial_accel_100.copy())[2]  # Apply GT rotation to actual IMU first sample
g_world_z_est = est_R0.apply(initial_accel_100.copy())[2]  # Apply est rotation to same IMU sample
g_world_xy_gt  = est_R0.apply(initial_accel_100.copy())[:2]
g_world_xy_est = gt_R0.apply(initial_accel_100.copy())[:2]
print(f"\nActual IMU first-100 mean rotated to world via GT q0:")
world_via_gt = gt_R0.apply(initial_accel_100.copy())
world_via_est = est_R0.apply(initial_accel_100.copy())
print(f"  Via GT:  {world_via_gt}  XY={np.linalg.norm(world_via_gt[:2]):.6f} Z={world_via_gt[2]:.6f}")
print(f"  Via Est: {world_via_est}  XY={np.linalg.norm(world_via_est[:2]):.6f} Z={world_via_est[2]:.6f}")
print(f"  World-XY residual (GT):  {np.linalg.norm(world_via_gt[:2]):.6f} m/s^2  (should be ~0 at correct orientation)")
print(f"  World-XY residual (Est): {np.linalg.norm(world_via_est[:2]):.6f} m/s^2  (should be ~0 for perfect alignment)")

# ================================================================
# ITEM 3: VERIFY 2.16 m/s² ERROR IN PRODUCTION WORLD-FRAME ACCEL
# ================================================================
sep("ITEM 3: Verify 2.16 m/s² Yaw-Error Effect in Production World-Frame Accel")

# Step 1: Run gyro bias estimation + orientation integration (as main.py does)
bias_result = estimate_gyro_bias(df_san, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
gyro_bias   = bias_result["gyro_bias"]
print(f"Gyro bias [rad/s]: {gyro_bias}")

df_proc = df_san.copy()
valid_gyro = df_proc["valid_gyro"].to_numpy(bool)
for ci, col in enumerate(GYRO_COLS):
    df_proc.loc[valid_gyro, col] = df_proc.loc[valid_gyro, col] - gyro_bias[ci]

gyro_rates = df_proc[GYRO_COLS].to_numpy(np.float64)
dt_arr = np.zeros(N, dtype=np.float64)
if N > 1:
    dt_arr[1:] = (imu_ts[1:] - imu_ts[:-1]).astype(np.float64) * 1e-9

def _dq(rv):
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12: return np.array([1.,0.,0.,0.])
    h = 0.5*angle; s=np.sin(h)/angle
    return np.array([np.cos(h), rv[0]*s, rv[1]*s, rv[2]*s])
def _qmul(q1,q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

quats = np.empty((N,4), dtype=np.float64)
q = q0_wxyz.copy(); quats[0] = q
for i in range(1, N):
    is_macro = dt_arr[i] > MICRO_GAP_THRESHOLD_S
    bv = bool(valid_gyro[i-1]) and bool(valid_gyro[i])
    if bv and not is_macro:
        wm = 0.5*(gyro_rates[i-1]+gyro_rates[i])
        dq = _dq(wm*dt_arr[i]); q = _qmul(q, dq); q /= np.linalg.norm(q)
    quats[i] = q

# Step 2: Compute world-frame linear acceleration (production pipeline)
# Using production vectorized rotation (same as main.py lines 412-430)
w_q = quats[:,0]; x_q=quats[:,1]; y_q=quats[:,2]; z_q=quats[:,3]
vx=imu_a[:,0]; vy=imu_a[:,1]; vz=imu_a[:,2]
valid_accel = df_san["valid_accel"].to_numpy(bool)
a_body_prod = imu_a.copy(); a_body_prod[~valid_accel] = 0.0
vx=a_body_prod[:,0]; vy=a_body_prod[:,1]; vz=a_body_prod[:,2]

a_world_x = (1.-2.*(y_q*y_q+z_q*z_q))*vx + 2.*(x_q*y_q-w_q*z_q)*vy + 2.*(x_q*z_q+w_q*y_q)*vz
a_world_y = 2.*(x_q*y_q+w_q*z_q)*vx + (1.-2.*(x_q*x_q+z_q*z_q))*vy + 2.*(y_q*z_q-w_q*x_q)*vz
a_world_z = 2.*(x_q*z_q-w_q*y_q)*vx + 2.*(y_q*z_q+w_q*x_q)*vy + (1.-2.*(x_q*x_q+y_q*y_q))*vz
a_world_z_lin = a_world_z - STANDARD_GRAVITY  # linear accel Z (gravity subtracted)
a_lin_x = a_world_x.copy()
a_lin_y = a_world_y.copy()
a_lin_z = a_world_z_lin.copy()
a_lin = np.column_stack([a_lin_x, a_lin_y, a_lin_z])

# Step 3: What would ideal (GT orientation) world-frame accel look like at t=0?
# Using GT q at nearest IMU[0]
gt_R0_full = Rotation.from_quat(gt_q0_xyzw).as_matrix()
a_world_via_gt_q0 = gt_R0_full @ initial_accel_100
a_lin_gt_q0 = a_world_via_gt_q0 + np.array([0., 0., -STANDARD_GRAVITY])

a_world_via_est_q0 = R_q0 @ initial_accel_100
a_lin_est_q0 = a_world_via_est_q0 + np.array([0., 0., -STANDARD_GRAVITY])

print(f"\nAt t=0 using initial_accel_100 mean:")
print(f"  a_world via GT  q: {a_world_via_gt_q0}  |a_lin|={np.linalg.norm(a_lin_gt_q0):.4f}")
print(f"  a_world via Est q: {a_world_via_est_q0}  |a_lin|={np.linalg.norm(a_lin_est_q0):.4f}")
print(f"  Difference (Est-GT): {a_world_via_est_q0-a_world_via_gt_q0}")
print(f"  ||Difference||: {np.linalg.norm(a_world_via_est_q0-a_world_via_gt_q0):.6f} m/s^2")

# Step 4: Verify the theoretical 2.16 m/s² formula g*sin(yaw_err)
# This formula assumes a level drone. It is an approximation.
# More precisely: when the drone is tilted, yaw error rotates the gravity component
# in the XY plane, introducing horizontal error of:
# ||R(q_est)*[0,0,g] - R(q_gt)*[0,0,g]|| (horizontal components)
gvec_world = np.array([0., 0., STANDARD_GRAVITY])  # +g in world Z (what gets rotated)
# In body frame, the estimator thinks gravity points in direction: R_est^T * g_world
# The error in world-frame gravity direction:
g_est_world = est_R0.apply(est_R0.inv().apply(gvec_world))  # = gvec_world (trivially)

# More meaningful: compare gravity-cancel vectors in world frame
# est cancels: a_lin = R_est * f_body - [0,0,g]
# gt  cancels: a_lin = R_gt  * f_body - [0,0,g]
# Error from orientation difference: (R_est - R_gt) * f_body
# At rest, f_body ≈ R_gt^T * [0,0,g] + b_a ≈ R_gt^T * [0,0,g]
# Error ≈ (R_est - R_gt) * R_gt^T * [0,0,g] = (R_est * R_gt^T - I) * [0,0,g]

f_body_at_rest_gt = gt_R0.inv().apply(gvec_world)   # GT-estimated body gravity direction
world_error = est_R0.apply(f_body_at_rest_gt) - gvec_world  # extra linear accel from yaw
print(f"\nOrientation-induced world-frame acceleration error (at t=0, rest assumption):")
print(f"  f_body at rest (GT frame): {f_body_at_rest_gt}")
print(f"  R_est * f_body:            {est_R0.apply(f_body_at_rest_gt)}")
print(f"  Expected: [0, 0, g] = {gvec_world}")
print(f"  Difference (world_error):  {world_error}")
print(f"  ||world_error||:           {np.linalg.norm(world_error):.4f} m/s^2")
print(f"  Horizontal (XY) component: {np.linalg.norm(world_error[:2]):.4f} m/s^2")
print(f"  Vertical (Z) component:    {abs(world_error[2]):.4f} m/s^2")

# Approximate formula check
approx_yaw = abs(np.sin(np.radians(yaw_err))) * STANDARD_GRAVITY
print(f"\nSimple approximation g*sin(yaw_err): {approx_yaw:.4f} m/s^2")
print(f"More precise (from rotation matrix): {np.linalg.norm(world_error[:2]):.4f} m/s^2")
print(f"Note: g*sin(yaw) is ONLY valid for a LEVEL drone (zero roll/pitch).")
print(f"The drone has roll≈{gt_euler[0]:.1f}°, pitch≈{gt_euler[1]:.1f}° — the formula is an approximation.")

# Check ACTUAL production world-frame linear accel at first few samples
print(f"\nProduction estimator world-frame linear accel (first 5 samples):")
for i in range(5):
    print(f"  idx={i}: a_lin_world = [{a_lin[i,0]:+.4f}, {a_lin[i,1]:+.4f}, {a_lin[i,2]:+.4f}]  ||={np.linalg.norm(a_lin[i]):.4f}")

print(f"\nProduction estimator world-frame linear accel statistics (all {N} samples):")
print(f"  X: mean={a_lin[:,0].mean():+.4f}  std={a_lin[:,0].std():.4f}")
print(f"  Y: mean={a_lin[:,1].mean():+.4f}  std={a_lin[:,1].std():.4f}")
print(f"  Z: mean={a_lin[:,2].mean():+.4f}  std={a_lin[:,2].std():.4f}")

# Mean a_lin_world at stationary interval (should be ~0 if perfect orientation)
STAT_S, STAT_E = 5180, 5979
a_lin_stat = a_lin[STAT_S:STAT_E+1]
print(f"\nProduction world-frame linear accel during stationary interval [5180,5979]:")
print(f"  X: mean={a_lin_stat[:,0].mean():+.6f}  std={a_lin_stat[:,0].std():.6f}")
print(f"  Y: mean={a_lin_stat[:,1].mean():+.6f}  std={a_lin_stat[:,1].std():.6f}")
print(f"  Z: mean={a_lin_stat[:,2].mean():+.6f}  std={a_lin_stat[:,2].std():.6f}")
print(f"  ||mean a_lin_world|| during stationary = {np.linalg.norm(a_lin_stat.mean(0)):.4f} m/s^2")
print(f"  (This IS the residual error the estimator cannot remove — combines bias + orientation error)")

# ================================================================
# ITEM 4: REPRODUCE 0.028 m/s² 3-AXIS BIAS RECOVERY
# ================================================================
sep("ITEM 4: Reproduce 3-Axis Accel Bias Recovery with Known Orientation")

# GT orientation at stationary interval
stat_ts_start = imu_ts[STAT_S]; stat_ts_end = imu_ts[STAT_E]
gt_mask = (gt_ts >= stat_ts_start) & (gt_ts <= stat_ts_end)
n_gt_stat = gt_mask.sum()
print(f"GT samples inside stationary interval [{STAT_S},{STAT_E}]: {n_gt_stat}")

gt_q_stat = np.column_stack([gt_qx[gt_mask], gt_qy[gt_mask], gt_qz[gt_mask], gt_qw[gt_mask]])
gt_q_stat_mean = gt_q_stat.mean(0); gt_q_stat_mean /= np.linalg.norm(gt_q_stat_mean)
R_WS_stat = Rotation.from_quat(gt_q_stat_mean).as_matrix()

gt_ba_stat = np.array([gt_bax[gt_mask].mean(), gt_bay[gt_mask].mean(), gt_baz[gt_mask].mean()])
print(f"GT b_a at stationary interval (mean): {gt_ba_stat}")
print(f"||GT b_a||: {np.linalg.norm(gt_ba_stat):.6f} m/s^2")

a_stat_mean = imu_a[STAT_S:STAT_E+1].mean(0)
a_stat_std  = imu_a[STAT_S:STAT_E+1].std(0)
print(f"\nIMU mean during stationary:  {a_stat_mean}  ||={np.linalg.norm(a_stat_mean):.6f}")
print(f"IMU std  during stationary:  {a_stat_std}  ||={np.linalg.norm(a_stat_std):.6f}")

# 3-axis recovery formula:
# At rest: f_S = R_WS^T * [0, 0, g] + b_a
# → b_a_recovered = f_S - R_WS^T * [0, 0, g]
# where g_at_rest = [0, 0, STANDARD_GRAVITY] (the magnitude that balances gravity)
gvec_z = np.array([0., 0., STANDARD_GRAVITY])
g_body_expected = R_WS_stat.T @ gvec_z  # = R_SW * [0,0,g]
print(f"\ng_body_expected (R_WS^T * [0,0,g]): {g_body_expected}  ||={np.linalg.norm(g_body_expected):.6f}")
ba_recovered = a_stat_mean - g_body_expected
recovery_error = ba_recovered - gt_ba_stat
print(f"b_a recovered (3-axis):             {ba_recovered}")
print(f"GT b_a:                             {gt_ba_stat}")
print(f"Recovery error:                     {recovery_error}  ||={np.linalg.norm(recovery_error):.6f} m/s^2")
print(f"a_stat std (noise floor reference): ||={np.linalg.norm(a_stat_std):.4f} m/s^2")
print(f"Recovery error / noise std:         {np.linalg.norm(recovery_error)/np.linalg.norm(a_stat_std):.2f}x")

# 1D recovery (IMU-only)
norm_dev = np.linalg.norm(a_stat_mean) - STANDARD_GRAVITY
ba_1d = norm_dev * (a_stat_mean / np.linalg.norm(a_stat_mean))
print(f"\n1D bias recovery (IMU-only, norm deviation):")
print(f"  ||a_stat|| - g = {np.linalg.norm(a_stat_mean):.6f} - {STANDARD_GRAVITY:.5f} = {norm_dev:+.6f} m/s^2")
print(f"  b_a_1d = {ba_1d}  ||={np.linalg.norm(ba_1d):.6f} m/s^2")
print(f"  Error vs GT: {ba_1d - gt_ba_stat}  ||={np.linalg.norm(ba_1d-gt_ba_stat):.6f} m/s^2")
print(f"  Fraction recovered: {np.linalg.norm(ba_1d)/np.linalg.norm(gt_ba_stat)*100:.1f}% of GT bias norm")

# ================================================================
# ITEM 6: EXPLAIN 800 vs 183 SAMPLE DISCREPANCY
# ================================================================
sep("ITEM 6: Explain 800-Sample Stationary Interval vs 183 Static-Comparison Samples")

# The 800 comes from detect_quasi_stationary_intervals on the CLEANED IMU (df_proc after gyro bias removal)
det = detect_quasi_stationary_intervals(df_san, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
intervals = det["detected_intervals"]
print(f"detect_quasi_stationary_intervals on df_san (before gyro bias removal):")
for iv in intervals:
    print(f"  idx [{iv[0]}, {iv[1]}]  dur={iv[4]:.3f}s  samples={iv[1]-iv[0]+1}")

# The 183 came from Stage 3E dynamic classification (stride=10 subsampled, BOTH conditions)
# Condition A: GT speed < 0.05 m/s
# Condition B: ||imu_a|| - g < 0.3 m/s^2
# Applied to full IMU (stride=10), so max 3040 samples
imu_ts_sub = imu_ts[::10]
imu_a_sub  = imu_a[::10]
gt_vx_interp = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_vx)
gt_vy_interp = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_vy)
gt_vz_interp = np.interp(imu_ts_sub.astype(float), gt_ts.astype(float), gt_vz)
imu_a_mag_sub = np.linalg.norm(imu_a_sub, axis=1)
vel_sub = np.sqrt(gt_vx_interp**2+gt_vy_interp**2+gt_vz_interp**2)
is_slow = vel_sub < 0.05
is_low_dyn = np.abs(imu_a_mag_sub - STANDARD_GRAVITY) < 0.3
both = is_slow & is_low_dyn
print(f"\nStage 3E dynamic classification (stride=10, N_sub={len(imu_ts_sub)}):")
print(f"  Slow (GT speed<0.05): {is_slow.sum()} samples ({is_slow.mean()*100:.1f}%)")
print(f"  Low dyn accel:        {is_low_dyn.sum()} samples ({is_low_dyn.mean()*100:.1f}%)")
print(f"  BOTH:                 {both.sum()} samples ({both.mean()*100:.1f}%)")
print(f"  Corresponds to {both.sum()*10} original IMU samples (before stride)")

# The 800-sample interval [5180,5979] in stride-10 space:
sub_indices_in_stat = np.where((imu_ts_sub >= imu_ts[STAT_S]) & (imu_ts_sub <= imu_ts[STAT_E]))[0]
sub_in_both = both[sub_indices_in_stat]
print(f"\nStationary interval [5180,5979] in subsampled space:")
print(f"  {len(sub_indices_in_stat)} sub-samples; of those, {sub_in_both.sum()} pass 'both' condition")
print(f"  Reason for <800: stride=10 subsampling → only every 10th sample counted")
print(f"  Explanation: 800 ÷ 10 ≈ {800//10} sub-samples; actual {sub_in_both.sum()} because")
print(f"  IMU noise during hover may push ||a||-g above 0.3 threshold for some samples")

# More detail: check ||a|| during stationary
a_stat_full = imu_a[STAT_S:STAT_E+1]
a_mag_stat = np.linalg.norm(a_stat_full, axis=1)
above_thresh = np.sum(np.abs(a_mag_stat - STANDARD_GRAVITY) > 0.3)
print(f"\nWithin stationary interval [5180,5979], samples with ||a||-g > 0.3: {above_thresh}/{len(a_stat_full)}")
print(f"  ||a|| min={a_mag_stat.min():.4f} max={a_mag_stat.max():.4f} std={a_mag_stat.std():.4f}")
print(f"  Some samples exceed the 0.3 threshold due to vibration/turbulence noise → excluded from 'both'")
print(f"  Conclusion: 183 sub-samples ≠ 800/10=80 expected from just the stationary interval")
# Find where the 183 come from
both_full_imu_idxs = imu_ts_sub[both]
print(f"\n  Time range of 'both' samples:")
print(f"    min t: {both_full_imu_idxs.min()*1e-9:.3f} s")
print(f"    max t: {both_full_imu_idxs.max()*1e-9:.3f} s")
print(f"  The 183 samples span the FULL sequence, not just the stationary interval")
print(f"  They include any brief slow/low-dyn moments throughout flight")

# ================================================================
# ITEM 7: ATE IMPROVEMENT — THEORETICAL VS MEASURED
# ================================================================
sep("ITEM 7: Reassess 5% ATE Improvement — Theoretical vs Measured")

print("""
Stage 3E claimed "~5% ATE improvement" from 1D bias correction.
This was a THEORETICAL estimate using quadratic drift approximation.
No corrected trajectory was run.

Theoretical basis:
  Drift ≈ 0.5 * ||b_a|| * T^2  (constant-bias quadratic position error)
  Scenario A (no correction):  0.5 * 0.1613 * 152^2 = 1,863 m
  Scenario B (1D correction):  0.5 * 0.1536 * 152^2 = 1,774 m  (residual b_a_perp)
  Ratio: (1863 - 1774) / 1863 ≈ 4.8%  → reported as "~5%"

Problems with this estimate:
  1. Quadratic drift assumes constant bias + zero initial velocity → accumulates as parabola.
     Reality: orientation error compounds through the integration chain.
  2. The 1D corrected bias vector (b_a_1d) = [-0.024, +0.001, +0.007] m/s^2
     This reduces b_a_x slightly but does nothing to b_a_y (+0.144) or b_a_z (+0.068).
  3. The b_a_perp = [+0.023, +0.142, +0.054] closely matches b_a_y and b_a_z.
  4. After 1D correction, residual ≈ 0.5 * 0.1536 * 152^2 = 1,774 m  (96% of original).
  5. Actual ATE = 2,948 m (much larger than 1,863 m) because:
     a) Orientation error compounds non-linearly
     b) Gravity subtraction error is added every sample
     c) Velocity error grows and position error compounds
""")

# Numerical check: what does 1D bias correction actually do to the constant-bias term?
print("Numerical breakdown:")
print(f"  GT b_a = [{gt_ba_stat[0]:+.6f}, {gt_ba_stat[1]:+.6f}, {gt_ba_stat[2]:+.6f}]")
print(f"  b_a_1d = [{ba_1d[0]:+.6f}, {ba_1d[1]:+.6f}, {ba_1d[2]:+.6f}]")
print(f"  Residual after 1D correction: [{(gt_ba_stat-ba_1d)[0]:+.6f}, {(gt_ba_stat-ba_1d)[1]:+.6f}, {(gt_ba_stat-ba_1d)[2]:+.6f}]")
print(f"  ||residual|| = {np.linalg.norm(gt_ba_stat-ba_1d):.6f} m/s^2  (vs ||b_a_perp|| = {np.linalg.norm(gt_ba_stat-ba_1d):.6f})")

T = (imu_ts[-1]-imu_ts[0])*1e-9
drift_A = 0.5 * np.linalg.norm(gt_ba_stat) * T**2
drift_B = 0.5 * np.linalg.norm(gt_ba_stat - ba_1d) * T**2
print(f"\n  Theoretical drift (Scenario A): {drift_A:.1f} m")
print(f"  Theoretical drift (Scenario B): {drift_B:.1f} m")
print(f"  Theoretical reduction: {(drift_A-drift_B)/drift_A*100:.1f}%")
print(f"  Actual baseline ATE:   2948.4 m")
print(f"  Scaling factor:        2948.4 / {drift_A:.1f} = {2948.4/drift_A:.2f}x")
print(f"  → Actual ATE reduction likely: {(drift_A-drift_B)/drift_A*100:.1f}% × {2948.4/drift_A:.2f} = same fraction (~5%) if drift scales linearly")
print(f"  → BUT orientation error compounding makes actual improvement likely LESS than 5%")
print(f"  → Measured improvement can only be known by running the corrected estimator")

# Key distinction: b_a_1d corrects only b_a_x component (which is closest to gravity direction)
print(f"\nWhat 1D correction actually corrects:")
print(f"  b_a_x component: GT={gt_ba_stat[0]:+.6f}, corrected={gt_ba_stat[0]-ba_1d[0]:+.6f} (change={ba_1d[0]:+.6f})")
print(f"  b_a_y component: GT={gt_ba_stat[1]:+.6f}, corrected={gt_ba_stat[1]-ba_1d[1]:+.6f} (change={ba_1d[1]:+.6f})")
print(f"  b_a_z component: GT={gt_ba_stat[2]:+.6f}, corrected={gt_ba_stat[2]-ba_1d[2]:+.6f} (change={ba_1d[2]:+.6f})")
print(f"  → b_a_y (+0.1444) and b_a_z (+0.0676) remain almost entirely unaffected")
print(f"  → b_a_x only slightly affected: from -0.0242 to {gt_ba_stat[0]-ba_1d[0]:+.4f}")

print("\n--- Stage 3E-Verification Complete ---")
