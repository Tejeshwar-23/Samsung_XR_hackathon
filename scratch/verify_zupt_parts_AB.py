"""Stage 3D-Verification: Parts A and B — READ-ONLY audit script."""
import sys, numpy as np
sys.path.insert(0, 'src')
from imu_pipeline import load_imu_data, sanitize_imu_measurements, detect_quasi_stationary_intervals
from evaluator import load_ground_truth

imu_path = 'data/raw/euroc/machine_hall/MH_02_easy/mav0/imu0/data.csv'
gt_path  = 'data/raw/euroc/machine_hall/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv'

df_raw  = load_imu_data(imu_path)
df_san  = sanitize_imu_measurements(df_raw)
df_proc = df_san.copy()

ts_ns = df_proc['timestamp_ns'].values.astype(np.int64)
gx = df_proc['gyr_x'].values;  gy = df_proc['gyr_y'].values;  gz = df_proc['gyr_z'].values
ax_ = df_proc['acc_x'].values; ay_ = df_proc['acc_y'].values; az_ = df_proc['acc_z'].values

N = len(ts_ns)
print(f"IMU samples N = {N}")

# ---- gravity alignment (identical to Stage 3D) ----
N0 = 100
a0 = np.array([ax_[:N0].mean(), ay_[:N0].mean(), az_[:N0].mean()])
g_mag = np.linalg.norm(a0)
g_world = np.array([0.0, 0.0, g_mag])
v_cross = np.cross(a0 / g_mag, g_world / g_mag)
s = np.linalg.norm(v_cross); c = np.dot(a0 / g_mag, g_world / g_mag)
if s < 1e-10:
    q0 = np.array([1.0, 0, 0, 0])
else:
    ax_rot = v_cross / s
    half = np.arcsin(np.clip(s, -1, 1)) / 2
    q0 = np.array([np.cos(half), *(ax_rot * np.sin(half))])
    q0 /= np.linalg.norm(q0)
print(f"Initial quaternion q0 = {q0}")
print(f"g_mag = {g_mag:.6f} m/s^2")

def qmul(q, r):
    w0,x0,y0,z0 = q; w1,x1,y1,z1 = r
    return np.array([w0*w1-x0*x1-y0*y1-z0*z1,
                     w0*x1+x0*w1+y0*z1-z0*y1,
                     w0*y1-x0*z1+y0*w1+z0*x1,
                     w0*z1+x0*y1-y0*x1+z0*w1])

def qrot(q, v):
    qv = np.array([0.0, *v])
    return qmul(qmul(q, qv), np.array([q[0], -q[1], -q[2], -q[3]]))[1:]

# ---- detect ZUPT interval ----
det = detect_quasi_stationary_intervals(df_proc)
intervals = det['detected_intervals']
print(f"\nDetected stationary intervals: {intervals}")
ZUPT_START = intervals[0][0] if intervals else None
ZUPT_END   = intervals[0][1] if intervals else None
print(f"ZUPT window: [{ZUPT_START}, {ZUPT_END}] (expected [5180, 5979])")

# ---- full integration with ZUPT ----
q   = q0.copy()
vel = np.zeros(3)
pos = np.zeros(3)
vel_arr = np.zeros((N, 3))
pos_arr = np.zeros((N, 3))
q_arr   = np.zeros((N, 4))
q_arr[0]   = q
vel_arr[0] = vel
pos_arr[0] = pos

for i in range(1, N):
    dt = (ts_ns[i] - ts_ns[i-1]) * 1e-9
    omega = np.array([gx[i-1], gy[i-1], gz[i-1]])
    angle = np.linalg.norm(omega) * dt
    if angle > 1e-12:
        ax_v = omega / np.linalg.norm(omega)
        dq = np.array([np.cos(angle/2), *(ax_v * np.sin(angle/2))])
    else:
        dq = np.array([1.0, 0, 0, 0])
    q = qmul(q, dq); q /= np.linalg.norm(q)
    a_body  = np.array([ax_[i-1], ay_[i-1], az_[i-1]])
    a_world = qrot(q, a_body)
    a_lin   = a_world - np.array([0, 0, g_mag])
    if ZUPT_START is not None and ZUPT_START <= i <= ZUPT_END:
        vel = np.zeros(3)
        pos_arr[i] = pos_arr[i-1]
    else:
        vel   = vel + a_lin * dt
        pos   = pos_arr[i-1] + vel * dt
        pos_arr[i] = pos
    vel_arr[i] = vel
    q_arr[i]   = q

print("\n" + "="*60)
print("PART A: Velocity at index 5179 (last sample BEFORE ZUPT at 5180)")
print("="*60)
v5179 = vel_arr[5179]
print(f"vel[5179]  X = {v5179[0]:+.6f} m/s")
print(f"vel[5179]  Y = {v5179[1]:+.6f} m/s")
print(f"vel[5179]  Z = {v5179[2]:+.6f} m/s")
print(f"vel[5179] norm = {np.linalg.norm(v5179):.6f} m/s")

# Compare with GT velocity at ts_ns[5179]
gt_df = load_ground_truth(gt_path)
gt_ts = gt_df['timestamp_ns'].values.astype(np.int64)
# GT velocity columns
gt_vx_col = [c for c in gt_df.columns if 'vel' in c.lower() and 'x' in c.lower()]
gt_vy_col = [c for c in gt_df.columns if 'vel' in c.lower() and 'y' in c.lower()]
gt_vz_col = [c for c in gt_df.columns if 'vel' in c.lower() and 'z' in c.lower()]
print(f"\nGT velocity columns: vx={gt_vx_col}, vy={gt_vy_col}, vz={gt_vz_col}")
print(f"All GT columns: {list(gt_df.columns)}")

target_ts = ts_ns[5179]
idx_gt = np.argmin(np.abs(gt_ts - target_ts))
dt_match = (gt_ts[idx_gt] - target_ts) * 1e-9
print(f"\nIMU ts[5179] = {target_ts} ns ({target_ts*1e-9:.3f} s)")
print(f"Nearest GT index = {idx_gt}, GT ts = {gt_ts[idx_gt]} ns ({gt_ts[idx_gt]*1e-9:.3f} s)")
print(f"Time delta = {dt_match:.4f} s")

if gt_vx_col and gt_vy_col and gt_vz_col:
    gvx = gt_df[gt_vx_col[0]].values[idx_gt]
    gvy = gt_df[gt_vy_col[0]].values[idx_gt]
    gvz = gt_df[gt_vz_col[0]].values[idx_gt]
    gt_vel = np.array([gvx, gvy, gvz])
    print(f"\nGT vel at t=25.9s: X={gvx:+.6f}, Y={gvy:+.6f}, Z={gvz:+.6f} m/s")
    print(f"GT vel norm = {np.linalg.norm(gt_vel):.6f} m/s")
    err = v5179 - gt_vel
    print(f"Error (est-GT): X={err[0]:+.6f}, Y={err[1]:+.6f}, Z={err[2]:+.6f} m/s, norm={np.linalg.norm(err):.6f} m/s")

print("\n--- Cumulative integrated accel breakdown index 0 → 5179 ---")
vel_noZ = np.zeros(3); cum_raw_world = np.zeros(3); cum_grav_sub = np.zeros(3)
for i in range(1, 5180):
    dt = (ts_ns[i] - ts_ns[i-1]) * 1e-9
    a_body  = np.array([ax_[i-1], ay_[i-1], az_[i-1]])
    a_world = qrot(q_arr[i], a_body)
    g_comp  = np.array([0, 0, g_mag])
    cum_raw_world += a_world * dt
    cum_grav_sub  += (-g_comp) * dt
    vel_noZ       += (a_world - g_comp) * dt
print(f"Cumulative raw world accel integral: [{cum_raw_world[0]:.4f}, {cum_raw_world[1]:.4f}, {cum_raw_world[2]:.4f}] m/s")
print(f"Cumulative gravity subtraction:      [{cum_grav_sub[0]:.4f}, {cum_grav_sub[1]:.4f}, {cum_grav_sub[2]:.4f}] m/s")
print(f"Net linear velocity (no-ZUPT check): [{vel_noZ[0]:.4f}, {vel_noZ[1]:.4f}, {vel_noZ[2]:.4f}] m/s, norm={np.linalg.norm(vel_noZ):.4f}")

print("\n" + "="*60)
print("PART B: Verify ZUPT output file")
print("="*60)
zupt_path = 'outputs/exp_MH_02_easy_zupt.txt'
import os
size_bytes = os.path.getsize(zupt_path)
print(f"File: {zupt_path}  size={size_bytes:,} bytes")

poses = []
with open(zupt_path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        poses.append([float(x) for x in parts])
poses = np.array(poses)
print(f"Row count in ZUPT output: {len(poses)} (expected 30,400)")

# Quaternion validity
q_out = poses[:, 4:]  # w x y z — check column order
# also try poses[:,1:5] if that's the layout (ts tx ty tz qw qx qy qz)
q_check = poses[:, 4:]  # columns 4-7 = qw qx qy qz
norms = np.linalg.norm(q_check, axis=1)
print(f"\nQuaternion norm stats:  min={norms.min():.8f}, max={norms.max():.8f}, mean={norms.mean():.8f}")
finite = np.all(np.isfinite(poses), axis=1)
print(f"All-finite rows: {finite.sum()} / {len(poses)}")

# ZUPT boundary checks (re-use pos_arr, vel_arr from integration)
print("\n--- ZUPT boundary checks ---")
print(f"pos[5179] = {pos_arr[5179]}")
print(f"pos[5180] = {pos_arr[5180]}  (should == pos[5179] during ZUPT)")
print(f"pos unchanged at ZUPT start: {np.allclose(pos_arr[5180], pos_arr[5179])}")
print(f"vel[5180] = {vel_arr[5180]}  (should be [0,0,0])")
print(f"vel zeroed at ZUPT start:    {np.allclose(vel_arr[5180], np.zeros(3))}")

# Transition OUT at index 5980
dt5980 = (ts_ns[5980] - ts_ns[5979]) * 1e-9
expected_pos5980 = pos_arr[5979] + vel_arr[5979] * dt5980
print(f"\nvel[5979]       = {vel_arr[5979]}  (last ZUPT sample, should be [0,0,0])")
print(f"pos[5979]       = {pos_arr[5979]}")
print(f"pos[5980]       = {pos_arr[5980]}")
print(f"expected pos[5980] = pos[5979] + vel[5979]*dt = {expected_pos5980}")
print(f"Transition correct: {np.allclose(pos_arr[5980], expected_pos5980, atol=1e-10)}")

# Pre-ZUPT: baseline identical (re-run baseline for comparison)
print("\n--- Verifying pre-ZUPT baseline identity (first 5180 samples) ---")
q_b = q0.copy(); vel_b = np.zeros(3); pos_b_arr = np.zeros((5181, 3))
for i in range(1, 5181):
    dt = (ts_ns[i] - ts_ns[i-1]) * 1e-9
    omega = np.array([gx[i-1], gy[i-1], gz[i-1]])
    angle = np.linalg.norm(omega) * dt
    if angle > 1e-12:
        ax_v = omega / np.linalg.norm(omega)
        dq = np.array([np.cos(angle/2), *(ax_v * np.sin(angle/2))])
    else:
        dq = np.array([1.0, 0, 0, 0])
    q_b = qmul(q_b, dq); q_b /= np.linalg.norm(q_b)
    a_body  = np.array([ax_[i-1], ay_[i-1], az_[i-1]])
    a_world = qrot(q_b, a_body)
    a_lin   = a_world - np.array([0, 0, g_mag])
    vel_b   = vel_b + a_lin * dt
    pos_b_arr[i] = pos_b_arr[i-1] + vel_b * dt

max_pos_diff = np.max(np.abs(pos_arr[:5180] - pos_b_arr[:5180]))
print(f"Max position difference pre-ZUPT vs pure baseline: {max_pos_diff:.2e} m (should be ~0)")

# Detector scope check
print(f"\nDetected interval matches expectation [5180,5979]: {intervals == [(5180, 5979)] if intervals else 'NO intervals detected'}")
print(f"Number of detected intervals: {len(intervals)} (expected 1)")

print("\nParts A and B complete.")
