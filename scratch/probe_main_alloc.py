"""
Trace exactly which line in src/main.py produced the 30.29 MB peak.
"""
import sys, os, tracemalloc, time, gc
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from data_loader import load_imu_data, IMU_TIMESTAMP_COL
from preprocessing import GYRO_COLS, ACCEL_COLS
from bias_estimation import estimate_gyro_bias
from trajectory import compute_gravity_alignment_quaternion, STANDARD_GRAVITY
from trajectory_export import export_tum_trajectory, validate_tum_file
from degradation import apply_provisional_degradation, sanitize_imu_measurements

tracemalloc.start()
def probe(label):
    c, p = tracemalloc.get_traced_memory()
    print(f"[{label:<35}] Curr: {c/(1024*1024):6.2f} MiB | Peak: {p/(1024*1024):6.2f} MiB")

probe("Start")
df_raw = load_imu_data("data/raw/euroc/machine_hall/MH_02_easy/mav0/imu0/data.csv")
probe("After load_imu_data")
df_degraded = df_raw
probe("After scenario")
df_sanitized, san_meta = sanitize_imu_measurements(df_degraded)
del df_degraded
gc.collect()
probe("After sanitize & del df_degraded")

MICRO_GAP_THRESHOLD_S = 0.025
bias_result = estimate_gyro_bias(df_sanitized, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
gyro_bias = bias_result["gyro_bias"]
probe("After estimate_gyro_bias")

df_proc = df_sanitized
valid_gyro_mask = df_proc["valid_gyro"].to_numpy(dtype=bool)
for col_idx, col in enumerate(GYRO_COLS):
    df_proc.loc[valid_gyro_mask, col] = df_proc.loc[valid_gyro_mask, col] - gyro_bias[col_idx]
probe("After in-place gyro bias sub")

initial_accel_sample = df_proc.loc[df_proc["valid_accel"], ACCEL_COLS].iloc[:100].mean().to_numpy()
q0 = compute_gravity_alignment_quaternion(initial_accel_sample, yaw_rad=0.0)
probe("After q0")

# Integration
n_samples = len(df_proc)
ts_ns = df_proc[IMU_TIMESTAMP_COL].to_numpy(dtype=np.int64)
dt_s = np.zeros(n_samples, dtype=np.float64)
if n_samples > 1:
    dt_s[1:] = (ts_ns[1:] - ts_ns[:-1]).astype(np.float64) * 1e-9
gyro_rates = df_proc[GYRO_COLS].to_numpy(dtype=np.float64)
probe("After extract numpy arrays")

quats_wxyz = np.empty((n_samples, 4), dtype=np.float64)
curr_q = q0.copy()
quats_wxyz[0] = curr_q

def _rotvec_to_dq_wxyz(rv):
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12: return np.array([1.,0.,0.,0.])
    half = 0.5*angle; s=np.sin(half)/angle
    return np.array([np.cos(half), rv[0]*s, rv[1]*s, rv[2]*s])
def _qmul_wxyz(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

for i in range(1, n_samples):
    is_macro_gap = dt_s[i] > MICRO_GAP_THRESHOLD_S
    both_gyro_valid = bool(valid_gyro_mask[i - 1]) and bool(valid_gyro_mask[i])
    if both_gyro_valid and not is_macro_gap:
        w_mid = 0.5 * (gyro_rates[i - 1] + gyro_rates[i])
        dq = _rotvec_to_dq_wxyz(w_mid * dt_s[i])
        curr_q = _qmul_wxyz(curr_q, dq)
        curr_q /= np.linalg.norm(curr_q)
    quats_wxyz[i] = curr_q
probe("After orientation loop")

valid_accel_mask = df_proc["valid_accel"].to_numpy(dtype=bool)
a_body = df_proc[ACCEL_COLS].to_numpy(dtype=np.float64, copy=True)
a_body[~valid_accel_mask] = 0.0

w_q = quats_wxyz[:, 0]; x_q = quats_wxyz[:, 1]; y_q = quats_wxyz[:, 2]; z_q = quats_wxyz[:, 3]
vx = a_body[:, 0]; vy = a_body[:, 1]; vz = a_body[:, 2]
a_world_x = (1.0 - 2.0 * (y_q * y_q + z_q * z_q)) * vx + 2.0 * (x_q * y_q - w_q * z_q) * vy + 2.0 * (x_q * z_q + w_q * y_q) * vz
a_world_y = 2.0 * (x_q * y_q + w_q * z_q) * vx + (1.0 - 2.0 * (x_q * x_q + z_q * z_q)) * vy + 2.0 * (y_q * z_q - w_q * x_q) * vz
a_world_z = 2.0 * (x_q * z_q - w_q * y_q) * vx + 2.0 * (y_q * z_q + w_q * x_q) * vy + (1.0 - 2.0 * (x_q * x_q + y_q * y_q)) * vz
a_world_z += -STANDARD_GRAVITY
a_lin_world = np.column_stack([a_world_x, a_world_y, a_world_z])
a_lin_world[~valid_accel_mask] = 0.0
probe("After a_lin_world vectorization")

pos_arr = np.zeros((n_samples, 3), dtype=np.float64)
vel_arr = np.zeros((n_samples, 3), dtype=np.float64)
for i in range(1, n_samples):
    dt = dt_s[i]
    is_macro_gap = dt > MICRO_GAP_THRESHOLD_S
    both_accel_valid = bool(valid_accel_mask[i - 1]) and bool(valid_accel_mask[i])
    if is_macro_gap or not both_accel_valid:
        vel_arr[i] = vel_arr[i - 1]
        pos_arr[i] = pos_arr[i - 1] + vel_arr[i - 1] * dt
    else:
        v_next = vel_arr[i - 1] + 0.5 * (a_lin_world[i - 1] + a_lin_world[i]) * dt
        pos_arr[i] = pos_arr[i - 1] + 0.5 * (vel_arr[i - 1] + v_next) * dt
        vel_arr[i] = v_next
probe("After trajectory loop")

# Stream export
export_tum_trajectory(
    output_path="scratch/temp_probe_out.txt",
    timestamps_ns=ts_ns,
    pos_arr=pos_arr,
    quats_wxyz=quats_wxyz,
)
probe("After streaming export")

val_res = validate_tum_file("scratch/temp_probe_out.txt", expected_row_count=n_samples)
probe("After validate_tum_file")

snap = tracemalloc.take_snapshot()
tracemalloc.stop()
if os.path.exists("scratch/temp_probe_out.txt"):
    os.remove("scratch/temp_probe_out.txt")

print("\nTop 10 allocations at peak:")
for s in snap.statistics("lineno")[:10]:
    print(s)
