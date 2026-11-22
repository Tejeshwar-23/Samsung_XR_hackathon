"""
Stage 3F-A — Full-Pipeline Memory Audit (READ-ONLY)
Measures algorithm-owned Python heap memory across every phase of the pipeline.
Uses tracemalloc throughout the full execution pipeline from CSV load to export.
Does not modify any production source files or overwrite outputs.
"""
import sys, os, gc, tracemalloc
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path

# Process memory tracking
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    _k32 = ctypes.windll.kernel32
    _psapi = ctypes.windll.psapi
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.GetCurrentProcess.argtypes = []
    _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    _psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD
    ]
    def get_rss_bytes():
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = _k32.GetCurrentProcess()
        if _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), ctypes.sizeof(PROCESS_MEMORY_COUNTERS)):
            return counters.WorkingSetSize, counters.PeakWorkingSetSize
        return 0, 0
else:
    def get_rss_bytes():
        return 0, 0

from data_loader import load_imu_data, IMU_TIMESTAMP_COL
from preprocessing import GYRO_COLS, ACCEL_COLS
from bias_estimation import estimate_gyro_bias
from trajectory import compute_gravity_alignment_quaternion, STANDARD_GRAVITY
from trajectory_export import export_tum_trajectory
from degradation import sanitize_imu_measurements

IMU_PATH = "data/raw/euroc/machine_hall/MH_02_easy/mav0/imu0/data.csv"
DUMMY_EXPORT_PATH = "scratch/temp_mem_audit_traj.txt"

def sep(title):
    print("\n" + "="*75)
    print(title)
    print("="*75)

sep("PHASE-BY-PHASE FULL-PIPELINE MEMORY PROFILING")

gc.collect()
tracemalloc.start()
tm_baseline_curr, tm_baseline_peak = tracemalloc.get_traced_memory()
rss_curr, rss_peak = get_rss_bytes()

phase_snapshots = []

def record_phase(phase_name):
    curr, peak = tracemalloc.get_traced_memory()
    r_curr, r_peak = get_rss_bytes()
    snapshot = tracemalloc.take_snapshot()
    phase_snapshots.append((phase_name, curr, peak, r_curr, r_peak, snapshot))
    print(f"[{phase_name:<30}] Heap Curr: {curr/(1024*1024):6.2f} MiB | Heap Peak: {peak/(1024*1024):6.2f} MiB | RSS: {r_curr/(1024*1024):6.2f} MiB")

record_phase("0. Baseline (imports done)")

# Phase 1: Raw CSV Load
df_raw = load_imu_data(IMU_PATH)
record_phase("1. Load IMU Data (df_raw)")

# Phase 2: Scenario copy
df_degraded = df_raw.copy()
record_phase("2. Scenario Handling (df_degraded)")

# Phase 3: Sanitization
df_sanitized, san_meta = sanitize_imu_measurements(df_degraded)
record_phase("3. Sanitize Measurements (df_san)")

# Phase 4: Stationary Bias Estimation
MICRO_GAP_THRESHOLD_S = 0.025
bias_result = estimate_gyro_bias(df_sanitized, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
gyro_bias = bias_result["gyro_bias"]
record_phase("4. Gyro Bias Estimation")

# Phase 5: Bias subtraction (df_proc creation)
df_proc = df_sanitized.copy()
valid_gyro_mask = df_proc["valid_gyro"].to_numpy(dtype=bool)
for col_idx, col in enumerate(GYRO_COLS):
    df_proc.loc[valid_gyro_mask, col] = df_proc.loc[valid_gyro_mask, col] - gyro_bias[col_idx]
record_phase("5. Bias Subtraction (df_proc)")

# Phase 6: Initial Gravity Alignment
initial_accel_sample = df_proc.loc[df_proc["valid_accel"], ACCEL_COLS].iloc[:100].mean().to_numpy()
q0 = compute_gravity_alignment_quaternion(initial_accel_sample, yaw_rad=0.0)
record_phase("6. Initial Gravity Alignment (q0)")

# --- Notice: in production main.py, tracemalloc stops HERE.
# That explains the earlier 15.47 MiB measurement!
# In this audit, tracemalloc CONTINUES across all remaining phases:

# Phase 7: Orientation Integration Setup & Loop
n_samples = len(df_proc)
ts_ns = df_proc[IMU_TIMESTAMP_COL].to_numpy(dtype=np.int64)
dt_s = np.zeros(n_samples, dtype=np.float64)
if n_samples > 1:
    dt_s[1:] = (ts_ns[1:] - ts_ns[:-1]).astype(np.float64) * 1e-9

gyro_rates = df_proc[GYRO_COLS].to_numpy(dtype=np.float64)

def _rotvec_to_dq_wxyz(rv: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half = 0.5 * angle
    s = np.sin(half) / angle
    return np.array([np.cos(half), rv[0] * s, rv[1] * s, rv[2] * s], dtype=np.float64)

def _qmul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)

quats_wxyz = np.empty((n_samples, 4), dtype=np.float64)
curr_q = q0.copy()
quats_wxyz[0] = curr_q

for i in range(1, n_samples):
    is_macro_gap = dt_s[i] > MICRO_GAP_THRESHOLD_S
    both_gyro_valid = bool(valid_gyro_mask[i - 1]) and bool(valid_gyro_mask[i])
    if both_gyro_valid and not is_macro_gap:
        w_mid = 0.5 * (gyro_rates[i - 1] + gyro_rates[i])
        dq = _rotvec_to_dq_wxyz(w_mid * dt_s[i])
        curr_q = _qmul_wxyz(curr_q, dq)
        curr_q /= np.linalg.norm(curr_q)
    quats_wxyz[i] = curr_q

record_phase("7. Orientation Loop (quats_wxyz)")

# Phase 8: Linear Acceleration Computation
valid_accel_mask = df_proc["valid_accel"].to_numpy(dtype=bool)
a_body = df_proc[ACCEL_COLS].to_numpy(dtype=np.float64, copy=True)
a_body[~valid_accel_mask] = 0.0

w_q = quats_wxyz[:, 0]
x_q = quats_wxyz[:, 1]
y_q = quats_wxyz[:, 2]
z_q = quats_wxyz[:, 3]
vx = a_body[:, 0]
vy = a_body[:, 1]
vz = a_body[:, 2]

a_world_x = (1.0 - 2.0 * (y_q * y_q + z_q * z_q)) * vx + 2.0 * (x_q * y_q - w_q * z_q) * vy + 2.0 * (x_q * z_q + w_q * y_q) * vz
a_world_y = 2.0 * (x_q * y_q + w_q * z_q) * vx + (1.0 - 2.0 * (x_q * x_q + z_q * z_q)) * vy + 2.0 * (y_q * z_q - w_q * x_q) * vz
a_world_z = 2.0 * (x_q * z_q - w_q * y_q) * vx + 2.0 * (y_q * z_q + w_q * x_q) * vy + (1.0 - 2.0 * (x_q * x_q + y_q * y_q)) * vz
a_world_z += -STANDARD_GRAVITY
a_lin_world = np.column_stack([a_world_x, a_world_y, a_world_z])
a_lin_world[~valid_accel_mask] = 0.0
record_phase("8. World Accel (a_lin_world)")

# Phase 9: Trajectory Integration (Position & Velocity)
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

record_phase("9. Trajectory Loop (pos/vel_arr)")

# Phase 10: Trajectory DataFrame Creation
df_trajectory_est = pd.DataFrame({
    IMU_TIMESTAMP_COL: df_proc[IMU_TIMESTAMP_COL].values,
    "p_x": pos_arr[:, 0],
    "p_y": pos_arr[:, 1],
    "p_z": pos_arr[:, 2],
    "v_x": vel_arr[:, 0],
    "v_y": vel_arr[:, 1],
    "v_z": vel_arr[:, 2],
    "q_w": quats_wxyz[:, 0],
    "q_x": quats_wxyz[:, 1],
    "q_y": quats_wxyz[:, 2],
    "q_z": quats_wxyz[:, 3],
})
record_phase("10. Output DataFrame (df_traj)")

# Phase 11: Export TUM Trajectory (to scratch/dummy)
export_tum_trajectory(
    df_trajectory=df_trajectory_est,
    output_path=DUMMY_EXPORT_PATH,
    time_col=IMU_TIMESTAMP_COL,
    pos_cols=("p_x", "p_y", "p_z"),
    quat_cols_wxyz=("q_w", "q_x", "q_y", "q_z"),
)
record_phase("11. TUM Export File Generation")

total_curr, total_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Clean up temporary scratch export file
if os.path.exists(DUMMY_EXPORT_PATH):
    os.remove(DUMMY_EXPORT_PATH)

sep("OBJECT SIZE ANALYSIS & AUDIT OF ALGORITHM-OWNED MEMORY")

print(f"IMU Rows N = {n_samples}")
objects_to_measure = [
    ("df_raw (DataFrame)", df_raw),
    ("df_degraded (DataFrame)", df_degraded),
    ("df_sanitized (DataFrame)", df_sanitized),
    ("df_proc (DataFrame)", df_proc),
    ("df_trajectory_est (DataFrame)", df_trajectory_est),
    ("ts_ns (int64 array)", ts_ns),
    ("dt_s (float64 array)", dt_s),
    ("gyro_rates (N,3 float64)", gyro_rates),
    ("quats_wxyz (N,4 float64)", quats_wxyz),
    ("a_body (N,3 float64)", a_body),
    ("a_lin_world (N,3 float64)", a_lin_world),
    ("pos_arr (N,3 float64)", pos_arr),
    ("vel_arr (N,3 float64)", vel_arr),
    ("valid_gyro_mask (bool array)", valid_gyro_mask),
    ("valid_accel_mask (bool array)", valid_accel_mask),
]

total_direct_algorithm_bytes = 0
print(f"{'Object Name':<32} {'Type':<18} {'Shape/Length':<16} {'Size (KiB)':>12} {'Size (MiB)':>12}")
print("-" * 92)
for name, obj in objects_to_measure:
    if isinstance(obj, pd.DataFrame):
        mem = obj.memory_usage(deep=True).sum()
        shape_str = str(obj.shape)
    elif isinstance(obj, np.ndarray):
        mem = obj.nbytes
        shape_str = str(obj.shape)
    else:
        mem = sys.getsizeof(obj)
        shape_str = "-"
    total_direct_algorithm_bytes += mem
    print(f"{name:<32} {type(obj).__name__:<18} {shape_str:<16} {mem/1024:12.1f} {mem/(1024*1024):12.3f}")

print("-" * 92)
print(f"Sum of concurrent algorithm data structures in memory: {total_direct_algorithm_bytes/(1024*1024):.2f} MiB")

sep("DETAILED TOP ALLOCATIONS AT FINAL PHASE SNAPSHOT")
final_snapshot = phase_snapshots[-1][5]
top_stats = final_snapshot.statistics("lineno")
print("Top 15 memory allocation lines:")
for stat in top_stats[:15]:
    print(stat)

sep("SUMMARY METRICS")
print(f"Overall Full-Pipeline Python Heap Peak (tracemalloc): {total_peak/(1024*1024):.2f} MiB")
print(f"Current Phase 6 Production Measurement Peak:          {phase_snapshots[6][2]/(1024*1024):.2f} MiB")
print(f"Total Process Peak Working Set (Peak RSS):            {phase_snapshots[-1][4]/(1024*1024):.2f} MiB")

print("\n--- Memory Audit Script Finished ---")
