"""
Samsung XR Hackathon 2026 - Track 3 (Path Finder: Trajectory Estimation Under Sensor Degradation)
Autonomous Dead-Reckoning Pipeline Entrypoint.
"""

import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import gc
import json
import logging
import math
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

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
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]

try:
    from src.data_loader import (
        IMU_TIMESTAMP_COL,
        _clean_header_name,
        resolve_imu_column_mapping,
    )
    from src.preprocessing import GYRO_COLS, ACCEL_COLS
    from src.bias_estimation import estimate_gyro_bias, detect_quasi_stationary_intervals
    from src.trajectory import (
        compute_gravity_alignment_quaternion,
        STANDARD_GRAVITY,
    )
    from src.trajectory_export import export_tum_trajectory, validate_tum_file
    from src.degradation import (
        GYRO_SATURATION_RAD_S,
        ACCEL_SATURATION_M_S2,
    )
except ModuleNotFoundError:
    from data_loader import (
        IMU_TIMESTAMP_COL,
        _clean_header_name,
        resolve_imu_column_mapping,
    )
    from preprocessing import GYRO_COLS, ACCEL_COLS
    from bias_estimation import estimate_gyro_bias, detect_quasi_stationary_intervals
    from trajectory import (
        compute_gravity_alignment_quaternion,
        STANDARD_GRAVITY,
    )
    from trajectory_export import export_tum_trajectory, validate_tum_file
    from degradation import (
        GYRO_SATURATION_RAD_S,
        ACCEL_SATURATION_M_S2,
    )


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


logger = logging.getLogger("samsung_xr_pipeline")

_CANONICAL_ORDER = [IMU_TIMESTAMP_COL] + GYRO_COLS + ACCEL_COLS


def get_peak_process_memory_bytes() -> Tuple[int, str, bool]:
    if sys.platform == "win32":
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = _k32.GetCurrentProcess()
            success = _psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            )
            if success and counters.PeakWorkingSetSize > 0:
                return (
                    int(counters.PeakWorkingSetSize),
                    "GetProcessMemoryInfo/PeakWorkingSetSize",
                    True,
                )
        except Exception:
            pass

    if _HAS_PSUTIL:
        try:
            mem_info = psutil.Process(os.getpid()).memory_info()
            if hasattr(mem_info, "peak_wset") and mem_info.peak_wset > 0:
                return int(mem_info.peak_wset), "psutil/peak_wset", True
            if hasattr(mem_info, "rss") and mem_info.rss > 0:
                return int(mem_info.rss), "psutil/rss", True
        except Exception:
            pass

    return 0, "UNAVAILABLE", False


# Sensor Outlier & Saturation Gating Thresholds (Task C compliance)
GYRO_OUTLIER_THRESHOLD_RAD_S = 10.0      # rad/s (~573 deg/s, reject severe non-physical gyro rates)
ACCEL_OUTLIER_THRESHOLD_M_S2 = 50.0      # m/s^2 (~5.1 g, reject non-physical acceleration impacts)

# Velocity Cap & Outage Decay Parameters (Configuration A)
V_MAX = 10.0                             # m/s (Maximum physical velocity cap)
OUTAGE_DECAY_RATE = 1.5                  # s^-1 (Exponential outage velocity decay rate)


def _load_imu_arrays_lean(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorized CSV loading directly into compact NumPy arrays while
    preserving every input row 1:1.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"IMU file not found: {path}")

    header_df = pd.read_csv(path, nrows=0)
    col_map = resolve_imu_column_mapping(header_df.columns)
    del header_df

    usecols = [col_map[req] for req in _CANONICAL_ORDER]
    df_sub = pd.read_csv(path, usecols=usecols)
    if df_sub.empty:
        raise ValueError(f"IMU dataset at '{path}' is empty.")

    ts_col_raw = col_map[IMU_TIMESTAMP_COL]
    gyro_raw = [col_map[c] for c in GYRO_COLS]
    accel_raw = [col_map[c] for c in ACCEL_COLS]

    ts_ns = df_sub[ts_col_raw].to_numpy(dtype=np.int64, copy=True)
    gyro = df_sub[gyro_raw].to_numpy(dtype=np.float64, copy=True)
    accel = df_sub[accel_raw].to_numpy(dtype=np.float64, copy=True)
    del df_sub

    if len(ts_ns) > 1 and np.any((ts_ns[1:] - ts_ns[:-1]) < 0):
        raise ValueError(f"IMU timestamps are not monotonically non-decreasing in '{path}'.")

    return ts_ns, gyro, accel


def _sanitize_arrays_inplace(
    gyro: np.ndarray,
    accel: np.ndarray,
    gyro_sat_thresh: float = GYRO_OUTLIER_THRESHOLD_RAD_S,
    accel_sat_thresh: float = ACCEL_OUTLIER_THRESHOLD_M_S2,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(gyro)
    w_finite = np.isfinite(gyro).all(axis=1)
    a_finite = np.isfinite(accel).all(axis=1)

    w_norm_sq = np.zeros(n, dtype=np.float64)
    a_norm_sq = np.zeros(n, dtype=np.float64)
    w_norm_sq[w_finite] = np.sum(gyro[w_finite] * gyro[w_finite], axis=1)
    a_norm_sq[a_finite] = np.sum(accel[a_finite] * accel[a_finite], axis=1)

    w_in_range = w_finite & (w_norm_sq <= gyro_sat_thresh * gyro_sat_thresh)
    a_in_range = a_finite & (a_norm_sq <= accel_sat_thresh * accel_sat_thresh)
    del w_norm_sq, a_norm_sq

    is_six_axis_zero = w_finite & a_finite & (np.sum(gyro * gyro, axis=1) < 1e-12) & (np.sum(accel * accel, axis=1) < 1e-10)
    terminal_failure = np.zeros(n, dtype=bool)
    if n > 0 and is_six_axis_zero[-1]:
        k = n - 1
        while k >= 0 and is_six_axis_zero[k]:
            terminal_failure[k] = True
            k -= 1
    del is_six_axis_zero

    # Detect frozen/stuck samples (>= 5 consecutive identical non-zero/finite readings)
    stuck_gyro = np.zeros(n, dtype=bool)
    stuck_accel = np.zeros(n, dtype=bool)
    if n >= 5:
        same_g = (
            w_in_range[1:]
            & w_in_range[:-1]
            & (gyro[1:, 0] == gyro[:-1, 0])
            & (gyro[1:, 1] == gyro[:-1, 1])
            & (gyro[1:, 2] == gyro[:-1, 2])
        )
        run_g = same_g[:-3] & same_g[1:-2] & same_g[2:-1] & same_g[3:]
        stuck_gyro[4:][run_g] = True
        del same_g, run_g

        same_a = (
            a_in_range[1:]
            & a_in_range[:-1]
            & (accel[1:, 0] == accel[:-1, 0])
            & (accel[1:, 1] == accel[:-1, 1])
            & (accel[1:, 2] == accel[:-1, 2])
        )
        run_a = same_a[:-3] & same_a[1:-2] & same_a[2:-1] & same_a[3:]
        stuck_accel[4:][run_a] = True
        del same_a, run_a

    # Detect sudden single-sample impulse spikes relative to previous valid sample
    spike_gyro = np.zeros(n, dtype=bool)
    spike_accel = np.zeros(n, dtype=bool)
    if n >= 2:
        both_g = w_in_range[1:] & w_in_range[:-1]
        dg = gyro[1:] - gyro[:-1]
        dg_norm = np.sqrt(np.sum(dg * dg, axis=1))
        spike_gyro[1:][both_g & (dg_norm > 10.0)] = True
        del both_g, dg, dg_norm

        both_a = a_in_range[1:] & a_in_range[:-1]
        da = accel[1:] - accel[:-1]
        da_norm = np.sqrt(np.sum(da * da, axis=1))
        spike_accel[1:][both_a & (da_norm > 60.0)] = True
        del both_a, da, da_norm

    valid_gyro = w_in_range & (~terminal_failure) & (~stuck_gyro) & (~spike_gyro)
    valid_accel = a_in_range & (~terminal_failure) & (~stuck_accel) & (~spike_accel)
    del w_finite, a_finite, w_in_range, a_in_range, terminal_failure, stuck_gyro, stuck_accel, spike_gyro, spike_accel

    gyro[~valid_gyro] = 0.0
    accel[~valid_accel] = 0.0
    return valid_gyro, valid_accel


def run_pipeline(
    input_path: str,
    output_path: str = "outputs/estimated_trajectory.txt",
    metadata_output: Optional[str] = None,
    log_level: str = "INFO",
    enable_zupt: bool = False,
) -> Dict[str, Any]:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not tracemalloc.is_tracing():
        tracemalloc.start()
    t_start = time.perf_counter()

    logger.info("================================================================================")
    logger.info("Samsung XR Hackathon 2026 - Track 3 Autonomous Dead-Reckoning Estimator")
    logger.info("================================================================================")
    logger.info("Input IMU file: %s", input_path)
    logger.info("Output TUM file: %s", output_path)

    ts_ns, gyro_rates, a_body = _load_imu_arrays_lean(input_path)
    n_input_frames = len(ts_ns)
    logger.info("Loaded %d IMU frames from %s", n_input_frames, input_path)

    valid_gyro_mask, valid_accel_mask = _sanitize_arrays_inplace(gyro_rates, a_body)
    n_invalid_gyro = int((~valid_gyro_mask).sum())
    n_invalid_accel = int((~valid_accel_mask).sum())
    n_corrupted = int((~(valid_gyro_mask & valid_accel_mask)).sum())
    if n_corrupted > 0:
        logger.warning(
            "Sanitizer flagged %d corrupted rows (Invalid gyro: %d, Invalid accel: %d). "
            "Isolated channels zeroed; row count preserved.",
            n_corrupted,
            n_invalid_gyro,
            n_invalid_accel,
        )

    MICRO_GAP_THRESHOLD_S: float = 0.025

    df_bias_view = pd.DataFrame({
        IMU_TIMESTAMP_COL: ts_ns,
        GYRO_COLS[0]: gyro_rates[:, 0],
        GYRO_COLS[1]: gyro_rates[:, 1],
        GYRO_COLS[2]: gyro_rates[:, 2],
        ACCEL_COLS[0]: a_body[:, 0],
        ACCEL_COLS[1]: a_body[:, 1],
        ACCEL_COLS[2]: a_body[:, 2],
    }, copy=False)
    bias_result = estimate_gyro_bias(df_bias_view, max_allowed_gap_s=MICRO_GAP_THRESHOLD_S)
    del df_bias_view

    gyro_bias = bias_result["gyro_bias"]
    interval_info = bias_result.get("interval_info") or {}
    logger.info(
        "Stationary Gyro Bias Estimation: [%.6f, %.6f, %.6f] rad/s (Confidence: %s, Status: %s, Duration: %.2fs)",
        gyro_bias[0],
        gyro_bias[1],
        gyro_bias[2],
        bias_result["confidence"],
        bias_result["status"],
        interval_info.get("duration_s", 0.0),
    )

    gyro_rates[valid_gyro_mask] -= gyro_bias

    n_samples = len(ts_ns)
    dt_s = np.zeros(n_samples, dtype=np.float64)
    if n_samples > 1:
        dt_s[1:] = (ts_ns[1:] - ts_ns[:-1]).astype(np.float64) * 1e-9

    def _rotvec_to_dq_wxyz(rv: np.ndarray) -> np.ndarray:
        angle = math.sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2])
        if angle < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        half = 0.5 * angle
        s = math.sin(half) / angle
        return np.array([math.cos(half), rv[0] * s, rv[1] * s, rv[2] * s], dtype=np.float64)

    def _qmul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)

    def _qrot_vec_arr(q_arr: np.ndarray, v_arr: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q_arr[:, 0], q_arr[:, 1], q_arr[:, 2], q_arr[:, 3]
        vx, vy, vz = v_arr[:, 0], v_arr[:, 1], v_arr[:, 2]
        out = np.empty_like(v_arr)
        out[:, 0] = (1.0 - 2.0 * (qy * qy + qz * qz)) * vx + 2.0 * (qx * qy - qw * qz) * vy + 2.0 * (qx * qz + qw * qy) * vz
        out[:, 1] = 2.0 * (qx * qy + qw * qz) * vx + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy + 2.0 * (qy * qz - qw * qx) * vz
        out[:, 2] = 2.0 * (qx * qz - qw * qy) * vx + 2.0 * (qy * qz + qw * qx) * vy + (1.0 - 2.0 * (qx * qx + qy * qy)) * vz
        return out

    q_rel = np.empty((n_samples, 4), dtype=np.float64)
    qw_out = q_rel[:, 0]
    qx_out = q_rel[:, 1]
    qy_out = q_rel[:, 2]
    qz_out = q_rel[:, 3]
    qw_out[0] = 1.0
    qx_out[0] = 0.0
    qy_out[0] = 0.0
    qz_out[0] = 0.0

    mv_dt_o = memoryview(dt_s)
    mv_gyro = memoryview(gyro_rates)
    mv_vg = memoryview(valid_gyro_mask)
    gap_thresh = MICRO_GAP_THRESHOLD_S
    _sin = math.sin
    _cos = math.cos
    _sqrt = math.sqrt

    cq_w, cq_x, cq_y, cq_z = 1.0, 0.0, 0.0, 0.0
    for i in range(1, n_samples):
        dt = mv_dt_o[i]
        if dt <= gap_thresh and mv_vg[i - 1] and mv_vg[i]:
            half_dt = 0.5 * dt
            rx = (mv_gyro[i - 1, 0] + mv_gyro[i, 0]) * half_dt
            ry = (mv_gyro[i - 1, 1] + mv_gyro[i, 1]) * half_dt
            rz = (mv_gyro[i - 1, 2] + mv_gyro[i, 2]) * half_dt
            ang = _sqrt(rx * rx + ry * ry + rz * rz)
            if ang < 1e-12:
                dw, dx, dy, dz = 1.0, 0.0, 0.0, 0.0
            else:
                half_a = 0.5 * ang
                s = _sin(half_a) / ang
                dw = _cos(half_a)
                dx = rx * s
                dy = ry * s
                dz = rz * s
            nw = cq_w * dw - cq_x * dx - cq_y * dy - cq_z * dz
            nx = cq_w * dx + cq_x * dw + cq_y * dz - cq_z * dy
            ny = cq_w * dy - cq_x * dz + cq_y * dw + cq_z * dx
            nz = cq_w * dz + cq_x * dy - cq_y * dx + cq_z * dw
            inv_n = 1.0 / _sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
            cq_w = nw * inv_n
            cq_x = nx * inv_n
            cq_y = ny * inv_n
            cq_z = nz * inv_n
        qw_out[i] = cq_w
        qx_out[i] = cq_x
        qy_out[i] = cq_y
        qz_out[i] = cq_z
    mv_dt_o.release(); mv_gyro.release(); mv_vg.release()
    del mv_dt_o, mv_gyro, mv_vg, qw_out, qx_out, qy_out, qz_out

    aligned_from_stationary = False
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
            q_s = q_rel[stat_mask]
            v_s = a_body[stat_mask]
            qw_s, qx_s, qy_s, qz_s = q_s[:, 0], q_s[:, 1], q_s[:, 2], q_s[:, 3]
            vx_s, vy_s, vz_s = v_s[:, 0], v_s[:, 1], v_s[:, 2]
            rx_s = (1.0 - 2.0 * (qy_s * qy_s + qz_s * qz_s)) * vx_s + 2.0 * (qx_s * qy_s - qw_s * qz_s) * vy_s + 2.0 * (qx_s * qz_s + qw_s * qy_s) * vz_s
            ry_s = 2.0 * (qx_s * qy_s + qw_s * qz_s) * vx_s + (1.0 - 2.0 * (qx_s * qx_s + qz_s * qz_s)) * vy_s + 2.0 * (qy_s * qz_s - qw_s * qx_s) * vz_s
            rz_s = 2.0 * (qx_s * qz_s - qw_s * qy_s) * vx_s + 2.0 * (qy_s * qz_s + qw_s * qx_s) * vy_s + (1.0 - 2.0 * (qx_s * qx_s + qy_s * qy_s)) * vz_s
            g_est_rel = np.array([float(np.mean(rx_s)), float(np.mean(ry_s)), float(np.mean(rz_s))], dtype=np.float64)
            del q_s, v_s, qw_s, qx_s, qy_s, qz_s, vx_s, vy_s, vz_s, rx_s, ry_s, rz_s
            if not np.isnan(g_est_rel).any() and np.linalg.norm(g_est_rel) >= 1.0:
                q0 = compute_gravity_alignment_quaternion(g_est_rel, yaw_rad=0.0)
                aligned_from_stationary = True
        del stat_mask

    if not aligned_from_stationary:
        valid_accel_idx = np.flatnonzero(valid_accel_mask)[:100]
        if len(valid_accel_idx) == 0:
            logger.warning("Initial accelerometer invalid or near zero. Defaulting to identity orientation.")
            q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            initial_accel_sample = a_body[valid_accel_idx].mean(axis=0)
            if np.isnan(initial_accel_sample).any() or np.linalg.norm(initial_accel_sample) < 1.0:
                logger.warning("Initial accelerometer invalid or near zero. Defaulting to identity orientation.")
                q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                q0 = compute_gravity_alignment_quaternion(initial_accel_sample, yaw_rad=0.0)
        del valid_accel_idx

    w1, x1, y1, z1 = float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])
    w2 = q_rel[:, 0].copy()
    x2 = q_rel[:, 1].copy()
    y2 = q_rel[:, 2].copy()
    z2 = q_rel[:, 3].copy()
    quats_wxyz = q_rel
    quats_wxyz[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    quats_wxyz[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    quats_wxyz[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    quats_wxyz[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    del w2, x2, y2, z2
    q_norms = np.sqrt(np.sum(quats_wxyz * quats_wxyz, axis=1, keepdims=True))
    q_norms[q_norms == 0.0] = 1.0
    quats_wxyz /= q_norms
    del q_norms
    logger.info("Initial Attitude Quaternion [w, x, y, z]: [%.6f, %.6f, %.6f, %.6f]", q0[0], q0[1], q0[2], q0[3])

    zupt_mask = np.zeros(n_samples, dtype=bool)
    if enable_zupt:
        for s_i, e_i, _, _, _ in bias_result.get("detected_intervals", []):
            zupt_mask[s_i:e_i + 1] = True

    del gyro_rates, valid_gyro_mask

    w_q = quats_wxyz[:, 0]
    x_q = quats_wxyz[:, 1]
    y_q = quats_wxyz[:, 2]
    z_q = quats_wxyz[:, 3]
    vx = a_body[:, 0].copy()
    vy = a_body[:, 1].copy()
    vz = a_body[:, 2].copy()

    a_body[:, 0] = (1.0 - 2.0 * (y_q * y_q + z_q * z_q)) * vx + 2.0 * (x_q * y_q - w_q * z_q) * vy + 2.0 * (x_q * z_q + w_q * y_q) * vz
    a_body[:, 1] = 2.0 * (x_q * y_q + w_q * z_q) * vx + (1.0 - 2.0 * (x_q * x_q + z_q * z_q)) * vy + 2.0 * (y_q * z_q - w_q * x_q) * vz
    a_body[:, 2] = 2.0 * (x_q * z_q - w_q * y_q) * vx + 2.0 * (y_q * z_q + w_q * x_q) * vy + (1.0 - 2.0 * (x_q * x_q + y_q * y_q)) * vz - STANDARD_GRAVITY
    del vx, vy, vz
    a_body[~valid_accel_mask] = 0.0

    pos_arr = np.zeros((n_samples, 3), dtype=np.float64)
    vel_arr = np.zeros((n_samples, 3), dtype=np.float64)
    px_out = pos_arr[:, 0]
    py_out = pos_arr[:, 1]
    pz_out = pos_arr[:, 2]
    vx_out = vel_arr[:, 0]
    vy_out = vel_arr[:, 1]
    vz_out = vel_arr[:, 2]

    mv_dt_p = memoryview(dt_s)
    mv_acc = memoryview(a_body)
    mv_va = memoryview(valid_accel_mask)
    mv_zp = memoryview(zupt_mask)
    use_zupt = bool(enable_zupt)
    gap_p = MICRO_GAP_THRESHOLD_S
    v_max_val = float(V_MAX)
    decay_rate_val = float(OUTAGE_DECAY_RATE)
    _exp = math.exp
    _sqrt = math.sqrt

    px, py, pz = 0.0, 0.0, 0.0
    vx, vy, vz = 0.0, 0.0, 0.0
    for i in range(1, n_samples):
        dt = mv_dt_p[i]
        if dt == 0.0:
            pass  # Duplicate timestamp (dt == 0): no state change and no decay
        elif dt > gap_p or not (mv_va[i - 1] and mv_va[i]):
            # Outage / gap / invalid sensor reading: exponential velocity decay across interval
            decay_factor = _exp(-decay_rate_val * dt)
            vx *= decay_factor
            vy *= decay_factor
            vz *= decay_factor
            px += vx * dt
            py += vy * dt
            pz += vz * dt
        else:
            half_dt = 0.5 * dt
            vnx = vx + (mv_acc[i - 1, 0] + mv_acc[i, 0]) * half_dt
            vny = vy + (mv_acc[i - 1, 1] + mv_acc[i, 1]) * half_dt
            vnz = vz + (mv_acc[i - 1, 2] + mv_acc[i, 2]) * half_dt
            if use_zupt and mv_zp[i]:
                vnx = 0.0
                vny = 0.0
                vnz = 0.0

            # Velocity cap
            v_norm = _sqrt(vnx * vnx + vny * vny + vnz * vnz)
            if v_norm > v_max_val:
                scale = v_max_val / v_norm
                vnx *= scale
                vny *= scale
                vnz *= scale

            px += (vx + vnx) * half_dt
            py += (vy + vny) * half_dt
            pz += (vz + vnz) * half_dt
            vx, vy, vz = vnx, vny, vnz
        px_out[i] = px
        py_out[i] = py
        pz_out[i] = pz
        vx_out[i] = vx
        vy_out[i] = vy
        vz_out[i] = vz
    mv_dt_p.release(); mv_acc.release(); mv_va.release(); mv_zp.release()
    del mv_dt_p, mv_acc, mv_va, mv_zp, px_out, py_out, pz_out, vx_out, vy_out, vz_out

    del a_body, valid_accel_mask, dt_s, vel_arr

    export_tum_trajectory(
        output_path=output_path,
        timestamps_ns=ts_ns,
        pos_arr=pos_arr,
        quats_wxyz=quats_wxyz,
    )
    expected_rows = n_samples

    del pos_arr, quats_wxyz
    gc.collect()
    logger.info("Successfully exported TUM trajectory to: %s", output_path)

    val_result = validate_tum_file(output_path, expected_row_count=expected_rows)
    logger.info(
        "TUM Validation PASSED: %d rows, duration: %.9f s, max quaternion norm error: %.2e",
        val_result["row_count"],
        val_result["duration_s"],
        val_result["max_quat_norm_deviation"],
    )

    t_end = time.perf_counter()
    runtime_s = t_end - t_start

    if tracemalloc.is_tracing():
        _heap_current, peak_heap_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        peak_heap_bytes = 0
    peak_rss_bytes, rss_method, rss_valid = get_peak_process_memory_bytes()
    peak_heap_mb = peak_heap_bytes / (1024.0 * 1024.0)
    peak_rss_mb = peak_rss_bytes / (1024.0 * 1024.0)

    telemetry_duration_s = float(val_result["duration_s"])
    ms_per_second_telemetry = (runtime_s * 1000.0) / max(telemetry_duration_s, 1e-6)

    meets_python_heap_budget = peak_heap_mb <= 8.0
    meets_process_ram_budget = rss_valid and (peak_rss_mb <= 8.0)
    meets_compute_latency_budget = ms_per_second_telemetry <= 20.0

    active_threadpools = []
    actual_threads_single = True
    try:
        import threadpoolctl
        active_threadpools = threadpoolctl.threadpool_info()
        for pool in active_threadpools:
            if pool.get("num_threads", 1) > 1:
                actual_threads_single = False
    except Exception:
        pass

    env_threads_single = bool(
        os.environ.get("OMP_NUM_THREADS") == "1" and
        os.environ.get("OPENBLAS_NUM_THREADS") == "1" and
        os.environ.get("MKL_NUM_THREADS") == "1"
    )
    single_thread_enforced = env_threads_single and (actual_threads_single if active_threadpools else True)

    telemetry: Dict[str, Any] = {
        "pipeline_version": "Stage 4 Step 3",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_frames": n_input_frames,
        "output_tum_poses": expected_rows,
        "telemetry_duration_s": telemetry_duration_s,
        "execution_runtime_s": runtime_s,
        "compute_latency_ms_per_s": ms_per_second_telemetry,
        "python_heap_peak_mb": peak_heap_mb,
        "process_peak_rss_mb": peak_rss_mb if rss_valid else "UNAVAILABLE",
        "process_peak_rss_method": rss_method,
        "process_peak_rss_valid": rss_valid,
        "meets_python_heap_budget": meets_python_heap_budget,
        "meets_process_ram_budget": meets_process_ram_budget,
        "meets_compute_latency_budget": meets_compute_latency_budget,
        "single_thread_enforced": single_thread_enforced,
        "thread_verification": {
            "env_vars_single": env_threads_single,
            "active_threadpools": active_threadpools,
            "actual_threadpools_single": actual_threads_single,
        },
        "validation": val_result,
        "bias_estimation": {
            "gyro_bias_rad_s": gyro_bias.tolist(),
            "confidence": bias_result["confidence"],
            "status": bias_result["status"],
            "interval_info": interval_info,
            "micro_gap_threshold_s": MICRO_GAP_THRESHOLD_S,
        },
    }

    logger.info("--------------------------------------------------------------------------------")
    logger.info("PERFORMANCE & RESOURCE COMPLIANCE SUMMARY:")
    logger.info("  Runtime: %.3f s | Telemetry: %.3f s | Latency: %.2f ms/s (Budget: <= 20 ms/s) [%s]",
                runtime_s, telemetry_duration_s, ms_per_second_telemetry,
                "PASS" if meets_compute_latency_budget else "FAIL")
    logger.info("  Python Heap Peak: %.2f MB (Budget: <= 8.0 MB) [%s]",
                peak_heap_mb, "PASS" if meets_python_heap_budget else "FAIL")
    if rss_valid:
        logger.info("  Total Process Peak RSS: %.2f MB via %s (Budget: <= 8.0 MB) [%s]",
                    peak_rss_mb, rss_method,
                    "PASS" if meets_process_ram_budget else "FAIL (C-Runtime overhead)")
    else:
        logger.warning("  Total Process Peak RSS: UNAVAILABLE (%s) — budget compliance cannot be confirmed [FAIL]",
                       rss_method)
    logger.info("  Single Thread Enforced: %s", telemetry["single_thread_enforced"])
    logger.info("================================================================================")

    if metadata_output:
        meta_path = Path(metadata_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, cls=CustomJSONEncoder)
        logger.info("Execution metadata saved to: %s", meta_path)

    return telemetry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Samsung XR Hackathon 2026 - Track 3 Autonomous Dead-Reckoning Pipeline Entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to input IMU CSV file (EuRoC format).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/estimated_trajectory.txt",
        type=str,
        help="Destination path for TUM formatted estimated_trajectory.txt.",
    )
    parser.add_argument(
        "--metadata-output",
        "-m",
        default=None,
        type=str,
        help="Optional path to output execution metadata JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--zupt",
        action="store_true",
        help="Enable Zero-Velocity Update (ZUPT) during detected quasi-stationary intervals.",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            metadata_output=args.metadata_output,
            log_level=args.log_level,
            enable_zupt=args.zupt,
        )
    except Exception as e:
        logger.exception("Pipeline execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
