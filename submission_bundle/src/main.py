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
    from src.data_loader import IMU_TIMESTAMP_COL, _clean_header_name
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
    from data_loader import IMU_TIMESTAMP_COL, _clean_header_name
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
    cleaned_to_raw = {_clean_header_name(c): c for c in header_df.columns}
    del header_df

    for req in _CANONICAL_ORDER:
        if req not in cleaned_to_raw:
            raise ValueError(
                f"Missing required IMU column '{req}' in '{path}'. "
                f"Found cleaned headers: {list(cleaned_to_raw.keys())}"
            )

    usecols = [cleaned_to_raw[req] for req in _CANONICAL_ORDER]
    df_sub = pd.read_csv(path, usecols=usecols)
    if df_sub.empty:
        raise ValueError(f"IMU dataset at '{path}' is empty.")

    ts_col_raw = cleaned_to_raw[IMU_TIMESTAMP_COL]
    gyro_raw = [cleaned_to_raw[c] for c in GYRO_COLS]
    accel_raw = [cleaned_to_raw[c] for c in ACCEL_COLS]

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
    gyro_sat_thresh: float = GYRO_SATURATION_RAD_S,
    accel_sat_thresh: float = ACCEL_SATURATION_M_S2,
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

    valid_gyro = w_in_range & (~terminal_failure)
    valid_accel = a_in_range & (~terminal_failure)
    del w_finite, a_finite, w_in_range, a_in_range, terminal_failure

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
    logger.info("Initial Attitude Quaternion [w, x, y, z]: [%.6f, %.6f, %.6f, %.6f]", q0[0], q0[1], q0[2], q0[3])

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
            _w, _x, _y, _z = curr_q
            inv_norm = 1.0 / math.sqrt(_w * _w + _x * _x + _y * _y + _z * _z)
            curr_q[0] = _w * inv_norm
            curr_q[1] = _x * inv_norm
            curr_q[2] = _y * inv_norm
            curr_q[3] = _z * inv_norm

        quats_wxyz[i] = curr_q

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

    for i in range(1, n_samples):
        dt = dt_s[i]
        is_macro_gap = dt > MICRO_GAP_THRESHOLD_S
        both_accel_valid = bool(valid_accel_mask[i - 1]) and bool(valid_accel_mask[i])

        if is_macro_gap or not both_accel_valid:
            vel_arr[i] = vel_arr[i - 1]
            pos_arr[i] = pos_arr[i - 1] + vel_arr[i - 1] * dt
        else:
            v_next = vel_arr[i - 1] + 0.5 * (a_body[i - 1] + a_body[i]) * dt
            if enable_zupt and zupt_mask[i]:
                v_next = np.zeros(3, dtype=np.float64)
            pos_arr[i] = pos_arr[i - 1] + 0.5 * (vel_arr[i - 1] + v_next) * dt
            vel_arr[i] = v_next

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

    _heap_current, peak_heap_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
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
