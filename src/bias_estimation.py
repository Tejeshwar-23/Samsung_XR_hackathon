"""
Dedicated IMU-only gyroscope bias estimation module for Samsung XR Hackathon Track 3.

Implements an opportunistic Zero Angular Rate Update (ZARU) / quasi-stationary detector
using sliding-window statistics across raw accelerometer and gyroscope streams.

Key Principles:
1. IMU-Only:
   - Does NOT use ground-truth orientation, position, velocity, or bias.
   - Does NOT hardcode dataset timestamps or assume the start of recording is stationary.
2. Sliding-Window Multi-Criterion Stability:
   - Specific force magnitude mean close to nominal Earth gravity 1g (9.81 m/s^2).
   - Low variance of specific force magnitude (rejecting dynamic accelerations).
   - Low variance of angular velocity magnitude and per-axis rates (rejecting rotational acceleration).
   - High angular stability of the gravity vector direction (rejecting re-orienting maneuvers).
3. Minimum Duration Enforcement:
   - Requires at least min_continuous_duration_s (default 2.0 s) of continuous qualifying
     windows before estimating bias.
4. Documented Assumptions & Known Limitations:
   - A constant angular rate (especially around the gravity axis) has low variance and can mimic
     a constant bias. A low-variance detector assumes the vehicle is at rest rather than performing
     an un-accelerated steady rotation.
5. Safety:
   - If no interval meets the criteria, returns zero bias with bias_estimated=False and reason.
   - Never mutates the original DataFrame.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

try:
    from src.data_loader import IMU_TIMESTAMP_COL
    from src.preprocessing import GYRO_COLS, ACCEL_COLS
except ModuleNotFoundError:
    from data_loader import IMU_TIMESTAMP_COL
    from preprocessing import GYRO_COLS, ACCEL_COLS


def detect_quasi_stationary_intervals(
    df_imu: pd.DataFrame,
    window_duration_s: float = 1.0,
    window_step_s: float = 0.1,
    min_continuous_duration_s: float = 2.0,
    accel_mag_std_thresh: float = 0.25,      # m/s^2 max standard deviation of accel magnitude
    accel_mag_mean_diff_thresh: float = 0.35, # m/s^2 max absolute deviation from nominal 1g
    gyro_mag_std_thresh: float = 0.015,      # rad/s max standard deviation of gyro magnitude
    gyro_axis_std_thresh: float = 0.015,     # rad/s max standard deviation along any gyro axis
    accel_dir_std_thresh: float = 0.03,      # max std of unit gravity vector direction
    max_allowed_gap_s: Optional[float] = None, # max allowed timestamp difference between consecutive samples
    nominal_gravity: float = 9.80665,        # m/s^2 standard gravity
    time_col: str = IMU_TIMESTAMP_COL,
    gyro_cols: Optional[List[str]] = None,
    accel_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Scan IMU data with a sliding window to detect continuous quasi-stationary / low-motion intervals.

    Returns:
        Dictionary containing:
            - 'detected_intervals': List of tuples (start_idx, end_idx, start_time_ns, end_time_ns, duration_s)
            - 'qualifying_window_count': int
            - 'longest_interval': tuple or None
            - 'window_results': list of window statistics dicts
    """
    if df_imu.empty or len(df_imu) < 10:
        return {
            "detected_intervals": [],
            "qualifying_window_count": 0,
            "longest_interval": None,
            "window_results": [],
        }

    g_cols = gyro_cols or GYRO_COLS
    a_cols = accel_cols or ACCEL_COLS

    for col in [time_col] + g_cols + a_cols:
        if col not in df_imu.columns:
            raise ValueError(f"Required column '{col}' missing from IMU DataFrame.")

    ts = df_imu[time_col].astype(np.int64).to_numpy()
    w_mat = df_imu[g_cols].to_numpy(dtype=np.float64)
    a_mat = df_imu[a_cols].to_numpy(dtype=np.float64)

    a_mag = np.linalg.norm(a_mat, axis=1)
    w_mag = np.linalg.norm(w_mat, axis=1)

    dt_all_s = np.diff(ts).astype(np.float64) * 1e-9 if len(ts) > 1 else np.array([0.005])
    dt_median_s = float(np.median(dt_all_s)) if len(dt_all_s) > 0 else 0.005
    if max_allowed_gap_s is None:
        max_allowed_gap_s = 1.5 * dt_median_s

    window_samples = max(int(round(window_duration_s / dt_median_s)), 5)
    step_samples = max(int(round(window_step_s / dt_median_s)), 1)

    n_samples = len(ts)
    window_qualifies = []
    window_times = []
    window_indices = []

    for start_idx in range(0, n_samples - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples

        # 0. Timestamp continuity check: reject window if it contains an outage
        ts_sub = ts[start_idx:end_idx]
        dt_sub = np.diff(ts_sub).astype(np.float64) * 1e-9
        has_internal_gap = bool(np.any(dt_sub > max_allowed_gap_s))

        if has_internal_gap:
            window_qualifies.append(False)
            window_times.append((ts[start_idx], ts[end_idx - 1]))
            window_indices.append((start_idx, end_idx - 1))
            continue

        w_sub = w_mat[start_idx:end_idx]
        a_sub = a_mat[start_idx:end_idx]
        a_m_sub = a_mag[start_idx:end_idx]
        w_m_sub = w_mag[start_idx:end_idx]

        # 1. Accelerometer magnitude variance check
        std_a_m = float(np.std(a_m_sub))
        cond_a_std = std_a_m <= accel_mag_std_thresh

        # 2. Mean acceleration close to 1g check
        mean_a_m = float(np.mean(a_m_sub))
        cond_a_mean = abs(mean_a_m - nominal_gravity) <= accel_mag_mean_diff_thresh

        # 3. Gyroscope magnitude variance check
        std_w_m = float(np.std(w_m_sub))
        cond_w_std = std_w_m <= gyro_mag_std_thresh

        # 4. Per-axis gyroscope stability check
        std_w_axes = np.std(w_sub, axis=0)
        cond_w_axes = bool(np.all(std_w_axes <= gyro_axis_std_thresh))

        # 5. Gravity vector direction stability check (no attitude rocking)
        mean_a_vec = np.mean(a_sub, axis=0)
        norm_mean_a = np.linalg.norm(mean_a_vec)
        if norm_mean_a > 0:
            _norm_tmp = np.linalg.norm(a_sub, axis=1, keepdims=True)
            unit_a_sub = np.divide(a_sub, _norm_tmp, out=np.zeros_like(a_sub), where=_norm_tmp > 1e-12)
            std_dir = float(np.linalg.norm(np.std(unit_a_sub, axis=0)))
            cond_dir = std_dir <= accel_dir_std_thresh
        else:
            cond_dir = False

        qualifies = (
            cond_a_std and cond_a_mean and cond_w_std and cond_w_axes and cond_dir
        )

        window_qualifies.append(qualifies)
        window_times.append((ts[start_idx], ts[end_idx - 1]))
        window_indices.append((start_idx, end_idx - 1))

    # Cluster continuous qualifying windows without bridging timestamp gaps
    detected_intervals = []
    in_cluster = False
    cluster_start_w = 0

    for w_i, qualifies in enumerate(window_qualifies):
        if qualifies and not in_cluster:
            in_cluster = True
            cluster_start_w = w_i
        elif qualifies and in_cluster:
            # Check only the new step edge(s) added since the previous qualifying window
            prev_end_idx = window_indices[w_i - 1][1]
            curr_start_idx = window_indices[w_i][0]
            edge_start = max(cluster_start_i := window_indices[cluster_start_w][0], prev_end_idx)
            edge_end = window_indices[w_i][1] + 1
            if edge_end - edge_start > 1 and np.any(np.diff(ts[edge_start:edge_end]).astype(np.float64) * 1e-9 > max_allowed_gap_s):
                # Gap occurred between clusters; finalize previous cluster
                in_cluster = False
                cluster_end_w = w_i - 1
                start_i = window_indices[cluster_start_w][0]
                end_i = window_indices[cluster_end_w][1]
                t_s = ts[start_i]
                t_e = ts[end_i]
                dur_s = float((t_e - t_s) * 1e-9)
                if dur_s >= min_continuous_duration_s:
                    detected_intervals.append((start_i, end_i, t_s, t_e, dur_s))
                # Start new cluster from this window
                in_cluster = True
                cluster_start_w = w_i
        elif not qualifies and in_cluster:
            in_cluster = False
            cluster_end_w = w_i - 1
            start_i = window_indices[cluster_start_w][0]
            end_i = window_indices[cluster_end_w][1]
            t_s = ts[start_i]
            t_e = ts[end_i]
            dur_s = float((t_e - t_s) * 1e-9)
            if dur_s >= min_continuous_duration_s:
                detected_intervals.append((start_i, end_i, t_s, t_e, dur_s))

    # Check terminal cluster
    if in_cluster:
        cluster_end_w = len(window_qualifies) - 1
        start_i = window_indices[cluster_start_w][0]
        end_i = window_indices[cluster_end_w][1]
        t_s = ts[start_i]
        t_e = ts[end_i]
        dur_s = float((t_e - t_s) * 1e-9)
        if dur_s >= min_continuous_duration_s:
            detected_intervals.append((start_i, end_i, t_s, t_e, dur_s))

    longest_interval = None
    if detected_intervals:
        longest_interval = max(detected_intervals, key=lambda x: x[4])

    return {
        "detected_intervals": detected_intervals,
        "qualifying_window_count": sum(window_qualifies),
        "longest_interval": longest_interval,
        "total_windows_scanned": len(window_qualifies),
    }


def estimate_gyro_bias(
    df_imu: pd.DataFrame,
    window_duration_s: float = 1.0,
    window_step_s: float = 0.1,
    min_continuous_duration_s: float = 2.0,
    accel_mag_std_thresh: float = 0.25,
    accel_mag_mean_diff_thresh: float = 0.35,
    gyro_mag_std_thresh: float = 0.015,
    gyro_axis_std_thresh: float = 0.015,
    accel_dir_std_thresh: float = 0.03,
    max_allowed_gap_s: Optional[float] = None,
    time_col: str = IMU_TIMESTAMP_COL,
    gyro_cols: Optional[List[str]] = None,
    accel_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Estimate 3-axis gyroscope bias from the longest verified quasi-stationary interval.

    Args:
        df_imu: Raw IMU DataFrame. Original data is NOT modified.
        ... configuration thresholds ...

    Returns:
        Dictionary containing:
            - 'bias_estimated': bool
            - 'gyro_bias': np.ndarray of shape (3,) [b_x, b_y, b_z] in rad/s (or zeros if False)
            - 'gyro_bias_dict': dict mapping column names to bias values
            - 'status': str ('CONFIRMED_QUASI_STATIONARY' or 'NO_RELIABLE_INTERVAL')
            - 'confidence': str ('HIGH', 'MEDIUM', 'NONE')
            - 'interval_info': dict containing start/end timestamps, sample count, duration
            - 'diagnostics': dict containing stability standard deviations over the interval
            - 'reason': detailed text explanation
            - 'limitations_notice': explanation of constant-rate rotation ambiguity
    """
    detection = detect_quasi_stationary_intervals(
        df_imu=df_imu,
        window_duration_s=window_duration_s,
        window_step_s=window_step_s,
        min_continuous_duration_s=min_continuous_duration_s,
        accel_mag_std_thresh=accel_mag_std_thresh,
        accel_mag_mean_diff_thresh=accel_mag_mean_diff_thresh,
        gyro_mag_std_thresh=gyro_mag_std_thresh,
        gyro_axis_std_thresh=gyro_axis_std_thresh,
        accel_dir_std_thresh=accel_dir_std_thresh,
        max_allowed_gap_s=max_allowed_gap_s,
        time_col=time_col,
        gyro_cols=gyro_cols,
        accel_cols=accel_cols,
    )

    g_cols = gyro_cols or GYRO_COLS
    a_cols = accel_cols or ACCEL_COLS

    limitations_notice = (
        "THEORETICAL LIMITATION NOTICE: Sliding-window variance tests evaluate motion stability. "
        "A steady, constant-rate rotational maneuver (especially around the gravity axis) exhibits "
        "low angular rate variance and could theoretically mimic a constant gyroscope bias. "
        "This estimator assumes that sustained low-variance 1g intervals represent an at-rest or hover state."
    )

    if not detection["detected_intervals"]:
        return {
            "bias_estimated": False,
            "gyro_bias": np.zeros(3, dtype=np.float64),
            "gyro_bias_dict": {col: 0.0 for col in g_cols},
            "status": "NO_RELIABLE_INTERVAL",
            "confidence": "NONE",
            "interval_info": None,
            "diagnostics": None,
            "reason": (
                f"No continuous quasi-stationary segment >= {min_continuous_duration_s:.1f}s satisfied "
                f"the 1g magnitude (|a - 9.81| <= {accel_mag_mean_diff_thresh} m/s^2), accel variance "
                f"(std(a) <= {accel_mag_std_thresh} m/s^2), and gyro stability (std(w) <= {gyro_mag_std_thresh} rad/s) criteria."
            ),
            "limitations_notice": limitations_notice,
        }

    # Select the longest detected interval
    best_interval = detection["longest_interval"]
    start_i, end_i, t_start_ns, t_end_ns, duration_s = best_interval

    sub_imu = df_imu.iloc[start_i : end_i + 1]
    sample_count = len(sub_imu)

    # Compute bias as the mean measured angular rate over this verified interval
    mean_w = sub_imu[g_cols].mean().to_numpy(dtype=np.float64)
    std_w = sub_imu[g_cols].std().to_numpy(dtype=np.float64)

    mean_a = sub_imu[a_cols].mean().to_numpy(dtype=np.float64)
    std_a = sub_imu[a_cols].std().to_numpy(dtype=np.float64)
    a_mag_sub = np.linalg.norm(sub_imu[a_cols].to_numpy(dtype=np.float64), axis=1)

    # Determine confidence: HIGH if duration >= 4.0s, else MEDIUM
    confidence = "HIGH" if duration_s >= 4.0 else "MEDIUM"

    interval_info = {
        "start_index": start_i,
        "end_index": end_i,
        "start_timestamp_ns": t_start_ns,
        "end_timestamp_ns": t_end_ns,
        "duration_s": duration_s,
        "sample_count": sample_count,
        "total_candidate_intervals_found": len(detection["detected_intervals"]),
    }

    diagnostics = {
        "gyro_std_per_axis": {col: float(val) for col, val in zip(g_cols, std_w)},
        "gyro_mag_std": float(np.std(np.linalg.norm(sub_imu[g_cols].to_numpy(), axis=1))),
        "accel_mag_mean": float(np.mean(a_mag_sub)),
        "accel_mag_std": float(np.std(a_mag_sub)),
        "accel_std_per_axis": {col: float(val) for col, val in zip(a_cols, std_a)},
    }

    reason = (
        f"Identified stable quasi-stationary interval spanning {duration_s:.2f} s ({sample_count:,} samples) "
        f"with specific force {np.mean(a_mag_sub):.3f} +/- {np.std(a_mag_sub):.3f} m/s^2 and angular rate "
        f"stability std(w) = {np.mean(std_w):.5f} rad/s."
    )

    return {
        "bias_estimated": True,
        "gyro_bias": mean_w,
        "gyro_bias_dict": {col: float(val) for col, val in zip(g_cols, mean_w)},
        "status": "CONFIRMED_QUASI_STATIONARY",
        "confidence": confidence,
        "interval_info": interval_info,
        "diagnostics": diagnostics,
        "reason": reason,
        "limitations_notice": limitations_notice,
    }


def main() -> None:
    """Run IMU-only quasi-stationary detector and bias estimator on MH_01_easy."""
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import load_imu_data, load_groundtruth_data

    imu_path = project_root / "data/raw/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv"
    gt_path = (
        project_root
        / "data/raw/euroc/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    )

    print("=" * 80)
    print("SAMSUNG XR HACKATHON: IMU-ONLY GYRO BIAS ESTIMATION (MH_01_easy)")
    print("=" * 80)
    print(f"IMU file: {imu_path}\n")

    try:
        df_imu = load_imu_data(imu_path)
    except Exception as exc:
        print(f"Error loading IMU data: {exc}", file=sys.stderr)
        sys.exit(1)

    t0_ns = df_imu["timestamp"].iloc[0]

    # 1. Run Autonomous IMU-Only Bias Estimator
    print("-" * 80)
    print("1. AUTONOMOUS QUASI-STATIONARY DETECTION & BIAS ESTIMATION")
    print("-" * 80)
    res = estimate_gyro_bias(df_imu)

    print(f"Bias Estimated:          {res['bias_estimated']}")
    print(f"Status:                  {res['status']}")
    print(f"Confidence Level:        {res['confidence']}")
    print(f"Reason:                  {res['reason']}\n")

    b = res["gyro_bias"]
    if res["bias_estimated"]:
        info = res["interval_info"]
        t_start_s = (info["start_timestamp_ns"] - t0_ns) * 1e-9
        t_end_s = (info["end_timestamp_ns"] - t0_ns) * 1e-9

        print("Selected Quasi-Stationary Interval:")
        print(f"  - Relative Time Window: {t_start_s:6.2f} s to {t_end_s:6.2f} s (Duration: {info['duration_s']:.2f} s)")
        print(f"  - Row Index Range:      {info['start_index']:,} to {info['end_index']:,} ({info['sample_count']:,} samples)")
        print(f"  - Total Intervals Met:  {info['total_candidate_intervals_found']}")

        print("\nEstimated Gyroscope Biases (rad/s & deg/s):")
        print(f"  - b_x: {b[0]:+10.6f} rad/s ({np.rad2deg(b[0]):+7.3f} deg/s)")
        print(f"  - b_y: {b[1]:+10.6f} rad/s ({np.rad2deg(b[1]):+7.3f} deg/s)")
        print(f"  - b_z: {b[2]:+10.6f} rad/s ({np.rad2deg(b[2]):+7.3f} deg/s)")

        diag = res["diagnostics"]
        print("\nInterval Stability Diagnostics:")
        print(f"  - Mean Accel Norm:     {diag['accel_mag_mean']:.4f} m/s^2 (close to 9.807 m/s^2)")
        print(f"  - Accel Norm Std Dev:  {diag['accel_mag_std']:.4f} m/s^2 (threshold <= 0.25)")
        print(f"  - Gyro Mag Std Dev:    {diag['gyro_mag_std']:.6f} rad/s (threshold <= 0.015)")
        print(f"  - Gyro Axis Std Dev:   x={diag['gyro_std_per_axis']['w_RS_S_x']:.6f}, y={diag['gyro_std_per_axis']['w_RS_S_y']:.6f}, z={diag['gyro_std_per_axis']['w_RS_S_z']:.6f}")

    print("\n" + "-" * 80)
    print("2. THEORETICAL ASSUMPTIONS & FAILURE MODES")
    print("-" * 80)
    print(res["limitations_notice"])

    # 3. Offline Diagnostic Benchmark Against Ground Truth
    print("\n" + "=" * 80)
    print("OFFLINE DIAGNOSTIC VALIDATION (EVALUATION-ONLY BENCHMARK)")
    print("Note: Ground truth is used strictly for offline verification and is NOT fed")
    print("      into the autonomous estimator above.")
    print("=" * 80)
    try:
        df_gt = load_groundtruth_data(gt_path)
        gt_bias_x = float(df_gt["b_w_RS_S_x"].mean())
        gt_bias_y = float(df_gt["b_w_RS_S_y"].mean())
        gt_bias_z = float(df_gt["b_w_RS_S_z"].mean())
        gt_b = np.array([gt_bias_x, gt_bias_y, gt_bias_z])

        print(f"Ground Truth Mean Bias:   [{gt_b[0]:+10.6f}, {gt_b[1]:+10.6f}, {gt_b[2]:+10.6f}] rad/s")
        if res["bias_estimated"]:
            diff = b - gt_b
            print(f"Autonomous Estimate:      [{b[0]:+10.6f}, {b[1]:+10.6f}, {b[2]:+10.6f}] rad/s")
            print(f"Estimation Absolute Error:[{abs(diff[0]):10.6f}, {abs(diff[1]):10.6f}, {abs(diff[2]):10.6f}] rad/s")
            print(f"Error in deg/s:           [{np.rad2deg(abs(diff[0])):7.4f}, {np.rad2deg(abs(diff[1])):7.4f}, {np.rad2deg(abs(diff[2])):7.4f}] deg/s")
            print(f"Total Bias Vector Error:   {np.linalg.norm(diff):.6f} rad/s ({np.rad2deg(np.linalg.norm(diff)):.4f} deg/s)")
    except Exception as exc:
        print(f"Ground truth not available for diagnostic: {exc}")


if __name__ == "__main__":
    main()
