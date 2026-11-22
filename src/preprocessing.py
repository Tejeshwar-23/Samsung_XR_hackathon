"""
Preprocessing and sensor quality assessment module for EuRoC MAV IMU data.

Provides:
- Timestamp analysis: dt calculation, dt conversion to seconds, and timestamp gap detection.
- Sensor statistics: per-channel min, max, mean, std, and 3D vector magnitudes.
- Initial stationarity analysis: statistical assessment of the initial time window
  (default 2 seconds) to evaluate if the sensor is at rest for bias estimation.
- Bias estimation: computes gyro and accel bias estimates only if stationarity is confirmed.
- Gravity assessment: computes the mean specific force vector, its magnitude, and direction
  for verified stationary periods.
- Strict data safety: original measurements and timestamps remain unmodified.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


GYRO_COLS = ["w_RS_S_x", "w_RS_S_y", "w_RS_S_z"]
ACCEL_COLS = ["a_RS_S_x", "a_RS_S_y", "a_RS_S_z"]
SENSOR_COLS = GYRO_COLS + ACCEL_COLS


def compute_timestamp_intervals(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    gap_threshold_s: Optional[float] = None,
    nominal_dt_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate sampling intervals (dt) between consecutive timestamps.

    Args:
        df: DataFrame containing the timestamp column.
        timestamp_col: Name of the timestamp column (expected in int64 nanoseconds).
        gap_threshold_s: Custom threshold in seconds to flag unusually large intervals.
                         If None, defaults to 1.5 * nominal_dt (or 1.5 * median_dt).
        nominal_dt_s: Expected nominal sampling interval in seconds (e.g. 0.005 for 200 Hz).

    Returns:
        Dictionary containing:
            - 'dt_ns': pd.Series of differences in nanoseconds (int64)
            - 'dt_s': pd.Series of differences in seconds (float64)
            - 'mean_dt_s': float
            - 'median_dt_s': float
            - 'min_dt_s': float
            - 'max_dt_s': float
            - 'std_dt_s': float
            - 'estimated_freq_hz': float (1.0 / mean_dt_s)
            - 'gap_threshold_s': float
            - 'gap_indices': list of integer indices where dt_s > gap_threshold_s
            - 'num_gaps': int
    """
    if df.empty or len(df) < 2:
        raise ValueError("DataFrame must contain at least 2 samples to compute intervals.")

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    ts = df[timestamp_col].astype(np.int64)
    dt_ns = ts.diff().iloc[1:].astype(np.int64)
    dt_s = dt_ns * 1e-9

    mean_dt = float(dt_s.mean())
    median_dt = float(dt_s.median())
    min_dt = float(dt_s.min())
    max_dt = float(dt_s.max())
    std_dt = float(dt_s.std())
    est_freq = 1.0 / mean_dt if mean_dt > 0 else 0.0

    if gap_threshold_s is None:
        base_dt = nominal_dt_s if nominal_dt_s is not None else median_dt
        gap_threshold_s = 1.5 * base_dt

    gap_mask = dt_s > gap_threshold_s
    # Map back to row indices corresponding to the arrival of the delayed sample
    gap_indices = dt_s[gap_mask].index.tolist()

    return {
        "dt_ns": dt_ns,
        "dt_s": dt_s,
        "mean_dt_s": mean_dt,
        "median_dt_s": median_dt,
        "min_dt_s": min_dt,
        "max_dt_s": max_dt,
        "std_dt_s": std_dt,
        "estimated_freq_hz": est_freq,
        "gap_threshold_s": float(gap_threshold_s),
        "gap_indices": gap_indices,
        "num_gaps": len(gap_indices),
    }


def compute_vector_magnitudes(
    df: pd.DataFrame,
    gyro_cols: Optional[list[str]] = None,
    accel_cols: Optional[list[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Euclidean norms (magnitudes) for gyroscope and accelerometer vectors.

    Args:
        df: DataFrame containing sensor columns.
        gyro_cols: List of 3 gyroscope column names. Defaults to standard GYRO_COLS.
        accel_cols: List of 3 accelerometer column names. Defaults to standard ACCEL_COLS.

    Returns:
        Tuple (gyro_magnitude, accel_magnitude) as float64 1D NumPy arrays.
    """
    g_cols = gyro_cols or GYRO_COLS
    a_cols = accel_cols or ACCEL_COLS

    for col in g_cols + a_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    gyro_mat = df[g_cols].to_numpy(dtype=np.float64)
    accel_mat = df[a_cols].to_numpy(dtype=np.float64)

    gyro_mag = np.linalg.norm(gyro_mat, axis=1)
    accel_mag = np.linalg.norm(accel_mat, axis=1)

    return gyro_mag, accel_mag


def compute_sensor_statistics(
    df: pd.DataFrame,
    sensor_cols: Optional[list[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate minimum, maximum, mean, and standard deviation for sensor channels,
    plus overall vector magnitudes.

    Args:
        df: DataFrame containing IMU sensor data.
        sensor_cols: List of sensor columns. Defaults to SENSOR_COLS.

    Returns:
        Dictionary mapping column names (and 'gyro_magnitude', 'accel_magnitude')
        to their statistics: {'min', 'max', 'mean', 'std'}.
    """
    cols = sensor_cols or SENSOR_COLS
    stats = {}

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        vals = df[col].to_numpy(dtype=np.float64)
        stats[col] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
        }

    gyro_mag, accel_mag = compute_vector_magnitudes(df)
    stats["gyro_magnitude"] = {
        "min": float(np.min(gyro_mag)),
        "max": float(np.max(gyro_mag)),
        "mean": float(np.mean(gyro_mag)),
        "std": float(np.std(gyro_mag, ddof=1)),
    }
    stats["accel_magnitude"] = {
        "min": float(np.min(accel_mag)),
        "max": float(np.max(accel_mag)),
        "mean": float(np.mean(accel_mag)),
        "std": float(np.std(accel_mag, ddof=1)),
    }

    return stats


def analyze_initial_stationarity(
    df: pd.DataFrame,
    duration_s: float = 2.0,
    timestamp_col: str = "timestamp",
    gyro_mag_mean_thresh: float = 0.08,    # rad/s (~4.5 deg/s max permissible resting angular rate)
    gyro_mag_max_thresh: float = 0.20,     # rad/s peak rotation threshold
    accel_mag_std_thresh: float = 0.35,    # m/s^2 standard deviation threshold
    accel_mag_mean_min: float = 9.0,       # m/s^2 minimum expected specific force near 1g
    accel_mag_mean_max: float = 10.6,      # m/s^2 maximum expected specific force near 1g
) -> Dict[str, Any]:
    """
    Evaluate whether the initial segment (default 2 seconds) is stationary.

    A stationary IMU at rest:
      1. Has near-zero angular velocity (gyro magnitude mean and peak below threshold).
      2. Has stable specific force close to Earth gravity 1g (~9.81 m/s^2) with low variance.

    Args:
        df: DataFrame with IMU data.
        duration_s: Time window duration in seconds from the first sample.
        timestamp_col: Name of timestamp column (int64 nanoseconds).
        gyro_mag_mean_thresh: Maximum acceptable mean gyro magnitude in rad/s.
        gyro_mag_max_thresh: Maximum acceptable peak gyro magnitude in rad/s.
        accel_mag_std_thresh: Maximum acceptable standard deviation of accel magnitude in m/s^2.
        accel_mag_mean_min: Minimum acceptable mean accel magnitude in m/s^2.
        accel_mag_mean_max: Maximum acceptable mean accel magnitude in m/s^2.

    Returns:
        Dictionary with:
            - 'is_stationary': bool
            - 'duration_s': float
            - 'sample_count': int
            - 'evidence': dict of measured metrics vs. thresholds
            - 'reason': str explanation
            - 'stationary_slice': pd.DataFrame view of the initial window
    """
    if df.empty:
        raise ValueError("Cannot perform stationarity analysis on empty DataFrame.")

    t0 = df[timestamp_col].iloc[0]
    t_end = t0 + int(duration_s * 1e9)
    sub_df = df[df[timestamp_col] <= t_end].copy()

    if len(sub_df) < 5:
        return {
            "is_stationary": False,
            "duration_s": duration_s,
            "sample_count": len(sub_df),
            "evidence": {},
            "reason": "Insufficient samples in the requested initial window.",
            "stationary_slice": sub_df,
        }

    gyro_mag, accel_mag = compute_vector_magnitudes(sub_df)

    g_mean = float(np.mean(gyro_mag))
    g_max = float(np.max(gyro_mag))
    g_std = float(np.std(gyro_mag, ddof=1))

    a_mean = float(np.mean(accel_mag))
    a_std = float(np.std(accel_mag, ddof=1))
    a_min = float(np.min(accel_mag))
    a_max = float(np.max(accel_mag))

    check_gyro_mean = g_mean <= gyro_mag_mean_thresh
    check_gyro_max = g_max <= gyro_mag_max_thresh
    check_accel_std = a_std <= accel_mag_std_thresh
    check_accel_mean_range = accel_mag_mean_min <= a_mean <= accel_mag_mean_max

    is_stationary = bool(
        check_gyro_mean and check_gyro_max and check_accel_std and check_accel_mean_range
    )

    reasons = []
    if not check_gyro_mean:
        reasons.append(
            f"Mean gyro rate ({g_mean:.3f} rad/s) exceeds threshold ({gyro_mag_mean_thresh:.3f} rad/s)."
        )
    if not check_gyro_max:
        reasons.append(
            f"Max gyro rate ({g_max:.3f} rad/s) exceeds threshold ({gyro_mag_max_thresh:.3f} rad/s)."
        )
    if not check_accel_std:
        reasons.append(
            f"Accel magnitude standard deviation ({a_std:.3f} m/s^2) exceeds threshold ({accel_mag_std_thresh:.3f} m/s^2)."
        )
    if not check_accel_mean_range:
        reasons.append(
            f"Mean accel magnitude ({a_mean:.3f} m/s^2) outside static gravity window [{accel_mag_mean_min}, {accel_mag_mean_max}]."
        )

    reason_str = (
        "Initial segment meets all stationarity criteria (low rotation and stable 1g gravity norm)."
        if is_stationary
        else "Stationarity rejected: " + " ".join(reasons)
    )

    evidence = {
        "gyro_mag_mean": g_mean,
        "gyro_mag_mean_thresh": gyro_mag_mean_thresh,
        "gyro_mag_max": g_max,
        "gyro_mag_max_thresh": gyro_mag_max_thresh,
        "gyro_mag_std": g_std,
        "accel_mag_mean": a_mean,
        "accel_mag_std": a_std,
        "accel_mag_std_thresh": accel_mag_std_thresh,
        "accel_mag_min": a_min,
        "accel_mag_max": a_max,
        "checks": {
            "gyro_mean_passed": check_gyro_mean,
            "gyro_max_passed": check_gyro_max,
            "accel_std_passed": check_accel_std,
            "accel_mean_passed": check_accel_mean_range,
        },
    }

    return {
        "is_stationary": is_stationary,
        "duration_s": duration_s,
        "sample_count": len(sub_df),
        "evidence": evidence,
        "reason": reason_str,
        "stationary_slice": sub_df,
    }


def estimate_sensor_bias(
    df: pd.DataFrame,
    stationarity_result: Optional[Dict[str, Any]] = None,
    duration_s: float = 2.0,
    nominal_gravity: float = 9.80665,
) -> Dict[str, Any]:
    """
    Estimate initial sensor biases, but only if stationarity checks pass.

    For a stationary body:
      - Gyroscope true angular rate is 0 -> b_w = mean(w_meas).
      - Accelerometer specific force matches gravity: true specific force has norm g.
        The accelerometer residual bias estimate is relative to the measured gravity direction.

    Args:
        df: DataFrame of IMU measurements.
        stationarity_result: Precomputed stationarity analysis dictionary.
                             If None, will run analyze_initial_stationarity(df, duration_s).
        duration_s: Duration to check if stationarity_result is None.
        nominal_gravity: Standard Earth gravitational acceleration in m/s^2 (default 9.80665).

    Returns:
        Dictionary containing:
            - 'bias_estimated': bool (True if stationarity passed, False otherwise)
            - 'gyro_bias': np.ndarray of shape (3,) or None
            - 'accel_bias': np.ndarray of shape (3,) or None
            - 'reason': explanation
    """
    if stationarity_result is None:
        stationarity_result = analyze_initial_stationarity(df, duration_s=duration_s)

    if not stationarity_result["is_stationary"]:
        return {
            "bias_estimated": False,
            "gyro_bias": None,
            "accel_bias": None,
            "reason": (
                "Bias estimation aborted: Stationarity checks failed. "
                + stationarity_result["reason"]
            ),
        }

    sub_df = stationarity_result["stationary_slice"]

    # Compute mean gyroscope bias (since true angular rate is zero when stationary)
    gyro_bias = sub_df[GYRO_COLS].mean().to_numpy(dtype=np.float64)

    # For accelerometer, the measured specific force is f = a - g = -g in stationary state.
    # The magnitude deviation from nominal Earth gravity provides the scalar specific force bias.
    accel_mean = sub_df[ACCEL_COLS].mean().to_numpy(dtype=np.float64)
    measured_g_mag = float(np.linalg.norm(accel_mean))
    accel_unit_dir = accel_mean / measured_g_mag if measured_g_mag > 0 else np.zeros(3)
    # Estimated scalar bias along gravity direction
    accel_bias = (measured_g_mag - nominal_gravity) * accel_unit_dir

    return {
        "bias_estimated": True,
        "gyro_bias": gyro_bias,
        "accel_bias": accel_bias,
        "gyro_bias_dict": {col: float(val) for col, val in zip(GYRO_COLS, gyro_bias)},
        "accel_bias_dict": {col: float(val) for col, val in zip(ACCEL_COLS, accel_bias)},
        "sample_count": len(sub_df),
        "reason": "Stationary initial segment verified; bias successfully estimated.",
    }


def assess_gravity_vector(
    df: pd.DataFrame,
    stationarity_result: Optional[Dict[str, Any]] = None,
    duration_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Calculate the mean acceleration (specific force) vector for the verified stationary segment.
    Reports magnitude and unit direction vector without performing gravity compensation.

    Args:
        df: IMU measurements DataFrame.
        stationarity_result: Precomputed stationarity analysis result dictionary.
        duration_s: Window duration if stationarity_result is None.

    Returns:
        Dictionary containing:
            - 'gravity_assessed': bool
            - 'mean_accel_vector': np.ndarray of shape (3,) (in m/s^2)
            - 'magnitude': float (in m/s^2)
            - 'unit_direction': np.ndarray of shape (3,)
            - 'reason': explanation
    """
    if stationarity_result is None:
        stationarity_result = analyze_initial_stationarity(df, duration_s=duration_s)

    if not stationarity_result["is_stationary"]:
        # We can still compute the raw mean acceleration over the window, but flag that
        # it is not verified as pure stationary static gravity.
        sub_df = stationarity_result["stationary_slice"]
        mean_accel = sub_df[ACCEL_COLS].mean().to_numpy(dtype=np.float64)
        mag = float(np.linalg.norm(mean_accel))
        unit_dir = mean_accel / mag if mag > 0 else np.zeros(3)
        return {
            "gravity_assessed": False,
            "mean_accel_vector": mean_accel,
            "magnitude": mag,
            "unit_direction": unit_dir,
            "reason": (
                "Segment is not confirmed stationary; acceleration vector contains dynamical motion. "
                + stationarity_result["reason"]
            ),
        }

    sub_df = stationarity_result["stationary_slice"]
    mean_accel = sub_df[ACCEL_COLS].mean().to_numpy(dtype=np.float64)
    mag = float(np.linalg.norm(mean_accel))
    unit_dir = mean_accel / mag if mag > 0 else np.zeros(3)

    return {
        "gravity_assessed": True,
        "mean_accel_vector": mean_accel,
        "magnitude": mag,
        "unit_direction": unit_dir,
        "vector_dict": {col: float(val) for col, val in zip(ACCEL_COLS, mean_accel)},
        "direction_dict": {col: float(val) for col, val in zip(ACCEL_COLS, unit_dir)},
        "reason": "Stationary segment verified; specific force vector represents static gravity reaction.",
    }


def main() -> None:
    """Run preprocessing and stationarity analysis on EuRoC MH_01_easy dataset."""
    from pathlib import Path
    import sys

    # Import data loader
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import load_imu_data

    imu_path = project_root / "data/raw/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv"

    print("=" * 80)
    print("SAMSUNG XR HACKATHON: IMU PREPROCESSING & STATIONARITY ANALYSIS")
    print("=" * 80)
    print(f"Dataset path: {imu_path}\n")

    try:
        df = load_imu_data(imu_path)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Timestamp Analysis
    print("-" * 80)
    print("1. TIMESTAMP & INTERVAL ANALYSIS")
    print("-" * 80)
    intervals = compute_timestamp_intervals(df)
    print(f"Total samples:               {len(df):,}")
    print(f"Sampling dt mean:            {intervals['mean_dt_s'] * 1e3:.4f} ms")
    print(f"Sampling dt median:          {intervals['median_dt_s'] * 1e3:.4f} ms")
    print(f"Sampling dt range [min, max]: [{intervals['min_dt_s'] * 1e3:.4f} ms, {intervals['max_dt_s'] * 1e3:.4f} ms]")
    print(f"Estimated frequency:         {intervals['estimated_freq_hz']:.2f} Hz")
    print(f"Unusually large gaps (> {intervals['gap_threshold_s'] * 1e3:.2f} ms): {intervals['num_gaps']}")

    # 2. Sensor Statistics & Vector Magnitudes
    print("\n" + "-" * 80)
    print("2. SENSOR CHANNEL STATISTICS & VECTOR MAGNITUDES")
    print("-" * 80)
    stats = compute_sensor_statistics(df)
    header = f"{'Channel':<18} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std Dev':>10}"
    print(header)
    print("-" * len(header))
    for key, s in stats.items():
        print(f"{key:<18} | {s['min']:10.4f} | {s['max']:10.4f} | {s['mean']:10.4f} | {s['std']:10.4f}")

    # 3. Initial Stationarity Analysis
    print("\n" + "-" * 80)
    print("3. INITIAL STATIONARITY ANALYSIS (FIRST 2.0 SECONDS)")
    print("-" * 80)
    stationarity = analyze_initial_stationarity(df, duration_s=2.0)
    ev = stationarity["evidence"]
    print(f"Segment samples:             {stationarity['sample_count']}")
    print(f"Stationarity decision:       {'STATIONARY' if stationarity['is_stationary'] else 'NON-STATIONARY (DYNAMIC / IN-FLIGHT)'}")
    print(f"Reason:                      {stationarity['reason']}")
    print("\nEvidence metrics:")
    print(f"  - Gyroscope magnitude mean:    {ev['gyro_mag_mean']:.4f} rad/s (thresh <= {ev['gyro_mag_mean_thresh']:.4f}) -> {'PASS' if ev['checks']['gyro_mean_passed'] else 'FAIL'}")
    print(f"  - Gyroscope magnitude max:     {ev['gyro_mag_max']:.4f} rad/s (thresh <= {ev['gyro_mag_max_thresh']:.4f}) -> {'PASS' if ev['checks']['gyro_max_passed'] else 'FAIL'}")
    print(f"  - Accelerometer magnitude std: {ev['accel_mag_std']:.4f} m/s^2 (thresh <= {ev['accel_mag_std_thresh']:.4f}) -> {'PASS' if ev['checks']['accel_std_passed'] else 'FAIL'}")
    print(f"  - Accelerometer magnitude mean:{ev['accel_mag_mean']:.4f} m/s^2 (in static range [9.0, 10.6]) -> {'PASS' if ev['checks']['accel_mean_passed'] else 'FAIL'}")

    # 4. Bias Estimation
    print("\n" + "-" * 80)
    print("4. SENSOR BIAS ESTIMATION")
    print("-" * 80)
    bias_res = estimate_sensor_bias(df, stationarity_result=stationarity)
    print(f"Bias estimated:              {bias_res['bias_estimated']}")
    print(f"Explanation:                 {bias_res['reason']}")
    if bias_res["bias_estimated"]:
        print(f"Gyro bias [rad/s]:           {bias_res['gyro_bias']}")
        print(f"Accel bias residual [m/s^2]: {bias_res['accel_bias']}")

    # 5. Gravity Vector Assessment
    print("\n" + "-" * 80)
    print("5. GRAVITY VECTOR ASSESSMENT")
    print("-" * 80)
    grav_res = assess_gravity_vector(df, stationarity_result=stationarity)
    print(f"Gravity assessed as static:  {grav_res['gravity_assessed']}")
    print(f"Explanation:                 {grav_res['reason']}")
    print(f"Mean specific force [m/s^2]: [{grav_res['mean_accel_vector'][0]:.4f}, {grav_res['mean_accel_vector'][1]:.4f}, {grav_res['mean_accel_vector'][2]:.4f}]")
    print(f"Magnitude:                   {grav_res['magnitude']:.4f} m/s^2")
    print(f"Unit direction:              [{grav_res['unit_direction'][0]:.4f}, {grav_res['unit_direction'][1]:.4f}, {grav_res['unit_direction'][2]:.4f}]")

    print("\n" + "=" * 80)
    print("PREPROCESSING ANALYSIS COMPLETE (Original data remains untouched)")
    print("=" * 80)


if __name__ == "__main__":
    main()
