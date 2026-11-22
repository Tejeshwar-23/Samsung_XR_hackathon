"""
Sensor Degradation and Measurement Sanitization Module for Samsung XR Hackathon Track 3.

IMPORTANT NOTICE & PROVENANCE:
All synthetic degradation models implemented in this module are explicitly marked:
    PROVISIONAL DEVELOPMENT TESTING - NOT OFFICIAL ORGANIZER DEGRADATION
These provisional scenarios serve strictly for local stress-testing and regression validation.
When official organizer degradation scripts/assets are provided, they will be ingested directly.

Provides:
1. apply_provisional_degradation():
   - Implements deterministic synthetic sensor degradation across 8 scenarios (S4-01 to S4-08).
   - Uses exact discrete sampled Gaussian standard deviations (not continuous noise density).
   - Preserves caller's input DataFrame immutably.
   - Preserves exact int64 nanosecond timestamps and monotonic ordering.
   - Distinguishes missing frames (dropped rows) from invalid in-stream measurements.
   - Returns a comprehensive provenance metadata dictionary.

2. sanitize_imu_measurements():
   - Separate sanitizer detecting NaN, Inf, and physical sensor saturation:
     * Gyroscope norm > 35.0 rad/s (~2000 deg/s ADIS16448 range).
     * Accelerometer norm > 180.0 m/s^2 (~18g ADIS16448 range).
   - Marks corrupted rows explicitly with boolean 'is_corrupted' column.
   - NEVER forward-fills stale sensor readings.
   - Prepares sanitized feeds for downstream integration without modifying existing estimator modules.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

try:
    from src.data_loader import IMU_TIMESTAMP_COL
    from src.preprocessing import GYRO_COLS, ACCEL_COLS
except ModuleNotFoundError:
    from data_loader import IMU_TIMESTAMP_COL
    from preprocessing import GYRO_COLS, ACCEL_COLS


PROVENANCE_TAG = "PROVISIONAL DEVELOPMENT TESTING - NOT OFFICIAL ORGANIZER DEGRADATION"

GYRO_SATURATION_RAD_S = 35.0       # rad/s (~2005 deg/s, ADIS16448 full scale range)
ACCEL_SATURATION_M_S2 = 180.0      # m/s^2 (~18.35 g, ADIS16448 full scale range)

# Baseline discrete sampled Gaussian noise standard deviations (sample rate ~200 Hz)
BASELINE_GYRO_SAMPLED_STD_RAD_S = 0.005   # rad/s per sample
BASELINE_ACCEL_SAMPLED_STD_M_S2 = 0.05    # m/s^2 per sample


def sanitize_imu_measurements(
    df_imu: pd.DataFrame,
    gyro_sat_thresh: float = GYRO_SATURATION_RAD_S,
    accel_sat_thresh: float = ACCEL_SATURATION_M_S2,
    time_col: str = IMU_TIMESTAMP_COL,
    gyro_cols: Optional[List[str]] = None,
    accel_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Sanitizes raw/degraded IMU telemetry by identifying and isolating corrupted measurements.

    Checks:
    - Timestamp validity: Must be finite float/int; non-finite timestamps are marked corrupted.
    - Non-finite sensor values: NaN, +Inf, -Inf in gyro or accel axes.
    - Physical sensor saturation: norm(gyro) > gyro_sat_thresh or norm(accel) > accel_sat_thresh.

    Policy:
    - NEVER forward-fills stale sensor measurements (prevents sustained false motion injection).
    - Preserves exact row count (1:1 frame mapping).
    - Preserves original sensor readings in dedicated columns:
      ['w_raw_x', 'w_raw_y', 'w_raw_z', 'a_raw_x', 'a_raw_y', 'a_raw_z'].
    - Adds independent channel-level validity flags:
      * 'valid_gyro' (bool): True if gyro readings are finite and within physical range.
      * 'valid_accel' (bool): True if accel readings are finite and within physical range.
      * 'valid_timestamp' (bool): True if timestamp is finite.
      * 'is_corrupted' (bool): True if any channel is invalid.
    - Placeholder Policy for Estimator-Facing Columns:
      * If valid_gyro is False, estimator-facing gyro columns are set to 0.0 (placeholder).
      * If valid_accel is False, estimator-facing accel columns are set to 0.0 (placeholder).
      IMPORTANT: These zeroed values are placeholders to prevent numerical crashes; downstream
      orchestration MUST inspect 'valid_accel' and 'valid_gyro' to hold velocity and orientation,
      rather than interpreting raw 0.0 as zero specific force or zero angular rate.

    Parameters
    ----------
    df_imu : pd.DataFrame
        Input IMU DataFrame. Left unmodified.
    gyro_sat_thresh : float
        Saturation threshold for angular rate norm (rad/s).
    accel_sat_thresh : float
        Saturation threshold for linear acceleration norm (m/s^2).
    time_col : str
        Timestamp column name.
    gyro_cols : List[str], optional
        List of 3 gyro column names.
    accel_cols : List[str], optional
        List of 3 accel column names.

    Returns
    -------
    df_sanitized : pd.DataFrame
        New DataFrame with raw measurements preserved, independent validity flags added,
        and sanitized placeholders set in estimator-facing columns.
    diagnostics : Dict[str, Any]
        Sanitization metrics and corruption details.
    """
    if df_imu.empty:
        raise ValueError("Cannot sanitize empty IMU DataFrame.")

    g_cols = gyro_cols or GYRO_COLS
    a_cols = accel_cols or ACCEL_COLS

    for col in [time_col] + g_cols + a_cols:
        if col not in df_imu.columns:
            raise ValueError(f"Required column '{col}' missing from IMU DataFrame.")

    # Never mutate caller's DataFrame
    df_out = df_imu.copy()

    ts = df_out[time_col].to_numpy()
    w_mat = df_out[g_cols].to_numpy(dtype=np.float64)
    a_mat = df_out[a_cols].to_numpy(dtype=np.float64)

    # 1. Non-finite checks
    ts_valid = np.isfinite(ts)
    w_finite = np.isfinite(w_mat).all(axis=1)
    a_finite = np.isfinite(a_mat).all(axis=1)

    # 2. Saturation checks (compute norms safely on finite entries)
    w_norm = np.zeros(len(df_out), dtype=np.float64)
    a_norm = np.zeros(len(df_out), dtype=np.float64)

    w_norm[w_finite] = np.linalg.norm(w_mat[w_finite], axis=1)
    a_norm[a_finite] = np.linalg.norm(a_mat[a_finite], axis=1)

    w_in_range = w_finite & (w_norm <= gyro_sat_thresh)
    a_in_range = a_finite & (a_norm <= accel_sat_thresh)

    # 3. Contiguous terminal blackout detection:
    # Physical sensors on Earth continuously experience reaction force from gravity (norm ~ 9.8 m/s^2).
    # A simultaneous 6-axis zero (norm(a) < 1e-5 and norm(w) < 1e-6) extending contiguously to the final
    # recording frame represents a terminal sensor failure / power loss / blackout.
    # Isolated zero-crossings during flight do not extend to the end of the recording.
    is_six_axis_zero = (w_finite & a_finite & (w_norm < 1e-6) & (a_norm < 1e-5))
    terminal_failure = np.zeros(len(df_out), dtype=bool)
    if is_six_axis_zero[-1]:
        # Backward scan from final row
        k = len(df_out) - 1
        while k >= 0 and is_six_axis_zero[k]:
            terminal_failure[k] = True
            k -= 1

    # Independent validity flags
    valid_gyro = ts_valid & w_in_range & (~terminal_failure)
    valid_accel = ts_valid & a_in_range & (~terminal_failure)
    valid_timestamp = ts_valid

    # Combined corruption mask
    is_corrupted = (~valid_gyro) | (~valid_accel) | (~valid_timestamp)

    # Preserve raw values in dedicated columns
    df_out["w_raw_x"] = df_out[g_cols[0]].copy()
    df_out["w_raw_y"] = df_out[g_cols[1]].copy()
    df_out["w_raw_z"] = df_out[g_cols[2]].copy()
    df_out["a_raw_x"] = df_out[a_cols[0]].copy()
    df_out["a_raw_y"] = df_out[a_cols[1]].copy()
    df_out["a_raw_z"] = df_out[a_cols[2]].copy()

    # Assign validity flags
    df_out["valid_timestamp"] = valid_timestamp
    df_out["valid_gyro"] = valid_gyro
    df_out["valid_accel"] = valid_accel
    df_out["is_corrupted"] = is_corrupted

    # For invalid entries, set placeholder 0.0 in estimator-facing columns
    # Downstream orchestration must check valid_gyro / valid_accel
    invalid_gyro_mask = ~valid_gyro
    if invalid_gyro_mask.any():
        df_out.loc[invalid_gyro_mask, g_cols[0]] = 0.0
        df_out.loc[invalid_gyro_mask, g_cols[1]] = 0.0
        df_out.loc[invalid_gyro_mask, g_cols[2]] = 0.0

    invalid_accel_mask = ~valid_accel
    if invalid_accel_mask.any():
        df_out.loc[invalid_accel_mask, a_cols[0]] = 0.0
        df_out.loc[invalid_accel_mask, a_cols[1]] = 0.0
        df_out.loc[invalid_accel_mask, a_cols[2]] = 0.0

    corrupted_count = int(is_corrupted.sum())

    diagnostics = {
        "total_samples": len(df_out),
        "corrupted_samples": corrupted_count,
        "corruption_fraction": float(corrupted_count / len(df_out)),
        "ts_non_finite_count": int((~ts_valid).sum()),
        "gyro_invalid_count": int((~valid_gyro).sum()),
        "accel_invalid_count": int((~valid_accel).sum()),
        "gyro_non_finite_count": int((~w_finite).sum()),
        "accel_non_finite_count": int((~a_finite).sum()),
        "gyro_saturated_count": int((w_finite & (w_norm > gyro_sat_thresh)).sum()),
        "accel_saturated_count": int((a_finite & (a_norm > accel_sat_thresh)).sum()),
        "terminal_failure_count": int(terminal_failure.sum()),
        "gyro_sat_thresh_rad_s": float(gyro_sat_thresh),
        "accel_sat_thresh_m_s2": float(accel_sat_thresh),
        "policy": "independent_validity_flags_raw_preserved_zero_placeholder_no_forward_fill",
    }

    return df_out, diagnostics


def apply_provisional_degradation(
    df_imu: pd.DataFrame,
    scenario_id: str,
    seed: int = 42,
    time_col: str = IMU_TIMESTAMP_COL,
    gyro_cols: Optional[List[str]] = None,
    accel_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies provisional synthetic sensor degradation to an IMU stream.

    Explicitly tagged:
        PROVISIONAL DEVELOPMENT TESTING - NOT OFFICIAL ORGANIZER DEGRADATION

    Supported Scenarios:
    - S4-01: Baseline (Unmodified input stream).
    - S4-02: Progressive Noise (Sampled discrete noise scales 1x to 5x over recording).
             Gyro: sigma = 0.005 rad/s, Accel: sigma = 0.05 m/s^2 at t=0.
    - S4-03: Progressive Linear Bias Drift:
             Gyro: [+0.05, -0.03, +0.04] deg/s per minute.
             Accel: [+0.05, +0.10, -0.05] m/s^2 per minute.
    - S4-04: Random Frame Drops:
             Deterministic pseudo-random 3.0% frame removal.
    - S4-05: Burst Frame Drops:
             Five 75 ms outages (15 consecutive frames dropped at 200 Hz) at
             t = 15.0s, 45.0s, 75.0s, 105.0s, 145.0s.
    - S4-06: 2-Second Dynamic Blackout:
             Complete sensor loss from t = 50.0s to 52.0s.
    - S4-07: 2-Second Quasi-Stationary Blackout:
             Complete sensor loss from t = 24.0s to 26.0s (inside 21.2s - 31.6s interval).
    - S4-08: 5-Second Terminal Failure:
             Complete sensor loss for the final 5.0 seconds of the IMU recording.

    Parameters
    ----------
    df_imu : pd.DataFrame
        Raw IMU DataFrame. Left completely unmodified.
    scenario_id : str
        One of 'S4-01' through 'S4-08'.
    seed : int
        Deterministic random seed (default: 42).
    time_col : str
        Timestamp column name.
    gyro_cols : List[str], optional
        Gyro column names.
    accel_cols : List[str], optional
        Accel column names.

    Returns
    -------
    df_degraded : pd.DataFrame
        Degraded IMU DataFrame. Original timestamps preserved as int64.
    metadata : Dict[str, Any]
        Complete provenance and scenario metadata.

    Raises
    ------
    ValueError
        If DataFrame is empty, required columns missing, or unknown scenario_id.
    """
    valid_scenarios = [f"S4-0{i}" for i in range(1, 9)]
    if scenario_id not in valid_scenarios:
        raise ValueError(f"Unknown scenario_id '{scenario_id}'. Must be one of {valid_scenarios}.")

    if df_imu.empty:
        raise ValueError("Cannot apply degradation to empty IMU DataFrame.")

    g_cols = gyro_cols or GYRO_COLS
    a_cols = accel_cols or ACCEL_COLS

    for col in [time_col] + g_cols + a_cols:
        if col not in df_imu.columns:
            raise ValueError(f"Required column '{col}' missing from IMU DataFrame.")

    # Never mutate caller's DataFrame
    df_work = df_imu.copy()

    # Preserve exact raw integer timestamps
    raw_timestamps_ns = df_work[time_col].to_numpy(dtype=np.int64)
    n_original = len(df_work)

    t0_ns = int(raw_timestamps_ns[0])
    tend_ns = int(raw_timestamps_ns[-1])
    total_duration_s = float((tend_ns - t0_ns) * 1e-9)

    # Relative time array in seconds from recording start
    t_rel_s = (raw_timestamps_ns - t0_ns).astype(np.float64) * 1e-9

    rng = np.random.default_rng(seed)

    base_metadata: Dict[str, Any] = {
        "provenance": PROVENANCE_TAG,
        "is_official_organizer_degradation": False,
        "scenario_id": scenario_id,
        "seed": seed,
        "original_sample_count": n_original,
        "t0_timestamp_ns": t0_ns,
        "tend_timestamp_ns": tend_ns,
        "recording_duration_s": total_duration_s,
    }

    # -------------------------------------------------------------
    # S4-01: Baseline Nominal
    # -------------------------------------------------------------
    if scenario_id == "S4-01":
        base_metadata.update({
            "scenario_name": "Baseline Nominal",
            "dropped_samples": 0,
            "final_sample_count": n_original,
            "description": "Unmodified original IMU telemetry stream.",
        })
        return df_work, base_metadata

    # -------------------------------------------------------------
    # S4-02: Progressive Noise
    # -------------------------------------------------------------
    if scenario_id == "S4-02":
        # Noise scales linearly from 1x at t=0 to 5x at t=total_duration_s
        alpha = 1.0 + 4.0 * (t_rel_s / max(total_duration_s, 1e-6))  # shape: (N,)

        # Sampled discrete Gaussian standard deviations
        w_noise = rng.normal(
            loc=0.0,
            scale=BASELINE_GYRO_SAMPLED_STD_RAD_S,
            size=(n_original, 3),
        ) * alpha[:, np.newaxis]

        a_noise = rng.normal(
            loc=0.0,
            scale=BASELINE_ACCEL_SAMPLED_STD_M_S2,
            size=(n_original, 3),
        ) * alpha[:, np.newaxis]

        df_work[g_cols[0]] += w_noise[:, 0]
        df_work[g_cols[1]] += w_noise[:, 1]
        df_work[g_cols[2]] += w_noise[:, 2]

        df_work[a_cols[0]] += a_noise[:, 0]
        df_work[a_cols[1]] += a_noise[:, 1]
        df_work[a_cols[2]] += a_noise[:, 2]

        base_metadata.update({
            "scenario_name": "Progressive Sampled Noise",
            "base_gyro_std_rad_s": BASELINE_GYRO_SAMPLED_STD_RAD_S,
            "base_accel_std_m_s2": BASELINE_ACCEL_SAMPLED_STD_M_S2,
            "scale_factor_range": (1.0, 5.0),
            "dropped_samples": 0,
            "final_sample_count": n_original,
            "description": "Discrete sampled Gaussian noise scaling linearly from 1x to 5x over recording.",
        })
        return df_work, base_metadata

    # -------------------------------------------------------------
    # S4-03: Progressive Linear Bias Drift
    # -------------------------------------------------------------
    if scenario_id == "S4-03":
        # Gyro drift: [+0.05, -0.03, +0.04] deg/s per minute -> convert to rad/s per second
        # 1 deg/s = (pi / 180) rad/s; 1 minute = 60 seconds
        deg2rad = np.pi / 180.0
        gyro_drift_rate_rad_s2 = np.array([+0.05, -0.03, +0.04]) * (deg2rad / 60.0)

        # Accel drift: [+0.05, +0.10, -0.05] m/s^2 per minute -> m/s^2 per second
        accel_drift_rate_m_s3 = np.array([+0.05, +0.10, -0.05]) / 60.0

        # Linear drift accumulation b(t) = rate * t
        w_drift = np.outer(t_rel_s, gyro_drift_rate_rad_s2)
        a_drift = np.outer(t_rel_s, accel_drift_rate_m_s3)

        df_work[g_cols[0]] += w_drift[:, 0]
        df_work[g_cols[1]] += w_drift[:, 1]
        df_work[g_cols[2]] += w_drift[:, 2]

        df_work[a_cols[0]] += a_drift[:, 0]
        df_work[a_cols[1]] += a_drift[:, 1]
        df_work[a_cols[2]] += a_drift[:, 2]

        base_metadata.update({
            "scenario_name": "Progressive Linear Bias Drift",
            "gyro_drift_rate_deg_s_per_min": [+0.05, -0.03, +0.04],
            "accel_drift_rate_m_s2_per_min": [+0.05, +0.10, -0.05],
            "final_gyro_drift_rad_s": list(w_drift[-1]),
            "final_accel_drift_m_s2": list(a_drift[-1]),
            "dropped_samples": 0,
            "final_sample_count": n_original,
            "description": "Deterministic linear bias drift added continuously over flight.",
        })
        return df_work, base_metadata

    # -------------------------------------------------------------
    # S4-04: Random Frame Drops (3.0% frame removal)
    # -------------------------------------------------------------
    if scenario_id == "S4-04":
        # Exactly round(0.03 * n_original) rows removed deterministically
        num_to_drop = int(round(0.03 * n_original))

        # Eligible interior indices (excluding first index 0 and last index n_original - 1)
        eligible_indices = np.arange(1, n_original - 1, dtype=np.int64)

        # Deterministic sampling without replacement
        drop_indices = rng.choice(eligible_indices, size=num_to_drop, replace=False)

        drop_mask = np.zeros(n_original, dtype=bool)
        drop_mask[drop_indices] = True

        df_degraded = df_work[~drop_mask].reset_index(drop=True)
        dropped_count = int(drop_mask.sum())

        base_metadata.update({
            "scenario_name": "Random Frame Drops",
            "drop_fraction_target": 0.03,
            "dropped_samples": dropped_count,
            "final_sample_count": len(df_degraded),
            "description": "Deterministic sampling without replacement of exactly 3% of IMU frames.",
        })
        return df_degraded, base_metadata

    # -------------------------------------------------------------
    # S4-05: Burst Frame Drops (Five 75 ms outages)
    # -------------------------------------------------------------
    if scenario_id == "S4-05":
        # Outage START times: 15.0, 45.0, 75.0, 105.0, 145.0 seconds
        burst_starts_s = [15.0, 45.0, 75.0, 105.0, 145.0]
        burst_duration_s = 0.075  # 75 ms (exactly 15 samples at 200 Hz)

        drop_mask = np.zeros(n_original, dtype=bool)
        for t_start in burst_starts_s:
            t_end = t_start + burst_duration_s
            in_burst = (t_rel_s >= t_start) & (t_rel_s < t_end)
            drop_mask |= in_burst

        # Preserve first and last samples
        drop_mask[0] = False
        drop_mask[-1] = False

        df_degraded = df_work[~drop_mask].reset_index(drop=True)
        dropped_count = int(drop_mask.sum())

        base_metadata.update({
            "scenario_name": "Consecutive Burst Dropouts",
            "burst_starts_s": burst_starts_s,
            "burst_duration_s": burst_duration_s,
            "dropped_samples": dropped_count,
            "final_sample_count": len(df_degraded),
            "description": "Five periodic 75 ms burst outages starting at 15, 45, 75, 105, 145s.",
        })
        return df_degraded, base_metadata

    # -------------------------------------------------------------
    # S4-06: 2-Second Dynamic Blackout (50.0s to 52.0s)
    # -------------------------------------------------------------
    if scenario_id == "S4-06":
        blackout_start_s = 50.0
        blackout_end_s = 52.0

        drop_mask = (t_rel_s >= blackout_start_s) & (t_rel_s < blackout_end_s)
        drop_mask[0] = False
        drop_mask[-1] = False

        df_degraded = df_work[~drop_mask].reset_index(drop=True)
        dropped_count = int(drop_mask.sum())

        base_metadata.update({
            "scenario_name": "2-Second Dynamic Blackout",
            "blackout_start_s": blackout_start_s,
            "blackout_end_s": blackout_end_s,
            "blackout_duration_s": blackout_end_s - blackout_start_s,
            "dropped_samples": dropped_count,
            "final_sample_count": len(df_degraded),
            "description": "Complete sensor outage for 2.0s during dynamic high-speed flight.",
        })
        return df_degraded, base_metadata

    # -------------------------------------------------------------
    # S4-07: 2-Second Quasi-Stationary Blackout (24.0s to 26.0s)
    # -------------------------------------------------------------
    if scenario_id == "S4-07":
        blackout_start_s = 24.0
        blackout_end_s = 26.0

        drop_mask = (t_rel_s >= blackout_start_s) & (t_rel_s < blackout_end_s)
        drop_mask[0] = False
        drop_mask[-1] = False

        df_degraded = df_work[~drop_mask].reset_index(drop=True)
        dropped_count = int(drop_mask.sum())

        base_metadata.update({
            "scenario_name": "2-Second Quasi-Stationary Blackout",
            "blackout_start_s": blackout_start_s,
            "blackout_end_s": blackout_end_s,
            "blackout_duration_s": blackout_end_s - blackout_start_s,
            "dropped_samples": dropped_count,
            "final_sample_count": len(df_degraded),
            "description": "Complete 2.0s sensor outage inside the quasi-stationary interval (21.2s - 31.6s).",
        })
        return df_degraded, base_metadata

    # -------------------------------------------------------------
    # S4-08: 5-Second Terminal Failure (Final 5.0 seconds)
    # -------------------------------------------------------------
    if scenario_id == "S4-08":
        failure_duration_s = 5.0
        t_fail_rel_s = max(0.0, total_duration_s - failure_duration_s)
        t_fail_ns = t0_ns + int(round(t_fail_rel_s * 1e9))

        # Drop all samples occurring after t_fail_rel_s
        keep_mask = t_rel_s <= t_fail_rel_s

        df_degraded = df_work[keep_mask].reset_index(drop=True)
        dropped_count = n_original - len(df_degraded)

        last_valid_ts_ns = int(df_degraded[time_col].iloc[-1])
        first_missing_ts_ns = int(df_work[~keep_mask][time_col].iloc[0]) if (~keep_mask).any() else None

        base_metadata.update({
            "scenario_name": "5-Second Terminal Sensor Failure",
            "failure_duration_s": failure_duration_s,
            "failure_start_rel_s": t_fail_rel_s,
            "failure_start_timestamp_ns": t_fail_ns,
            "last_valid_timestamp_ns": last_valid_ts_ns,
            "first_missing_timestamp_ns": first_missing_ts_ns,
            "dropped_samples": dropped_count,
            "final_sample_count": len(df_degraded),
            "description": "Complete sensor loss for the final 5.0s of the recording.",
        })
        return df_degraded, base_metadata

    raise RuntimeError(f"Unhandled scenario_id: {scenario_id}")
