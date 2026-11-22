"""
Timestamp synchronization and Ground Truth interpolation module for EuRoC MAV datasets.

Provides:
- Detection of overlapping timestamp intervals between IMU and Ground Truth.
- High-precision alignment of Ground Truth state to IMU query timestamps:
  - 3D Position: Linear interpolation.
  - 3D Velocity: Linear interpolation.
  - Unit Quaternion: Spherical Linear Interpolation (SLERP) preserving internal [w, x, y, z] convention.
  - Biases: Linear interpolation.
- Numerical safety:
  - Uses int64 nanosecond subtraction relative to a baseline reference timestamp
    before float conversion, eliminating loss of floating-point precision on Unix epoch timestamps.
  - Enforces strict boundary checks (never extrapolates outside the ground-truth interval).
  - Returns a boolean validity mask over IMU timestamps.
  - Preserves original input DataFrames immutably.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

try:
    from src.data_loader import (
        GT_POSITION_COLS,
        GT_QUATERNION_COLS,
        GT_VELOCITY_COLS,
        GT_GYRO_BIAS_COLS,
        GT_ACCEL_BIAS_COLS,
        GT_ALL_COLS,
    )
except ModuleNotFoundError:
    from data_loader import (
        GT_POSITION_COLS,
        GT_QUATERNION_COLS,
        GT_VELOCITY_COLS,
        GT_GYRO_BIAS_COLS,
        GT_ACCEL_BIAS_COLS,
        GT_ALL_COLS,
    )


def get_overlapping_interval(
    df_imu: pd.DataFrame,
    df_gt: pd.DataFrame,
    imu_time_col: str = "timestamp",
    gt_time_col: str = "timestamp",
) -> Dict[str, Any]:
    """
    Identify the overlapping timestamp interval between IMU and ground truth.

    Args:
        df_imu: DataFrame containing IMU records.
        df_gt: DataFrame containing ground truth records.
        imu_time_col: Column name of IMU timestamp.
        gt_time_col: Column name of ground truth timestamp.

    Returns:
        Dictionary containing:
            - 'has_overlap': bool
            - 'overlap_start_ns': int (int64) or None
            - 'overlap_end_ns': int (int64) or None
            - 'overlap_duration_s': float or 0.0
            - 'imu_overlap_count': int
            - 'gt_overlap_count': int
            - 'imu_overlap_mask': np.ndarray (bool)
            - 'gt_overlap_mask': np.ndarray (bool)
    """
    if df_imu.empty or df_gt.empty:
        raise ValueError("IMU and Ground Truth DataFrames must not be empty.")

    if imu_time_col not in df_imu.columns:
        raise ValueError(f"IMU timestamp column '{imu_time_col}' not found.")
    if gt_time_col not in df_gt.columns:
        raise ValueError(f"Ground Truth timestamp column '{gt_time_col}' not found.")

    ts_imu = df_imu[imu_time_col].astype(np.int64).to_numpy()
    ts_gt = df_gt[gt_time_col].astype(np.int64).to_numpy()

    t_imu_min, t_imu_max = int(ts_imu[0]), int(ts_imu[-1])
    t_gt_min, t_gt_max = int(ts_gt[0]), int(ts_gt[-1])

    t_overlap_start = max(t_imu_min, t_gt_min)
    t_overlap_end = min(t_imu_max, t_gt_max)

    has_overlap = t_overlap_start <= t_overlap_end

    if not has_overlap:
        imu_mask = np.zeros(len(ts_imu), dtype=bool)
        gt_mask = np.zeros(len(ts_gt), dtype=bool)
        return {
            "has_overlap": False,
            "overlap_start_ns": None,
            "overlap_end_ns": None,
            "overlap_duration_s": 0.0,
            "imu_overlap_count": 0,
            "gt_overlap_count": 0,
            "imu_overlap_mask": imu_mask,
            "gt_overlap_mask": gt_mask,
        }

    imu_mask = (ts_imu >= t_overlap_start) & (ts_imu <= t_overlap_end)
    gt_mask = (ts_gt >= t_overlap_start) & (ts_gt <= t_overlap_end)
    overlap_duration_s = float((t_overlap_end - t_overlap_start) * 1e-9)

    return {
        "has_overlap": True,
        "overlap_start_ns": int(t_overlap_start),
        "overlap_end_ns": int(t_overlap_end),
        "overlap_duration_s": overlap_duration_s,
        "imu_overlap_count": int(np.sum(imu_mask)),
        "gt_overlap_count": int(np.sum(gt_mask)),
        "imu_overlap_mask": imu_mask,
        "gt_overlap_mask": gt_mask,
    }


def interpolate_groundtruth_to_imu(
    df_imu: pd.DataFrame,
    df_gt: pd.DataFrame,
    imu_time_col: str = "timestamp",
    gt_time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Interpolate Ground Truth states onto IMU timestamps.

    Interpolation rules:
      - 3D Position (p_RS_R_*): Linear interpolation.
      - 3D Velocity (v_RS_R_*): Linear interpolation.
      - Quaternion (q_RS_w, q_RS_x, q_RS_y, q_RS_z): Spherical Linear Interpolation (SLERP).
        Internal conventions:
          * Loader order: [q_RS_w, q_RS_x, q_RS_y, q_RS_z] (w, x, y, z)
          * SciPy order: [q_x, q_y, q_z, q_w] (x, y, z, w)
          * Output DataFrame order: Preserved strictly as [q_RS_w, q_RS_x, q_RS_y, q_RS_z].
      - Sensor Biases (b_w_RS_S_*, b_a_RS_S_*): Linear interpolation.
      - Boundaries: Never interpolates outside [t_gt_min, t_gt_max].
        Returns a validity mask for IMU timestamps that fall within the ground truth interval.
        Out-of-range rows in the aligned DataFrame are populated with NaN.

    Precision:
      - Relative timestamps (t_rel_s = (t - t_ref) * 1e-9 in float64) are calculated
        AFTER integer subtraction with a baseline reference timestamp, preserving nanosecond fidelity.

    Returns:
        Tuple:
            - df_aligned_gt: pd.DataFrame with same length as df_imu, matching GT_ALL_COLS.
            - valid_mask: np.ndarray (bool) indicating IMU timestamps where GT was interpolated.
    """
    if df_imu.empty or df_gt.empty:
        raise ValueError("Input DataFrames must not be empty.")

    for col in GT_ALL_COLS:
        if col not in df_gt.columns:
            raise ValueError(f"Ground Truth column '{col}' missing from df_gt.")

    if imu_time_col not in df_imu.columns:
        raise ValueError(f"IMU timestamp column '{imu_time_col}' missing.")

    # Preserve original timestamps as int64 nanoseconds
    imu_ts = df_imu[imu_time_col].astype(np.int64).to_numpy()
    gt_ts = df_gt[gt_time_col].astype(np.int64).to_numpy()

    # Verify ground truth timestamps are strictly increasing
    dt_gt = np.diff(gt_ts)
    if not (dt_gt > 0).all():
        raise ValueError("Ground Truth timestamps must be strictly increasing for interpolation.")

    t_gt_min = gt_ts[0]
    t_gt_max = gt_ts[-1]

    # Validity mask: IMU timestamps inside ground-truth coverage
    valid_mask = (imu_ts >= t_gt_min) & (imu_ts <= t_gt_max)

    # Initialize aligned ground truth DataFrame
    n_imu = len(imu_ts)
    df_aligned = pd.DataFrame(index=df_imu.index)
    df_aligned["timestamp"] = imu_ts  # preserve original IMU int64 timestamps

    # If no overlapping samples exist, initialize remaining columns to NaN and return
    other_cols = [c for c in GT_ALL_COLS if c != "timestamp"]
    if not np.any(valid_mask):
        for col in other_cols:
            df_aligned[col] = np.nan
        return df_aligned[GT_ALL_COLS].copy(), valid_mask

    # Use t_gt_min as reference timestamp for high numerical precision
    t_ref = t_gt_min
    t_gt_rel_s = (gt_ts - t_ref).astype(np.float64) * 1e-9
    t_imu_valid_rel_s = (imu_ts[valid_mask] - t_ref).astype(np.float64) * 1e-9

    # Pre-allocate NaN columns for all state features
    for col in other_cols:
        df_aligned[col] = np.nan

    # 1. Linear interpolation for 3D Position
    p_interp = interp1d(
        t_gt_rel_s,
        df_gt[GT_POSITION_COLS].to_numpy(dtype=np.float64),
        axis=0,
        kind="linear",
        assume_sorted=True,
    )
    p_valid = p_interp(t_imu_valid_rel_s)
    for idx, col in enumerate(GT_POSITION_COLS):
        df_aligned.loc[valid_mask, col] = p_valid[:, idx]

    # 2. Linear interpolation for 3D Velocity
    v_interp = interp1d(
        t_gt_rel_s,
        df_gt[GT_VELOCITY_COLS].to_numpy(dtype=np.float64),
        axis=0,
        kind="linear",
        assume_sorted=True,
    )
    v_valid = v_interp(t_imu_valid_rel_s)
    for idx, col in enumerate(GT_VELOCITY_COLS):
        df_aligned.loc[valid_mask, col] = v_valid[:, idx]

    # 3. Linear interpolation for Biases
    bias_cols = GT_GYRO_BIAS_COLS + GT_ACCEL_BIAS_COLS
    bias_interp = interp1d(
        t_gt_rel_s,
        df_gt[bias_cols].to_numpy(dtype=np.float64),
        axis=0,
        kind="linear",
        assume_sorted=True,
    )
    bias_valid = bias_interp(t_imu_valid_rel_s)
    for idx, col in enumerate(bias_cols):
        df_aligned.loc[valid_mask, col] = bias_valid[:, idx]

    # 4. Spherical Linear Interpolation (SLERP) for Quaternions
    # Input GT order is [w, x, y, z] -> convert to SciPy Rotation [x, y, z, w]
    q_gt_wxyz = df_gt[GT_QUATERNION_COLS].to_numpy(dtype=np.float64)
    q_gt_xyzw = np.column_stack([
        q_gt_wxyz[:, 1],  # x
        q_gt_wxyz[:, 2],  # y
        q_gt_wxyz[:, 3],  # z
        q_gt_wxyz[:, 0],  # w
    ])

    # Normalize quaternions to ensure exact unit vectors for Rotation.from_quat
    q_gt_xyzw /= np.linalg.norm(q_gt_xyzw, axis=1, keepdims=True)

    rotations = Rotation.from_quat(q_gt_xyzw)
    slerp = Slerp(t_gt_rel_s, rotations)
    rot_valid = slerp(t_imu_valid_rel_s)
    q_valid_xyzw = rot_valid.as_quat()

    # Convert back from SciPy [x, y, z, w] to internal loader standard [w, x, y, z]
    q_valid_wxyz = np.column_stack([
        q_valid_xyzw[:, 3],  # w
        q_valid_xyzw[:, 0],  # x
        q_valid_xyzw[:, 1],  # y
        q_valid_xyzw[:, 2],  # z
    ])

    for idx, col in enumerate(GT_QUATERNION_COLS):
        df_aligned.loc[valid_mask, col] = q_valid_wxyz[:, idx]

    return df_aligned[GT_ALL_COLS].copy(), valid_mask


def main() -> None:
    """Demonstrate synchronization on EuRoC MH_01_easy dataset."""
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
    print("SAMSUNG XR HACKATHON: TIMESTAMP SYNCHRONIZATION & GT INTERPOLATION")
    print("=" * 80)
    print(f"IMU file: {imu_path}")
    print(f"GT file:  {gt_path}\n")

    try:
        df_imu = load_imu_data(imu_path)
        df_gt = load_groundtruth_data(gt_path)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    # 1. Overlap Analysis
    print("-" * 80)
    print("1. OVERLAPPING INTERVAL ANALYSIS")
    print("-" * 80)
    overlap = get_overlapping_interval(df_imu, df_gt)
    print(f"Has overlap:              {overlap['has_overlap']}")
    print(f"Overlap start timestamp:  {overlap['overlap_start_ns']} ns")
    print(f"Overlap end timestamp:    {overlap['overlap_end_ns']} ns")
    print(f"Overlap duration:         {overlap['overlap_duration_s']:.3f} s")
    print(f"Total IMU samples:        {len(df_imu):,}")
    print(f"IMU samples inside range: {overlap['imu_overlap_count']:,} ({overlap['imu_overlap_count'] / len(df_imu) * 100:.2f}%)")
    print(f"Total GT samples:         {len(df_gt):,}")
    print(f"GT samples inside range:  {overlap['gt_overlap_count']:,} ({overlap['gt_overlap_count'] / len(df_gt) * 100:.2f}%)")

    # 2. Alignment & Interpolation
    print("\n" + "-" * 80)
    print("2. GROUND TRUTH ALIGNMENT ON IMU TIMESTAMPS")
    print("-" * 80)
    df_aligned_gt, valid_mask = interpolate_groundtruth_to_imu(df_imu, df_gt)
    print(f"Aligned DataFrame shape:  {df_aligned_gt.shape}")
    print(f"Valid GT rows on IMU:     {int(np.sum(valid_mask)):,}")
    print(f"Invalid / outside range:  {int(np.sum(~valid_mask)):,}")

    # Check quaternion norms of interpolated poses
    q_aligned = df_aligned_gt.loc[valid_mask, GT_QUATERNION_COLS].to_numpy()
    norms = np.linalg.norm(q_aligned, axis=1)
    print(f"Interpolated SLERP Quaternions:")
    print(f"  - Quaternion order:     {GT_QUATERNION_COLS} (w, x, y, z)")
    print(f"  - Min norm:             {float(np.min(norms)):.8f}")
    print(f"  - Max norm:             {float(np.max(norms)):.8f}")
    print(f"  - Mean norm:            {float(np.mean(norms)):.8f}")

    print("\nFirst 3 valid aligned GT rows:")
    first_valid_idx = np.where(valid_mask)[0][:3]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df_aligned_gt.iloc[first_valid_idx].to_string())

    print("\n" + "=" * 80)
    print("SYNCHRONIZATION COMPLETE (Input DataFrames completely unmodified)")
    print("=" * 80)


if __name__ == "__main__":
    main()
