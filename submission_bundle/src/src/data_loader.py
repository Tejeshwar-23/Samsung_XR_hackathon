"""
Reusable data parsing and validation module for EuRoC MAV datasets.
Supports IMU and ground-truth trajectory state estimation loading.

Preserves int64 nanosecond timestamps, parses numerical columns as float64,
validates columns, check monotonicity, verifies quaternion norms, and checks
for data integrity without silently discarding records.
"""

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd


# Canonical column names for IMU
IMU_TIMESTAMP_COL = "timestamp"
IMU_SENSOR_COLS = [
    "w_RS_S_x",
    "w_RS_S_y",
    "w_RS_S_z",
    "a_RS_S_x",
    "a_RS_S_y",
    "a_RS_S_z",
]
IMU_ALL_COLS = [IMU_TIMESTAMP_COL] + IMU_SENSOR_COLS

# Canonical column names for Ground Truth
GT_TIMESTAMP_COL = "timestamp"
GT_POSITION_COLS = ["p_RS_R_x", "p_RS_R_y", "p_RS_R_z"]
GT_QUATERNION_COLS = ["q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z"]
GT_VELOCITY_COLS = ["v_RS_R_x", "v_RS_R_y", "v_RS_R_z"]
GT_GYRO_BIAS_COLS = ["b_w_RS_S_x", "b_w_RS_S_y", "b_w_RS_S_z"]
GT_ACCEL_BIAS_COLS = ["b_a_RS_S_x", "b_a_RS_S_y", "b_a_RS_S_z"]
GT_ALL_COLS = (
    [GT_TIMESTAMP_COL]
    + GT_POSITION_COLS
    + GT_QUATERNION_COLS
    + GT_VELOCITY_COLS
    + GT_GYRO_BIAS_COLS
    + GT_ACCEL_BIAS_COLS
)


def _clean_header_name(raw_name: str) -> str:
    """Strip leading '#' or whitespace and remove bracketed units like '[ns]' or '[rad s^-1]'."""
    cleaned = raw_name.strip()
    if cleaned.startswith("#"):
        cleaned = cleaned[1:].strip()
    if "[" in cleaned:
        cleaned = cleaned.split("[")[0].strip()
    return cleaned


def load_imu_data(
    file_path: Union[str, Path],
    allow_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Load and validate raw EuRoC IMU CSV data.

    Expected raw columns:
      - #timestamp [ns]
      - w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1]
      - a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2]

    Parameters:
      file_path: Path to the IMU CSV file.
      allow_duplicates: If True, permits non-decreasing timestamps (dt >= 0)
                        to support degradation scripts with duplicated rows while strictly
                        rejecting genuinely decreasing timestamps (dt < 0).
                        If False (default), enforces strictly increasing timestamps (dt > 0).

    Returns:
      pd.DataFrame with columns:
        ['timestamp', 'w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z', 'a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']
      where 'timestamp' is int64 (ns) and all sensor axes are float64.

    Raises:
      FileNotFoundError: If the file does not exist.
      ValueError: If required columns are missing, data contains missing values,
                  duplicate timestamps (when allow_duplicates=False), non-finite values,
                  non-chronological timestamps, or is empty.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"IMU file not found: {path}")

    try:
        df_raw = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read IMU CSV file '{path}': {exc}") from exc

    if df_raw.empty:
        raise ValueError(f"IMU dataset at '{path}' is empty.")

    # Standardize column headers
    renamed_cols = {_clean_header_name(c): c for c in df_raw.columns}
    col_mapping = {}

    for required in IMU_ALL_COLS:
        if required not in renamed_cols:
            raise ValueError(
                f"Missing required IMU column '{required}' in '{path}'. "
                f"Available cleaned columns: {list(renamed_cols.keys())}"
            )
        col_mapping[renamed_cols[required]] = required

    # Select and rename to standard columns
    df = df_raw[list(col_mapping.keys())].rename(columns=col_mapping)

    # 1. Check for missing values (NaN, None, empty)
    missing_count = df.isnull().sum()
    if missing_count.sum() > 0:
        missing_details = missing_count[missing_count > 0].to_dict()
        raise ValueError(
            f"IMU data contains missing values in '{path}': {missing_details}"
        )

    # 2. Convert and validate timestamp as int64 nanoseconds
    try:
        df[IMU_TIMESTAMP_COL] = df[IMU_TIMESTAMP_COL].astype(np.int64)
    except Exception as exc:
        raise ValueError(f"Failed to convert IMU timestamps to int64: {exc}") from exc

    # Check timestamps monotonicity and duplicates
    dt = df[IMU_TIMESTAMP_COL].diff().iloc[1:]
    if not allow_duplicates:
        num_duplicates = df[IMU_TIMESTAMP_COL].duplicated().sum()
        if num_duplicates > 0:
            raise ValueError(
                f"IMU dataset contains {num_duplicates} duplicate timestamps in '{path}'."
            )
        if not (dt > 0).all():
            non_positive_count = (dt <= 0).sum()
            raise ValueError(
                f"IMU timestamps are not strictly increasing in '{path}'. "
                f"Found {non_positive_count} non-positive time deltas."
            )
    else:
        # Accept non-decreasing timestamps (dt >= 0) while strictly rejecting decreasing timestamps (dt < 0)
        if (dt < 0).any():
            decreasing_count = (dt < 0).sum()
            raise ValueError(
                f"IMU timestamps are not strictly increasing in '{path}'. "
                f"Found {decreasing_count} negative time deltas."
            )

    # 3. Convert and validate sensor values as float64 and finite
    for col in IMU_SENSOR_COLS:
        try:
            df[col] = df[col].astype(np.float64)
        except Exception as exc:
            raise ValueError(
                f"Column '{col}' could not be parsed as float64 in '{path}': {exc}"
            ) from exc

        if not np.isfinite(df[col]).all():
            non_finite_count = (~np.isfinite(df[col])).sum()
            raise ValueError(
                f"IMU column '{col}' contains {non_finite_count} non-finite (NaN/Inf) values in '{path}'."
            )

    return df[IMU_ALL_COLS].copy()


def load_groundtruth_data(
    file_path: Union[str, Path],
    validate_quat_norm: bool = True,
    quat_norm_tolerance: float = 1e-3,
) -> pd.DataFrame:
    """
    Load and validate raw EuRoC Ground Truth state estimate CSV data.

    Expected raw columns:
      - #timestamp
      - position: p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m]
      - quaternion: q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []
      - velocity: v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1]
      - gyro bias: b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1]
      - accel bias: b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]

    Returns:
      pd.DataFrame with 17 canonical columns:
        ['timestamp',
         'p_RS_R_x', 'p_RS_R_y', 'p_RS_R_z',
         'q_RS_w', 'q_RS_x', 'q_RS_y', 'q_RS_z',
         'v_RS_R_x', 'v_RS_R_y', 'v_RS_R_z',
         'b_w_RS_S_x', 'b_w_RS_S_y', 'b_w_RS_S_z',
         'b_a_RS_S_x', 'b_a_RS_S_y', 'b_a_RS_S_z']
      where 'timestamp' is int64 (ns), quaternions are in w, x, y, z order,
      and all other quantities are float64.

    Raises:
      FileNotFoundError: If the file does not exist.
      ValueError: If columns are missing, data contains missing values,
                  non-finite values, duplicate or non-increasing timestamps,
                  or invalid unit quaternion norms.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    try:
        df_raw = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read Ground Truth CSV file '{path}': {exc}") from exc

    if df_raw.empty:
        raise ValueError(f"Ground truth dataset at '{path}' is empty.")

    # Standardize column headers
    renamed_cols = {_clean_header_name(c): c for c in df_raw.columns}
    col_mapping = {}

    for required in GT_ALL_COLS:
        if required not in renamed_cols:
            raise ValueError(
                f"Missing required Ground Truth column '{required}' in '{path}'. "
                f"Available cleaned columns: {list(renamed_cols.keys())}"
            )
        col_mapping[renamed_cols[required]] = required

    # Select and rename to standard columns
    df = df_raw[list(col_mapping.keys())].rename(columns=col_mapping)

    # 1. Check for missing values
    missing_count = df.isnull().sum()
    if missing_count.sum() > 0:
        missing_details = missing_count[missing_count > 0].to_dict()
        raise ValueError(
            f"Ground truth data contains missing values in '{path}': {missing_details}"
        )

    # 2. Convert and validate timestamp as int64 nanoseconds
    try:
        df[GT_TIMESTAMP_COL] = df[GT_TIMESTAMP_COL].astype(np.int64)
    except Exception as exc:
        raise ValueError(f"Failed to convert Ground Truth timestamps to int64: {exc}") from exc

    # Check duplicate timestamps
    num_duplicates = df[GT_TIMESTAMP_COL].duplicated().sum()
    if num_duplicates > 0:
        raise ValueError(
            f"Ground truth dataset contains {num_duplicates} duplicate timestamps in '{path}'."
        )

    # Check timestamps are strictly increasing
    dt = df[GT_TIMESTAMP_COL].diff().iloc[1:]
    if not (dt > 0).all():
        non_positive_count = (dt <= 0).sum()
        raise ValueError(
            f"Ground truth timestamps are not strictly increasing in '{path}'. "
            f"Found {non_positive_count} non-positive time deltas."
        )

    # 3. Convert all non-timestamp columns to float64 and check finiteness
    non_ts_cols = [c for c in GT_ALL_COLS if c != GT_TIMESTAMP_COL]
    for col in non_ts_cols:
        try:
            df[col] = df[col].astype(np.float64)
        except Exception as exc:
            raise ValueError(
                f"Ground truth column '{col}' could not be parsed as float64 in '{path}': {exc}"
            ) from exc

        if not np.isfinite(df[col]).all():
            non_finite_count = (~np.isfinite(df[col])).sum()
            raise ValueError(
                f"Ground truth column '{col}' contains {non_finite_count} non-finite values in '{path}'."
            )

    # 4. Validate unit quaternion norms ||q|| == 1.0 (w, x, y, z order preserved)
    if validate_quat_norm:
        q_vals = df[GT_QUATERNION_COLS].to_numpy(dtype=np.float64)
        q_norms = np.linalg.norm(q_vals, axis=1)
        deviations = np.abs(q_norms - 1.0)
        invalid_mask = deviations > quat_norm_tolerance

        if np.any(invalid_mask):
            num_invalid = int(np.sum(invalid_mask))
            max_dev = float(np.max(deviations))
            raise ValueError(
                f"Ground truth quaternions violate unit norm condition in '{path}'. "
                f"{num_invalid} samples exceed tolerance {quat_norm_tolerance:.1e} "
                f"(maximum norm deviation: {max_dev:.6e})."
            )

    return df[GT_ALL_COLS].copy()


def main() -> None:
    """Demonstrate parser on EuRoC MH_01_easy dataset with validation output."""
    project_root = Path(__file__).resolve().parent.parent
    imu_path = project_root / "data/raw/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv"
    gt_path = (
        project_root
        / "data/raw/euroc/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    )

    print("=" * 80)
    print("SAMSUNG XR HACKATHON: DATA LOADER MODULE TEST")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"IMU file:          {imu_path.relative_to(project_root)}")
    print(f"Ground truth file: {gt_path.relative_to(project_root)}\n")

    try:
        print("Loading IMU data...")
        df_imu = load_imu_data(imu_path)
        print("Loading Ground Truth data...")
        df_gt = load_groundtruth_data(gt_path)

        print("-" * 80)
        print("IMU DATASET SUMMARY")
        print("-" * 80)
        print(f"Sample count:            {len(df_imu):,}")
        print(f"First timestamp (ns):    {df_imu['timestamp'].iloc[0]}")
        print(f"Last timestamp (ns):     {df_imu['timestamp'].iloc[-1]}")
        print(
            f"Duration:                {(df_imu['timestamp'].iloc[-1] - df_imu['timestamp'].iloc[0]) * 1e-9:.3f} s"
        )
        print(f"Timestamp dtype:         {df_imu['timestamp'].dtype}")
        print(f"Parsed columns ({len(df_imu.columns)}):      {list(df_imu.columns)}")

        # Sampling interval statistics
        dt_imu_ns = df_imu['timestamp'].diff().dropna().astype(np.int64)
        print(f"Sampling interval statistics (dt):")
        print(f"  - Min dt:    {int(dt_imu_ns.min()):,} ns ({dt_imu_ns.min() * 1e-6:.4f} ms)")
        print(f"  - Max dt:    {int(dt_imu_ns.max()):,} ns ({dt_imu_ns.max() * 1e-6:.4f} ms)")
        print(f"  - Mean dt:   {float(dt_imu_ns.mean()):,.2f} ns ({dt_imu_ns.mean() * 1e-6:.4f} ms)")
        print(f"  - Median dt: {float(dt_imu_ns.median()):,.2f} ns ({dt_imu_ns.median() * 1e-6:.4f} ms)")
        print(f"  - Est. Rate: {1.0 / (float(dt_imu_ns.mean()) * 1e-9):.2f} Hz")
        print(f"Validation status:       PASSED (Monotonic, finite, zero missing/duplicates)")

        print("\n" + "-" * 80)
        print("GROUND TRUTH DATASET SUMMARY")
        print("-" * 80)
        print(f"Sample count:            {len(df_gt):,}")
        print(f"First timestamp (ns):    {df_gt['timestamp'].iloc[0]}")
        print(f"Last timestamp (ns):     {df_gt['timestamp'].iloc[-1]}")
        print(
            f"Duration:                {(df_gt['timestamp'].iloc[-1] - df_gt['timestamp'].iloc[0]) * 1e-9:.3f} s"
        )
        print(f"Timestamp dtype:         {df_gt['timestamp'].dtype}")
        print(f"Quaternion columns:      {GT_QUATERNION_COLS} (w, x, y, z order)")
        print(f"Parsed columns ({len(df_gt.columns)}):     {list(df_gt.columns)}")

        # Ground truth sampling interval and quaternion norm statistics
        dt_gt_ns = df_gt['timestamp'].diff().dropna().astype(np.int64)
        print(f"Sampling interval statistics (dt):")
        print(f"  - Min dt:    {int(dt_gt_ns.min()):,} ns ({dt_gt_ns.min() * 1e-6:.4f} ms)")
        print(f"  - Max dt:    {int(dt_gt_ns.max()):,} ns ({dt_gt_ns.max() * 1e-6:.4f} ms)")
        print(f"  - Mean dt:   {float(dt_gt_ns.mean()):,.2f} ns ({dt_gt_ns.mean() * 1e-6:.4f} ms)")
        print(f"  - Median dt: {float(dt_gt_ns.median()):,.2f} ns ({dt_gt_ns.median() * 1e-6:.4f} ms)")

        q_vals = df_gt[GT_QUATERNION_COLS].to_numpy(dtype=np.float64)
        q_norms = np.linalg.norm(q_vals, axis=1)
        print(f"Quaternion norm statistics (||[w, x, y, z]||):")
        print(f"  - Min norm:  {float(np.min(q_norms)):.8f}")
        print(f"  - Max norm:  {float(np.max(q_norms)):.8f}")
        print(f"  - Mean norm: {float(np.mean(q_norms)):.8f}")
        print(f"  - Max absolute deviation from 1.0: {float(np.max(np.abs(q_norms - 1.0))):.6e}")
        print(f"Validation status:       PASSED (Unit quaternion norm, monotonic, finite, zero missing)")

        print("\n" + "=" * 80)
        print("ALL VALIDATION CHECKS COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except Exception as exc:
        print(f"Loader failed with error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
