"""
Inspect Euroc MH_01_easy IMU and Ground Truth dataset.
Does not modify any data files.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Return the root directory of the project."""
    # Assuming this script is located in <project_root>/src/inspect_dataset.py
    return Path(__file__).resolve().parent.parent


def inspect_dataset() -> None:
    project_root = get_project_root()

    imu_rel_path = Path("data/raw/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv")
    gt_rel_path = Path("data/raw/euroc/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv")

    imu_path = project_root / imu_rel_path
    gt_path = project_root / gt_rel_path

    print("=" * 80)
    print("EUROC MAV DATASET INSPECTION: MH_01_easy")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"IMU file:          {imu_rel_path}")
    print(f"Ground truth file: {gt_rel_path}\n")

    # 1. Load both CSV files using pandas with error handling
    print("-" * 80)
    print("1. LOADING DATASETS")
    print("-" * 80)

    if not imu_path.exists():
        print(f"Error: IMU data file not found at {imu_path}", file=sys.stderr)
        sys.exit(1)

    if not gt_path.exists():
        print(f"Error: Ground truth data file not found at {gt_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df_imu = pd.read_csv(imu_path)
        print(f"Successfully loaded IMU data from {imu_rel_path}")
    except Exception as e:
        print(f"Error loading IMU CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_gt = pd.read_csv(gt_path)
        print(f"Successfully loaded Ground Truth data from {gt_rel_path}")
    except Exception as e:
        print(f"Error loading Ground Truth CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Clean whitespace from column names while preserving original names
    df_imu.columns = [c.strip() for c in df_imu.columns]
    df_gt.columns = [c.strip() for c in df_gt.columns]

    # Identify timestamp columns
    imu_time_col = df_imu.columns[0]
    gt_time_col = df_gt.columns[0]

    # 2 & 3. Sample Counts
    print("\n" + "-" * 80)
    print("2. SAMPLE COUNTS")
    print("-" * 80)
    num_imu_samples = len(df_imu)
    num_gt_samples = len(df_gt)
    print(f"Total IMU samples:          {num_imu_samples:,}")
    print(f"Total Ground-Truth samples: {num_gt_samples:,}")

    # 4 & 5. IMU Sampling Intervals and Frequency
    print("\n" + "-" * 80)
    print("3. IMU SAMPLING INTERVALS & FREQUENCY ANALYSIS")
    print("-" * 80)

    # Ensure timestamps are int64
    ts_imu = df_imu[imu_time_col].astype(np.int64)
    # Calculate dt in nanoseconds preserving int64
    dt_ns = ts_imu.diff().dropna().astype(np.int64)

    min_dt_ns = int(dt_ns.min())
    max_dt_ns = int(dt_ns.max())
    mean_dt_ns = float(dt_ns.mean())
    median_dt_ns = float(dt_ns.median())

    min_dt_s = min_dt_ns * 1e-9
    max_dt_s = max_dt_ns * 1e-9
    mean_dt_s = mean_dt_ns * 1e-9
    median_dt_s = median_dt_ns * 1e-9

    est_freq_mean_hz = 1.0 / mean_dt_s if mean_dt_s > 0 else 0.0
    est_freq_median_hz = 1.0 / median_dt_s if median_dt_s > 0 else 0.0

    print(f"Timestamp column: '{imu_time_col}' (stored as int64 nanoseconds)")
    print(f"Sampling interval (dt) statistics:")
    print(f"  - Minimum: {min_dt_ns:,} ns ({min_dt_s * 1000:.4f} ms)")
    print(f"  - Maximum: {max_dt_ns:,} ns ({max_dt_s * 1000:.4f} ms)")
    print(f"  - Mean:    {mean_dt_ns:,.2f} ns ({mean_dt_s * 1000:.4f} ms)")
    print(f"  - Median:  {median_dt_ns:,.2f} ns ({median_dt_s * 1000:.4f} ms)")
    print(f"Estimated Sampling Frequency:")
    print(f"  - Based on Mean dt:   {est_freq_mean_hz:.2f} Hz")
    print(f"  - Based on Median dt: {est_freq_median_hz:.2f} Hz")

    # 6. IMU Sensor Statistics
    print("\n" + "-" * 80)
    print("4. IMU SENSOR COLUMN STATISTICS (6 SENSOR AXES)")
    print("-" * 80)

    # All columns except timestamp are sensor columns
    imu_sensor_cols = [c for c in df_imu.columns if c != imu_time_col]
    print(f"{'Column Name':<28} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'Std Dev':>12}")
    print("-" * 85)
    for col in imu_sensor_cols:
        col_data = pd.to_numeric(df_imu[col], errors="coerce")
        col_min = col_data.min()
        col_max = col_data.max()
        col_mean = col_data.mean()
        col_std = col_data.std()
        print(f"{col:<28} | {col_min:12.6f} | {col_max:12.6f} | {col_mean:12.6f} | {col_std:12.6f}")

    # 7. Data Quality Checks
    print("\n" + "-" * 80)
    print("5. DATA QUALITY CHECKS (IMU DATASET)")
    print("-" * 80)

    # Missing values
    missing_counts = df_imu.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"A. Missing Values (NaN/None): {total_missing}")
    if total_missing > 0:
        for c, count in missing_counts[missing_counts > 0].items():
            print(f"   - {c}: {count} missing values")
    else:
        print("   -> No missing values detected.")

    # Duplicate timestamps
    dup_timestamps = ts_imu.duplicated().sum()
    print(f"B. Duplicate Timestamps: {dup_timestamps}")
    if dup_timestamps > 0:
        dup_vals = ts_imu[ts_imu.duplicated(keep=False)].unique()
        print(f"   -> Found {dup_timestamps} duplicate timestamp rows ({len(dup_vals)} unique values).")
    else:
        print("   -> No duplicate timestamps detected.")

    # Invalid values (non-finite or NaN in numeric sensor columns)
    invalid_counts = 0
    for col in imu_sensor_cols:
        series = pd.to_numeric(df_imu[col], errors="coerce")
        invalids = series.isna() | np.isinf(series)
        invalid_in_col = invalids.sum()
        invalid_counts += invalid_in_col
        if invalid_in_col > 0:
            print(f"   - Column '{col}' contains {invalid_in_col} invalid/non-numeric/infinite values.")
    print(f"C. Invalid / Non-finite Values across sensor columns: {invalid_counts}")
    if invalid_counts == 0:
        print("   -> All sensor values are valid finite numerical floats.")

    # Monotonicity check
    is_strictly_monotonic = (dt_ns > 0).all()
    print(f"D. Timestamp Monotonicity: {'Strictly increasing' if is_strictly_monotonic else 'Not strictly increasing'}")
    if not is_strictly_monotonic:
        non_positive = (dt_ns <= 0).sum()
        print(f"   -> Warning: {non_positive} intervals have dt <= 0 ns.")

    # Unusually large timestamp gaps
    # Nominal interval for 200 Hz is 5 ms (5,000,000 ns).
    # Gaps larger than 1.5x nominal (or median) are flagged.
    gap_threshold_ns = int(median_dt_ns * 1.5)
    large_gaps = dt_ns[dt_ns > gap_threshold_ns]
    print(f"E. Unusually Large Timestamp Gaps (> {gap_threshold_ns * 1e-6:.2f} ms / 1.5x median): {len(large_gaps)}")
    if len(large_gaps) > 0:
        print(f"   Top largest gaps found:")
        top_gaps = large_gaps.sort_values(ascending=False).head(5)
        for idx, gap_val in top_gaps.items():
            t_prev = ts_imu.loc[idx - 1]
            t_curr = ts_imu.loc[idx]
            print(f"   - Index {idx}: gap = {gap_val * 1e-6:.4f} ms ({gap_val:,} ns) between t={t_prev} and t={t_curr}")
    else:
        print("   -> No unusually large timestamp gaps detected.")

    # 8. Ground Truth Columns and First 5 Rows
    print("\n" + "-" * 80)
    print("6. GROUND TRUTH DATASET OVERVIEW")
    print("-" * 80)
    print("Column names:")
    for idx, col in enumerate(df_gt.columns, 1):
        print(f"  {idx:2d}. {col}")

    print("\nFirst 5 rows of Ground Truth data:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df_gt.head(5).to_string(index=True))

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE (No files were modified)")
    print("=" * 80)


if __name__ == "__main__":
    inspect_dataset()
