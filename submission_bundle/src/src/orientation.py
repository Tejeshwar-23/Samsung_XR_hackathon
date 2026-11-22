"""
Orientation estimation module for EuRoC MAV datasets using incremental gyroscope quaternion integration.

Mathematical & Implementation Conventions:
1. Coordinate Frames & Definition:
   - q represents the orientation of the sensor/body frame (S) relative to the reference frame (R),
     often denoted as R_RS or q_RS.
   - Angular velocities [w_x, w_y, w_z] are measured in the body sensor frame (rad/s).
2. Multiplication & Integration:
   - For an incremental rotation delta_q over interval dt:
     Rotation vector delta_theta = w * dt (in radians)
     Incremental rotation delta_R = Rotation.from_rotvec(delta_theta)
     Orientation update: R_{k+1} = R_k * delta_R (body-fixed / intrinsic rotation update).
3. Quaternion Format:
   - Canonical external format matches data_loader: [q_RS_w, q_RS_x, q_RS_y, q_RS_z] (w, x, y, z).
   - Internal SciPy format: [x, y, z, w]. Conversion is explicit and thoroughly documented.
   - Unit norm condition ||q|| == 1.0 is strictly maintained at every step.
4. Large Timestamp Gap Behavior:
   - If dt > gap_threshold_s:
     * If gap_policy == 'zero_order_hold' (default): Integrates using the gap duration, but warns/logs.
     * If gap_policy == 'hold_orientation': Pauses rotation integration across the anomalous gap (delta_theta = 0)
       to prevent runaway dead-reckoning divergence from missing measurements.
     * If gap_policy == 'clamp': Clamps dt to max_dt_clamp_s before integration.
   - Output rows are never deleted; every sample preserves its exact timestamp.
5. Evaluator & Error Metrics:
   - Angular difference between estimated rotation R_est and ground truth R_gt:
     delta_R = R_est.inv() * R_gt
     angular_error = norm(delta_R.as_rotvec()) in [0, pi] radians (or [0, 180] degrees).
"""

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

try:
    from src.data_loader import (
        IMU_TIMESTAMP_COL,
        GT_TIMESTAMP_COL,
        GT_QUATERNION_COLS,
    )
    from src.preprocessing import compute_timestamp_intervals, GYRO_COLS
    from src.synchronization import interpolate_groundtruth_to_imu
except ModuleNotFoundError:
    from data_loader import (
        IMU_TIMESTAMP_COL,
        GT_TIMESTAMP_COL,
        GT_QUATERNION_COLS,
    )
    from preprocessing import compute_timestamp_intervals, GYRO_COLS
    from synchronization import interpolate_groundtruth_to_imu


EST_QUATERNION_COLS = ["q_est_w", "q_est_x", "q_est_y", "q_est_z"]


def integrate_gyro_quaternion(
    df_imu: pd.DataFrame,
    initial_quaternion: Optional[Union[np.ndarray, list]] = None,
    gyro_cols: Optional[list[str]] = None,
    time_col: str = IMU_TIMESTAMP_COL,
    gap_threshold_s: Optional[float] = None,
    gap_policy: str = "hold_orientation",
    max_dt_clamp_s: float = 0.05,
) -> pd.DataFrame:
    """
    Integrate gyroscope angular rates incrementally into 3D orientations (quaternions).

    Args:
        df_imu: DataFrame containing IMU records. Must contain time_col and gyro_cols.
        initial_quaternion: Optional initial orientation quaternion [w, x, y, z].
                            If None, defaults to identity quaternion [1.0, 0.0, 0.0, 0.0]
                            (ASSUMPTION: Body frame aligned with reference frame at t=0).
                            Does not use ground truth or pre-calculated gravity vectors.
        gyro_cols: List of 3 gyroscope column names. Defaults to GYRO_COLS.
        time_col: Name of timestamp column (int64 nanoseconds).
        gap_threshold_s: Threshold in seconds to flag unusually large intervals.
                         If None, defaults to 1.5 * nominal/median dt.
        gap_policy: Behavior when dt > gap_threshold_s:
                    'hold_orientation' (DEFAULT): Do not integrate rotation across the outage
                                                  (delta_theta = 0); holds previous attitude at
                                                  the first recovery sample, resuming normal
                                                  integration on subsequent frames.
                    'zero_order_hold': Integrate using full dt (standard numerical extrapolation).
                    'clamp': Clamp effective dt to max_dt_clamp_s.
        max_dt_clamp_s: Maximum dt to allow when gap_policy='clamp'.

    Returns:
        pd.DataFrame with:
          - 'timestamp': Original int64 timestamps.
          - 'q_est_w', 'q_est_x', 'q_est_y', 'q_est_z': Estimated orientation quaternion (w, x, y, z).
          - 'dt_s': Effective time step used for integration.
          - 'is_gap': Boolean indicating whether the interval exceeded gap_threshold_s.
    """
    if df_imu.empty:
        raise ValueError("IMU DataFrame is empty.")

    g_cols = gyro_cols or GYRO_COLS
    for col in g_cols + [time_col]:
        if col not in df_imu.columns:
            raise ValueError(f"Required column '{col}' missing from IMU DataFrame.")

    n_samples = len(df_imu)
    ts = df_imu[time_col].astype(np.int64).to_numpy()
    gyro_data = df_imu[g_cols].to_numpy(dtype=np.float64)

    # 1. Process initial quaternion [w, x, y, z] -> SciPy [x, y, z, w]
    if initial_quaternion is None:
        # Documented default assumption: identity orientation
        q0_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q0_wxyz = np.asarray(initial_quaternion, dtype=np.float64)
        if q0_wxyz.shape != (4,):
            raise ValueError(f"initial_quaternion must be of length 4, got shape {q0_wxyz.shape}.")
        norm_q0 = np.linalg.norm(q0_wxyz)
        if norm_q0 == 0:
            raise ValueError("initial_quaternion cannot be zero vector.")
        q0_wxyz = q0_wxyz / norm_q0

    # Convert initial w,x,y,z to SciPy x,y,z,w
    q0_xyzw = np.array([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]], dtype=np.float64)
    q0_xyzw /= np.linalg.norm(q0_xyzw)

    # 2. Compute dt array in seconds preserving int64 subtraction
    dt_s = np.zeros(n_samples, dtype=np.float64)
    if n_samples > 1:
        dt_s[1:] = (ts[1:] - ts[:-1]).astype(np.float64) * 1e-9

    median_dt = float(np.median(dt_s[1:])) if n_samples > 1 else 0.005
    if gap_threshold_s is None:
        gap_threshold_s = 1.5 * median_dt

    is_gap = dt_s > gap_threshold_s

    # 3. Incremental integration loop
    # Pre-allocate quaternion array in SciPy [x, y, z, w]
    quats_xyzw = np.zeros((n_samples, 4), dtype=np.float64)
    quats_xyzw[0] = q0_xyzw
    curr_rot = Rotation.from_quat(q0_xyzw)

    for i in range(1, n_samples):
        dt_step = dt_s[i]

        if is_gap[i]:
            if gap_policy == "hold_orientation":
                dt_step = 0.0
            elif gap_policy == "clamp":
                dt_step = min(dt_step, max_dt_clamp_s)
            elif gap_policy == "zero_order_hold":
                pass
            else:
                raise ValueError(f"Unknown gap_policy '{gap_policy}'.")

        # Gyro angular rate [rad/s] at sample i (or midpoint average with i-1)
        # Using trapezoidal / midpoint angular velocity for second-order precision
        w_mid = 0.5 * (gyro_data[i - 1] + gyro_data[i])
        rotvec = w_mid * dt_step
        rotvec_norm = np.linalg.norm(rotvec)

        if rotvec_norm > 1e-12:
            delta_rot = Rotation.from_rotvec(rotvec)
            # Intrinsic / body-fixed rotation update: R_{k+1} = R_k * delta_R
            curr_rot = curr_rot * delta_rot
            q_step = curr_rot.as_quat()
            # Explicit re-normalization at every step to prevent drift
            q_step /= np.linalg.norm(q_step)
            curr_rot = Rotation.from_quat(q_step)
            quats_xyzw[i] = q_step
        else:
            quats_xyzw[i] = quats_xyzw[i - 1]

    # Convert SciPy [x, y, z, w] back to external [w, x, y, z] convention
    quats_wxyz = np.column_stack([
        quats_xyzw[:, 3],  # w
        quats_xyzw[:, 0],  # x
        quats_xyzw[:, 1],  # y
        quats_xyzw[:, 2],  # z
    ])

    df_out = pd.DataFrame(
        {
            time_col: ts,
            "q_est_w": quats_wxyz[:, 0],
            "q_est_x": quats_wxyz[:, 1],
            "q_est_y": quats_wxyz[:, 2],
            "q_est_z": quats_wxyz[:, 3],
            "dt_s": dt_s,
            "is_gap": is_gap,
        },
        index=df_imu.index,
    )

    return df_out


def compute_angular_error(
    quats_est_wxyz: np.ndarray,
    quats_gt_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute geodesic 3D orientation error angle between estimated and ground-truth quaternions.

    Args:
        quats_est_wxyz: Array of shape (N, 4) in [w, x, y, z] order.
        quats_gt_wxyz: Array of shape (N, 4) in [w, x, y, z] order.

    Returns:
        Tuple (errors_rad, errors_deg) as 1D arrays of shape (N,).
    """
    if quats_est_wxyz.shape != quats_gt_wxyz.shape:
        raise ValueError(
            f"Shape mismatch: est {quats_est_wxyz.shape} vs gt {quats_gt_wxyz.shape}."
        )

    if len(quats_est_wxyz) == 0:
        return np.array([]), np.array([])

    # Convert from [w, x, y, z] to SciPy [x, y, z, w]
    q_est_xyzw = np.column_stack([
        quats_est_wxyz[:, 1],
        quats_est_wxyz[:, 2],
        quats_est_wxyz[:, 3],
        quats_est_wxyz[:, 0],
    ])
    q_gt_xyzw = np.column_stack([
        quats_gt_wxyz[:, 1],
        quats_gt_wxyz[:, 2],
        quats_gt_wxyz[:, 3],
        quats_gt_wxyz[:, 0],
    ])

    # Normalize to ensure unit quaternions
    q_est_xyzw /= np.linalg.norm(q_est_xyzw, axis=1, keepdims=True)
    q_gt_xyzw /= np.linalg.norm(q_gt_xyzw, axis=1, keepdims=True)

    rot_est = Rotation.from_quat(q_est_xyzw)
    rot_gt = Rotation.from_quat(q_gt_xyzw)

    # Relative rotation delta_R = R_est^{-1} * R_gt
    delta_rot = rot_est.inv() * rot_gt
    rotvecs = delta_rot.as_rotvec()
    errors_rad = np.linalg.norm(rotvecs, axis=1)
    errors_deg = np.rad2deg(errors_rad)

    return errors_rad, errors_deg


def evaluate_orientation_against_groundtruth(
    df_imu: pd.DataFrame,
    df_gt: pd.DataFrame,
    df_est: Optional[pd.DataFrame] = None,
    initial_quaternion: Optional[Union[np.ndarray, list]] = None,
    imu_time_col: str = IMU_TIMESTAMP_COL,
    gt_time_col: str = GT_TIMESTAMP_COL,
) -> Dict[str, Any]:
    """
    Evaluate estimated orientation against synchronized ground truth.

    Reuses src/synchronization.py for ground truth alignment.
    Evaluates error strictly over valid overlapping timestamps.

    Args:
        df_imu: Original IMU DataFrame.
        df_gt: Original Ground Truth DataFrame.
        df_est: Optional precomputed orientation DataFrame from integrate_gyro_quaternion.
                If None, integrate_gyro_quaternion is called using initial_quaternion.
        initial_quaternion: Initial orientation if df_est is computed internally.

    Returns:
        Dictionary of statistics:
            - 'mean_deg', 'mean_rad'
            - 'median_deg', 'median_rad'
            - 'rmse_deg', 'rmse_rad'
            - 'p95_deg', 'p95_rad'
            - 'max_deg', 'max_rad'
            - 'valid_sample_count'
            - 'errors_deg', 'errors_rad'
            - 'df_aligned_gt'
            - 'valid_mask'
    """
    if df_est is None:
        df_est = integrate_gyro_quaternion(
            df_imu,
            initial_quaternion=initial_quaternion,
            time_col=imu_time_col,
        )

    # Synchronize ground truth to IMU timestamps using existing module
    df_aligned_gt, valid_mask = interpolate_groundtruth_to_imu(
        df_imu=df_imu,
        df_gt=df_gt,
        imu_time_col=imu_time_col,
        gt_time_col=gt_time_col,
    )

    if not np.any(valid_mask):
        raise ValueError("No overlapping ground truth samples found for evaluation.")

    q_est_valid = df_est.loc[valid_mask, EST_QUATERNION_COLS].to_numpy(dtype=np.float64)
    q_gt_valid = df_aligned_gt.loc[valid_mask, GT_QUATERNION_COLS].to_numpy(dtype=np.float64)

    errors_rad, errors_deg = compute_angular_error(q_est_valid, q_gt_valid)

    mean_rad = float(np.mean(errors_rad))
    mean_deg = float(np.mean(errors_deg))
    med_rad = float(np.median(errors_rad))
    med_deg = float(np.median(errors_deg))
    rmse_rad = float(np.sqrt(np.mean(errors_rad**2)))
    rmse_deg = float(np.sqrt(np.mean(errors_deg**2)))
    p95_rad = float(np.percentile(errors_rad, 95))
    p95_deg = float(np.percentile(errors_deg, 95))
    max_rad = float(np.max(errors_rad))
    max_deg = float(np.max(errors_deg))

    return {
        "valid_sample_count": int(np.sum(valid_mask)),
        "mean_rad": mean_rad,
        "mean_deg": mean_deg,
        "median_rad": med_rad,
        "median_deg": med_deg,
        "rmse_rad": rmse_rad,
        "rmse_deg": rmse_deg,
        "p95_rad": p95_rad,
        "p95_deg": p95_deg,
        "max_rad": max_rad,
        "max_deg": max_deg,
        "errors_rad": errors_rad,
        "errors_deg": errors_deg,
        "df_aligned_gt": df_aligned_gt,
        "valid_mask": valid_mask,
    }


def main() -> None:
    """Run gyro quaternion integration on EuRoC MH_01_easy and evaluate against ground truth."""
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
    print("SAMSUNG XR HACKATHON: GYRO QUATERNION INTEGRATION & EVALUATION")
    print("=" * 80)
    print(f"IMU file: {imu_path}")
    print(f"GT file:  {gt_path}\n")

    try:
        df_imu = load_imu_data(imu_path)
        df_gt = load_groundtruth_data(gt_path)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    # 1. Run with default identity initialization assumption
    print("-" * 80)
    print("CASE A: UNALIGNED DEFAULT INITIALIZATION (ASSUMPTION: IDENTITY [1, 0, 0, 0])")
    print("-" * 80)
    print("Note: Initial stationarity was rejected in preprocessing, so gravity alignment")
    print("      is omitted; identity orientation is an unaligned reference assumption.")

    df_est_ident = integrate_gyro_quaternion(df_imu, initial_quaternion=None)
    eval_ident = evaluate_orientation_against_groundtruth(
        df_imu=df_imu,
        df_gt=df_gt,
        df_est=df_est_ident,
    )
    print(f"Valid evaluation samples: {eval_ident['valid_sample_count']:,}")
    print(f"Angular Error Statistics (Identity Init):")
    print(f"  - Mean error:       {eval_ident['mean_deg']:8.3f} deg ({eval_ident['mean_rad']:.4f} rad)")
    print(f"  - Median error:     {eval_ident['median_deg']:8.3f} deg ({eval_ident['median_rad']:.4f} rad)")
    print(f"  - RMSE:             {eval_ident['rmse_deg']:8.3f} deg ({eval_ident['rmse_rad']:.4f} rad)")
    print(f"  - 95th Percentile:  {eval_ident['p95_deg']:8.3f} deg ({eval_ident['p95_rad']:.4f} rad)")
    print(f"  - Maximum error:    {eval_ident['max_deg']:8.3f} deg ({eval_ident['max_rad']:.4f} rad)")

    # 2. Benchmark Case B: Initialized with ground-truth orientation at the first valid synchronized timestamp
    # (Provided strictly as an evaluation-only benchmark to quantify uncorrected gyro drift alone)
    print("\n" + "-" * 80)
    print("CASE B: BENCHMARK WITH GROUND-TRUTH INITIAL POSE AT FIRST SYNCHRONIZED TIMESTAMPS")
    print("-" * 80)
    print("Evaluation benchmark only: Initialized using the interpolated ground-truth pose")
    print("at the exact first synchronized IMU timestamp, isolating sensor drift over time.")

    # Synchronize ground truth to identify first valid timestamp and its exact GT pose
    df_aligned_gt, valid_mask = interpolate_groundtruth_to_imu(df_imu=df_imu, df_gt=df_gt)
    first_valid_idx = int(np.where(valid_mask)[0][0])
    last_valid_idx = int(np.where(valid_mask)[0][-1])

    # Slice IMU strictly across the valid overlap window so integration starts at t_first_valid
    df_imu_sync = df_imu.iloc[first_valid_idx:last_valid_idx + 1].copy().reset_index(drop=True)
    t_sync_start = df_imu_sync["timestamp"].iloc[0]

    # Ground-truth quaternion corresponding to that exact same starting timestamp
    q_gt_sync_start = df_aligned_gt.loc[first_valid_idx, GT_QUATERNION_COLS].to_numpy(dtype=np.float64)

    print(f"Overlap start timestamp:            {t_sync_start} ns")
    print(f"Ground-truth orientation at start:  {q_gt_sync_start} (w, x, y, z)")
    print(f"Total synchronized evaluation samples: {len(df_imu_sync):,}")

    # Integrate strictly over the synchronized range starting with q_gt_sync_start
    df_est_benchmark = integrate_gyro_quaternion(df_imu_sync, initial_quaternion=q_gt_sync_start)
    eval_gt_init = evaluate_orientation_against_groundtruth(
        df_imu=df_imu_sync,
        df_gt=df_gt,
        df_est=df_est_benchmark,
    )
    print(f"Angular Error Statistics (Benchmark GT Init - Pure Gyro Dead-Reckoning Drift):")
    print(f"  - Initial error at start: {eval_gt_init['errors_deg'][0]:.6f} deg (exact alignment at t0)")
    print(f"  - Mean error:             {eval_gt_init['mean_deg']:8.3f} deg ({eval_gt_init['mean_rad']:.4f} rad)")
    print(f"  - Median error:           {eval_gt_init['median_deg']:8.3f} deg ({eval_gt_init['median_rad']:.4f} rad)")
    print(f"  - RMSE:                   {eval_gt_init['rmse_deg']:8.3f} deg ({eval_gt_init['rmse_rad']:.4f} rad)")
    print(f"  - 95th Percentile:        {eval_gt_init['p95_deg']:8.3f} deg ({eval_gt_init['p95_rad']:.4f} rad)")
    print(f"  - Maximum error:          {eval_gt_init['max_deg']:8.3f} deg ({eval_gt_init['max_rad']:.4f} rad)")

    print("\n" + "=" * 80)
    print("ORIENTATION EVALUATION COMPLETE (No data files modified)")
    print("=" * 80)


if __name__ == "__main__":
    main()
