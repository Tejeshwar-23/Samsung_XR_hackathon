"""
Evaluation and benchmark script for IMU trajectory dead-reckoning on EuRoC MH_01_easy.

Evaluates 4 comparative cases:
Case A: Autonomous IMU-Only (Identity orientation, zero initial velocity, uncorrected gyro).
Case B: Autonomous Bias-Corrected (Autonomous gyro bias corrected, gravity aligned if stationary, zero initial velocity).
Case C: Ground-Truth Initialized Benchmark (Uncorrected gyro, initialized with GT q(t0) and v(t0)).
Case D: Ground-Truth Initialized + Bias-Corrected Benchmark (Autonomous gyro bias corrected, initialized with GT q(t0) and v(t0)).

Evaluation Metrics:
1. Primary Metric: Raw Unaligned ATE (Absolute Trajectory Error) over time:
   e_{p, k} = ||p_{est, k} - p_{gt, k}||
2. Secondary Metric: Initial-Pose Aligned ATE (aligned strictly at t0):
   p_{aligned, k} = R_align * p_{est, k} + t_align
3. Velocity RMSE:
   RMSE_v = sqrt( (1/N) * sum ||v_{est, k} - v_{gt, k}||^2 )
4. Relative Pose Error (RPE) for fixed interval Delta T = 1.0 s:
   e_{RPE, k} = || (p_{est, k} - p_{est, k-1s}) - (p_{gt, k} - p_{gt, k-1s}) ||

Strict Separation:
- Ground-truth data is used strictly for offline synchronization and evaluation scoring.
- Autonomous estimators (Cases A & B) receive zero ground-truth information.
- Plots are saved to outputs/plots/trajectory_dead_reckoning_eval.png.
"""

from pathlib import Path
import sys
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import (
    load_imu_data,
    load_groundtruth_data,
    IMU_TIMESTAMP_COL,
    GT_POSITION_COLS,
    GT_VELOCITY_COLS,
    GT_QUATERNION_COLS,
)
from src.bias_estimation import estimate_gyro_bias
from src.synchronization import interpolate_groundtruth_to_imu
from src.orientation import integrate_gyro_quaternion, EST_QUATERNION_COLS
from src.trajectory import (
    compute_gravity_alignment_quaternion,
    estimate_accel_bias_gravity_aligned,
    integrate_trajectory,
    STANDARD_GRAVITY,
)


def compute_ate_unaligned(p_est: np.ndarray, p_gt: np.ndarray) -> np.ndarray:
    """Computes sample-by-sample Euclidean position error without alignment."""
    return np.linalg.norm(p_est - p_gt, axis=1)


def compute_ate_initial_aligned(
    p_est: np.ndarray,
    p_gt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes position error after rigidly translating the trajectory so p_est(t0) == p_gt(t0).
    Does not distort dead-reckoning trajectory geometry.
    """
    t_align = p_gt[0] - p_est[0]
    p_aligned = p_est + t_align
    ate_aligned = np.linalg.norm(p_aligned - p_gt, axis=1)
    return ate_aligned, p_aligned


def compute_velocity_rmse(v_est: np.ndarray, v_gt: np.ndarray) -> float:
    """Computes 3D Velocity Root Mean Squared Error."""
    diff = v_est - v_gt
    sq_err = np.sum(diff**2, axis=1)
    return float(np.sqrt(np.mean(sq_err)))


def compute_rpe_1s(
    p_est: np.ndarray,
    p_gt: np.ndarray,
    timestamps_ns: np.ndarray,
    delta_t_s: float = 1.0,
) -> np.ndarray:
    """
    Computes Relative Pose / Displacement Error for a fixed time interval delta_t_s.
    """
    t_s = (timestamps_ns - timestamps_ns[0]) / 1e9
    rpe_list = []

    dt_step = np.median(np.diff(t_s))
    steps = max(1, int(round(delta_t_s / dt_step)))

    if len(p_est) <= steps:
        return np.array([0.0])

    disp_est = p_est[steps:] - p_est[:-steps]
    disp_gt = p_gt[steps:] - p_gt[:-steps]
    rpe = np.linalg.norm(disp_est - disp_gt, axis=1)
    return rpe


def run_trajectory_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("SAMSUNG XR HACKATHON TRACK 3 - STAGE 3 TRAJECTORY EVALUATION")
    print("=" * 80)

    imu_path = project_root / "data/raw/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv"
    gt_path = (
        project_root
        / "data/raw/euroc/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    )

    print(f"Loading IMU data: {imu_path}")
    df_imu = load_imu_data(str(imu_path))
    print(f"Loading Ground Truth: {gt_path}")
    df_gt = load_groundtruth_data(str(gt_path))

    # Synchronize Ground Truth to IMU query timestamps
    print("\nSynchronizing Ground Truth to IMU timestamps (strictly for evaluation)...")
    df_gt_aligned, valid_mask = interpolate_groundtruth_to_imu(df_imu, df_gt)
    print(f"Valid synchronized samples: {valid_mask.sum()}/{len(valid_mask)}")

    # Extract strictly valid synchronized rows
    df_imu_sync = df_imu[valid_mask].copy().reset_index(drop=True)
    df_gt_sync = df_gt_aligned[valid_mask].copy().reset_index(drop=True)

    timestamps_ns = df_imu_sync[IMU_TIMESTAMP_COL].to_numpy(dtype=np.int64)
    t_s = (timestamps_ns - timestamps_ns[0]) / 1e9

    p_gt = df_gt_sync[GT_POSITION_COLS].to_numpy(dtype=np.float64)
    v_gt = df_gt_sync[GT_VELOCITY_COLS].to_numpy(dtype=np.float64)
    q_gt_wxyz = df_gt_sync[GT_QUATERNION_COLS].to_numpy(dtype=np.float64)

    # Validate ground-truth initial state at first valid synchronized timestamp
    q0_gt = q_gt_wxyz[0]
    p0_gt = p_gt[0]
    v0_gt = v_gt[0]

    q0_norm = float(np.linalg.norm(q0_gt))
    if not np.isfinite(q0_gt).all() or q0_norm < 1e-6:
        raise ValueError(
            f"Invalid initial ground-truth quaternion at timestamp {timestamps_ns[0]}: "
            f"q0={q0_gt}, norm={q0_norm}"
        )
    # Ensure exact unit norm
    q0_gt = q0_gt / q0_norm

    print(f"First synchronized timestamp: {timestamps_ns[0]} ns")
    print(f"Ground-Truth Initial Quaternion (wxyz): {q0_gt} (norm: {q0_norm:.6f})")
    print(f"Ground-Truth Initial Velocity:          {v0_gt} m/s")
    print(f"Ground-Truth Initial Position:          {p0_gt} m")

    # 1. Run Autonomous Gyro Bias Estimation (IMU-only)
    print("\nRunning autonomous IMU-only gyro bias estimation...")
    bias_res = estimate_gyro_bias(df_imu_sync)
    gyro_bias_est = bias_res["gyro_bias"]
    print(f"Estimated Gyro Bias: {gyro_bias_est} rad/s (Confidence: {bias_res['confidence']})")

    # Correct gyro for Cases B & D
    df_imu_corrected = df_imu_sync.copy()
    df_imu_corrected["w_RS_S_x"] -= gyro_bias_est[0]
    df_imu_corrected["w_RS_S_y"] -= gyro_bias_est[1]
    df_imu_corrected["w_RS_S_z"] -= gyro_bias_est[2]

    # Evaluate Cases:
    # -------------------------------------------------------------
    # CASE A: Autonomous IMU-Only (Identity orientation, zero initial velocity, uncorrected gyro)
    # -------------------------------------------------------------
    print("\n[Running Case A] Autonomous IMU-Only (Identity init, v0=0, uncorrected gyro)...")
    df_q_A = integrate_gyro_quaternion(df_imu_sync, initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]))
    q_est_A = df_q_A[EST_QUATERNION_COLS].to_numpy(dtype=np.float64)
    traj_A = integrate_trajectory(df_imu_sync, q_est_A, initial_position=np.zeros(3), initial_velocity=np.zeros(3))
    p_A = traj_A[["p_x", "p_y", "p_z"]].to_numpy()
    v_A = traj_A[["v_x", "v_y", "v_z"]].to_numpy()

    # -------------------------------------------------------------
    # CASE B: Autonomous Bias-Corrected (Autonomous gyro bias, gravity-aligned from stationary hover if available, v0=0)
    # -------------------------------------------------------------
    print("\n[Running Case B] Autonomous Bias-Corrected (Bias-corrected gyro, v0=0)...")
    # For Case B autonomous initialization, check if stationary intervals exist
    if bias_res["bias_estimated"] and bias_res["interval_info"] is not None:
        # Hover interval mean accel from the stationary interval
        start_idx = bias_res["interval_info"]["start_index"]
        end_idx = bias_res["interval_info"]["end_index"]
        f_hover = df_imu_sync.iloc[start_idx : end_idx + 1][["a_RS_S_x", "a_RS_S_y", "a_RS_S_z"]].mean().to_numpy()
        q0_B = compute_gravity_alignment_quaternion(f_hover, yaw_rad=0.0)
    else:
        q0_B = np.array([1.0, 0.0, 0.0, 0.0])

    df_q_B = integrate_gyro_quaternion(df_imu_corrected, initial_quaternion=q0_B)
    q_est_B = df_q_B[EST_QUATERNION_COLS].to_numpy(dtype=np.float64)
    traj_B = integrate_trajectory(df_imu_corrected, q_est_B, initial_position=np.zeros(3), initial_velocity=np.zeros(3))
    p_B = traj_B[["p_x", "p_y", "p_z"]].to_numpy()
    v_B = traj_B[["v_x", "v_y", "v_z"]].to_numpy()

    # -------------------------------------------------------------
    # CASE C: Ground-Truth Initialized Benchmark (Uncorrected gyro, initialized with GT q(t0) and v(t0))
    # -------------------------------------------------------------
    print("\n[Running Case C] GT-Initialized Benchmark (Uncorrected gyro, q0=GT, v0=GT)...")
    df_q_C = integrate_gyro_quaternion(df_imu_sync, initial_quaternion=q0_gt)
    q_est_C = df_q_C[EST_QUATERNION_COLS].to_numpy(dtype=np.float64)
    traj_C = integrate_trajectory(df_imu_sync, q_est_C, initial_position=p0_gt, initial_velocity=v0_gt)
    p_C = traj_C[["p_x", "p_y", "p_z"]].to_numpy()
    v_C = traj_C[["v_x", "v_y", "v_z"]].to_numpy()

    # -------------------------------------------------------------
    # CASE D: Ground-Truth Initialized + Bias-Corrected Benchmark (Autonomous gyro bias, initialized with GT q(t0) and v(t0))
    # -------------------------------------------------------------
    print("\n[Running Case D] GT-Initialized + Bias-Corrected (Bias-corrected gyro, q0=GT, v0=GT)...")
    df_q_D = integrate_gyro_quaternion(df_imu_corrected, initial_quaternion=q0_gt)
    q_est_D = df_q_D[EST_QUATERNION_COLS].to_numpy(dtype=np.float64)
    traj_D = integrate_trajectory(df_imu_corrected, q_est_D, initial_position=p0_gt, initial_velocity=v0_gt)
    p_D = traj_D[["p_x", "p_y", "p_z"]].to_numpy()
    v_D = traj_D[["v_x", "v_y", "v_z"]].to_numpy()

    # -------------------------------------------------------------
    # Metrics Computation
    # -------------------------------------------------------------
    cases = [
        ("Case A (Autonomous Uncorrected)", p_A, v_A),
        ("Case B (Autonomous Bias-Corrected)", p_B, v_B),
        ("Case C (GT-Init Uncorrected)", p_C, v_C),
        ("Case D (GT-Init Bias-Corrected)", p_D, v_D),
    ]

    results = {}
    print("\n" + "=" * 80)
    print("QUANTITATIVE TRAJECTORY EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Case':<35} | {'Raw ATE Mean (m)':<17} | {'Raw ATE Max (m)':<15} | {'Init-Aligned ATE (m)':<20} | {'Vel RMSE (m/s)':<15}")
    print("-" * 110)

    for label, p_est, v_est in cases:
        ate_raw = compute_ate_unaligned(p_est, p_gt)
        ate_aligned, p_aligned = compute_ate_initial_aligned(p_est, p_gt)
        v_rmse = compute_velocity_rmse(v_est, v_gt)
        rpe_1s = compute_rpe_1s(p_est, p_gt, timestamps_ns, delta_t_s=1.0)

        results[label] = {
            "ate_raw_mean": float(np.mean(ate_raw)),
            "ate_raw_rmse": float(np.sqrt(np.mean(ate_raw**2))),
            "ate_raw_max": float(np.max(ate_raw)),
            "ate_aligned_mean": float(np.mean(ate_aligned)),
            "ate_aligned_rmse": float(np.sqrt(np.mean(ate_aligned**2))),
            "ate_aligned_max": float(np.max(ate_aligned)),
            "v_rmse": v_rmse,
            "rpe_1s_mean": float(np.mean(rpe_1s)),
            "rpe_1s_max": float(np.max(rpe_1s)),
        }

        print(
            f"{label:<35} | "
            f"{results[label]['ate_raw_mean']:<17.3f} | "
            f"{results[label]['ate_raw_max']:<15.3f} | "
            f"{results[label]['ate_aligned_mean']:<20.3f} | "
            f"{v_rmse:<15.3f}"
        )

    print("=" * 80)
    print("1-Second RPE (Relative Pose Error) Summary:")
    for label in results:
        print(f"  {label:<35}: Mean = {results[label]['rpe_1s_mean']:.3f} m, Max = {results[label]['rpe_1s_max']:.3f} m")

    # Detailed Forensic & Physical State Diagnostics
    print("\n" + "=" * 80)
    print("PHYSICAL & INTERMEDIATE STATE DIAGNOSTICS")
    print("=" * 80)

    # World-frame accelerations for Case B and Case D
    a_lin_B = traj_B[["a_lin_x", "a_lin_y", "a_lin_z"]].to_numpy()
    a_lin_D = traj_D[["a_lin_x", "a_lin_y", "a_lin_z"]].to_numpy()

    print("\nFirst 10 World-Frame Linear Accelerations after Gravity Subtraction (Case D, m/s^2):")
    for idx in range(min(10, len(a_lin_D))):
        print(f"  t={t_s[idx]:.4f}s: a_lin=[{a_lin_D[idx, 0]:+8.4f}, {a_lin_D[idx, 1]:+8.4f}, {a_lin_D[idx, 2]:+8.4f}] m/s^2")

    print("\nAcceleration Statistics across Full Flight (Case D, m/s^2):")
    print(f"  X-axis: mean={np.mean(a_lin_D[:, 0]):+8.4f}, std={np.std(a_lin_D[:, 0]):8.4f}, min={np.min(a_lin_D[:, 0]):+8.4f}, max={np.max(a_lin_D[:, 0]):+8.4f}")
    print(f"  Y-axis: mean={np.mean(a_lin_D[:, 1]):+8.4f}, std={np.std(a_lin_D[:, 1]):8.4f}, min={np.min(a_lin_D[:, 1]):+8.4f}, max={np.max(a_lin_D[:, 1]):+8.4f}")
    print(f"  Z-axis: mean={np.mean(a_lin_D[:, 2]):+8.4f}, std={np.std(a_lin_D[:, 2]):8.4f}, min={np.min(a_lin_D[:, 2]):+8.4f}, max={np.max(a_lin_D[:, 2]):+8.4f}")

    print("\nFinal State Comparison at End of Flight:")
    final_states = [("Case A", p_A, v_A), ("Case B", p_B, v_B), ("Case C", p_C, v_C), ("Case D", p_D, v_D)]
    for name, p_arr, v_arr in final_states:
        print(f"  {name:<8}: Final Pos = [{p_arr[-1, 0]:10.2f}, {p_arr[-1, 1]:10.2f}, {p_arr[-1, 2]:10.2f}] m | Final Vel = [{v_arr[-1, 0]:8.2f}, {v_arr[-1, 1]:8.2f}, {v_arr[-1, 2]:8.2f}] m/s")

    print(f"  GT Final: Final Pos = [{p_gt[-1, 0]:10.2f}, {p_gt[-1, 1]:10.2f}, {p_gt[-1, 2]:10.2f}] m | Final Vel = [{v_gt[-1, 0]:8.2f}, {v_gt[-1, 1]:8.2f}, {v_gt[-1, 2]:8.2f}] m/s")

    diff_AB = np.linalg.norm(p_A - p_B)
    diff_CD = np.linalg.norm(p_C - p_D)
    print(f"\nTrajectory Discrepancy Checks:")
    print(f"  Difference between Case A and Case B trajectories: {diff_AB:.4f} m (Distinct: {diff_AB > 1e-4})")
    print(f"  Difference between Case C and Case D trajectories: {diff_CD:.4f} m (Distinct: {diff_CD > 1e-4})")

    # Generate Plots
    output_dir = project_root / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "trajectory_dead_reckoning_eval.png"

    print(f"\nGenerating evaluation plot: {plot_file}")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Subplot 1: Raw Unaligned ATE
    axes[0].plot(t_s, compute_ate_unaligned(p_A, p_gt), label="Case A (Autonomous)", color="gray", linestyle=":")
    axes[0].plot(t_s, compute_ate_unaligned(p_B, p_gt), label="Case B (Autonomous Bias-Corr)", color="orange", linestyle="--")
    axes[0].plot(t_s, compute_ate_unaligned(p_C, p_gt), label="Case C (GT-Init Uncorrected)", color="blue", alpha=0.7)
    axes[0].plot(t_s, compute_ate_unaligned(p_D, p_gt), label="Case D (GT-Init Bias-Corr)", color="green", linewidth=2.0)
    axes[0].set_ylabel("Raw ATE Error (m)")
    axes[0].set_title("Strapdown IMU Dead-Reckoning: Raw Absolute Trajectory Error (ATE)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")

    # Subplot 2: Initial-Pose Aligned ATE
    axes[1].plot(t_s, compute_ate_initial_aligned(p_A, p_gt)[0], label="Case A (Aligned)", color="gray", linestyle=":")
    axes[1].plot(t_s, compute_ate_initial_aligned(p_B, p_gt)[0], label="Case B (Aligned)", color="orange", linestyle="--")
    axes[1].plot(t_s, compute_ate_initial_aligned(p_C, p_gt)[0], label="Case C (Aligned)", color="blue", alpha=0.7)
    axes[1].plot(t_s, compute_ate_initial_aligned(p_D, p_gt)[0], label="Case D (Aligned)", color="green", linewidth=2.0)
    axes[1].set_ylabel("Aligned ATE Error (m)")
    axes[1].set_title("Initial-Pose Aligned ATE (p_est(t0) aligned to p_gt(t0))")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left")

    # Subplot 3: Velocity Norm Error
    v_err_A = np.linalg.norm(v_A - v_gt, axis=1)
    v_err_B = np.linalg.norm(v_B - v_gt, axis=1)
    v_err_C = np.linalg.norm(v_C - v_gt, axis=1)
    v_err_D = np.linalg.norm(v_D - v_gt, axis=1)

    axes[2].plot(t_s, v_err_A, label="Case A (Autonomous)", color="gray", linestyle=":")
    axes[2].plot(t_s, v_err_B, label="Case B (Autonomous Bias-Corr)", color="orange", linestyle="--")
    axes[2].plot(t_s, v_err_C, label="Case C (GT-Init Uncorrected)", color="blue", alpha=0.7)
    axes[2].plot(t_s, v_err_D, label="Case D (GT-Init Bias-Corr)", color="green", linewidth=2.0)
    axes[2].set_xlabel("Time from flight start (s)")
    axes[2].set_ylabel("Velocity Error (m/s)")
    axes[2].set_title("3D Velocity Error Norm Over Time")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(str(plot_file), dpi=200)
    plt.close()
    print(f"Plot successfully saved to: {plot_file}")

    return results


if __name__ == "__main__":
    run_trajectory_evaluation()
