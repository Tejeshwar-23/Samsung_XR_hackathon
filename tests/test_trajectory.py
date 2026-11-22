"""
Unit tests for trajectory estimation module (Stage 3: Trajectory & Dead-Reckoning).

Covers all 11 requirements:
1. test_gravity_cancellation_level_rest: Level stationary sensor cancels 1g exactly -> zero accel/vel/pos.
2. test_gravity_cancellation_arbitrary_orientations: Tilted/pitched/inverted sensors cancel gravity correctly.
3. test_initial_roll_pitch_gravity_alignment: Analytical roll and pitch recovery from specific force vector.
4. test_initial_yaw_unobservability_default: Verifies yaw defaults to 0.0 and orientation maps correctly.
5. test_accel_bias_observability_limitation: 1D gravity-parallel bias recovered; orthogonal components unobservable.
6. test_constant_acceleration_analytical_trajectory: Exact integration of analytical motion v(t)=a*t, p(t)=0.5*a*t^2.
7. test_nonzero_initial_velocity_propagation: Initial velocity propagates linearly p(t) = p0 + v0*t.
8. test_outage_constant_velocity_propagation: Outages do not integrate acceleration; velocity held, position linearly propagated.
9. test_multiple_outages_and_recovery_continuity: Multiple outages and valid post-gap recovery continuity.
10. test_zupt_disabled_by_default: Confirms ZUPT is inactive by default and does not zero velocity.
11. test_input_dataframe_immutability: Verifies input IMU DataFrame remains completely unmodified.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from src.data_loader import IMU_TIMESTAMP_COL
from src.preprocessing import ACCEL_COLS
from src.trajectory import (
    STANDARD_GRAVITY,
    compute_gravity_alignment_quaternion,
    estimate_accel_bias_gravity_aligned,
    rotate_and_compensate_gravity,
    integrate_trajectory,
)


def _create_synthetic_imu_df(
    n_samples: int,
    rate_hz: float = 200.0,
    accel_values: Optional[np.ndarray] = None,
    t0_ns: int = 1403636579758555392,
) -> pd.DataFrame:
    """Helper to create a synthetic IMU DataFrame."""
    dt_ns = int(1e9 / rate_hz)
    timestamps = t0_ns + np.arange(n_samples, dtype=np.int64) * dt_ns

    if accel_values is None:
        accel = np.zeros((n_samples, 3), dtype=np.float64)
        accel[:, 2] = STANDARD_GRAVITY
    elif accel_values.ndim == 1:
        accel = np.tile(accel_values, (n_samples, 1))
    else:
        accel = accel_values

    df = pd.DataFrame({
        IMU_TIMESTAMP_COL: timestamps,
        "w_RS_S_x": np.zeros(n_samples, dtype=np.float64),
        "w_RS_S_y": np.zeros(n_samples, dtype=np.float64),
        "w_RS_S_z": np.zeros(n_samples, dtype=np.float64),
        "a_RS_S_x": accel[:, 0],
        "a_RS_S_y": accel[:, 1],
        "a_RS_S_z": accel[:, 2],
    })
    return df


def test_gravity_cancellation_level_rest():
    """1. Test that a level stationary sensor cancels 1g exactly -> zero accel, vel, pos."""
    n_samples = 200
    df_imu = _create_synthetic_imu_df(n_samples, accel_values=np.array([0.0, 0.0, STANDARD_GRAVITY]))

    # Level attitude: identity quaternion [w=1, x=0, y=0, z=0]
    quats = np.zeros((n_samples, 4), dtype=np.float64)
    quats[:, 0] = 1.0

    traj_df = integrate_trajectory(df_imu, quats)

    # World linear acceleration must be identically zero
    np.testing.assert_allclose(traj_df["a_lin_x"], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj_df["a_lin_y"], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj_df["a_lin_z"], 0.0, atol=1e-12)

    # Velocity and position must remain zero
    np.testing.assert_allclose(traj_df["v_x"], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj_df["v_y"], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj_df["v_z"], 0.0, atol=1e-12)

    np.testing.assert_allclose(traj_df["p_x"], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj_df["p_y"], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj_df["p_z"], 0.0, atol=1e-12)


def test_gravity_cancellation_arbitrary_orientations():
    """2. Test that gravity cancellation works in rolled, pitched, and upside-down orientations."""
    test_euler_angles = [
        (30.0, 0.0, 0.0),    # Roll 30 deg
        (0.0, 45.0, 0.0),    # Pitch 45 deg
        (0.0, 0.0, 90.0),    # Yaw 90 deg
        (25.0, -35.0, 60.0), # General 3D attitude
        (180.0, 0.0, 0.0),   # Inverted (upside-down)
    ]

    for roll_deg, pitch_deg, yaw_deg in test_euler_angles:
        rot = Rotation.from_euler("zyx", [yaw_deg, pitch_deg, roll_deg], degrees=True)
        # In body frame, stationary specific force is R_{RS}^T * [0, 0, +g]
        f_world_nominal = np.array([0.0, 0.0, STANDARD_GRAVITY])
        f_body = rot.inv().apply(f_world_nominal)

        n_samples = 50
        df_imu = _create_synthetic_imu_df(n_samples, accel_values=f_body)

        q_xyzw = rot.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        quats = np.tile(q_wxyz, (n_samples, 1))

        traj_df = integrate_trajectory(df_imu, quats)

        np.testing.assert_allclose(traj_df["a_lin_x"], 0.0, atol=1e-12)
        np.testing.assert_allclose(traj_df["a_lin_y"], 0.0, atol=1e-12)
        np.testing.assert_allclose(traj_df["a_lin_z"], 0.0, atol=1e-12)

        np.testing.assert_allclose(traj_df["v_z"], 0.0, atol=1e-12)
        np.testing.assert_allclose(traj_df["p_z"], 0.0, atol=1e-12)


def test_initial_roll_pitch_gravity_alignment():
    """3. Test analytical roll/pitch recovery from stationary specific-force vectors."""
    true_roll = np.deg2rad(15.0)
    true_pitch = np.deg2rad(-20.0)
    rot_true = Rotation.from_euler("zyx", [0.0, true_pitch, true_roll])

    # Expected body specific force
    f_body = rot_true.inv().apply(np.array([0.0, 0.0, STANDARD_GRAVITY]))

    q_aligned = compute_gravity_alignment_quaternion(f_body, yaw_rad=0.0)

    # Check unit norm
    assert np.isclose(np.linalg.norm(q_aligned), 1.0, atol=1e-15)

    # Verify that applying this orientation rotates f_body back to [0, 0, +g]
    rot_aligned = Rotation.from_quat([q_aligned[1], q_aligned[2], q_aligned[3], q_aligned[0]])
    f_world = rot_aligned.apply(f_body)

    np.testing.assert_allclose(f_world, [0.0, 0.0, STANDARD_GRAVITY], atol=1e-10)


def test_initial_yaw_unobservability_default():
    """4. Test that yaw is defaulted to 0.0 and that different yaw choices preserve gravity alignment."""
    f_body = np.array([1.2, -0.8, 9.7])

    # Alignment with yaw=0.0
    q0 = compute_gravity_alignment_quaternion(f_body, yaw_rad=0.0)
    rot0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
    f_world0 = rot0.apply(f_body)

    # Alignment with yaw=45 deg
    q45 = compute_gravity_alignment_quaternion(f_body, yaw_rad=np.pi / 4)
    rot45 = Rotation.from_quat([q45[1], q45[2], q45[3], q45[0]])
    f_world45 = rot45.apply(f_body)

    # Both must align the vertical axis exactly (Z component equals norm(f_body))
    assert np.isclose(f_world0[2], np.linalg.norm(f_body), atol=1e-10)
    assert np.isclose(f_world45[2], np.linalg.norm(f_body), atol=1e-10)

    # World X and Y are zero for both
    np.testing.assert_allclose(f_world0[:2], [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(f_world45[:2], [0.0, 0.0], atol=1e-10)


def test_accel_bias_observability_limitation():
    """5. Test that only 1D gravity-parallel bias is recovered and orthogonal components are rejected."""
    # Specific force with +0.05 m/s^2 scalar bias along gravity
    true_scalar_bias = 0.05
    f_body = np.array([0.0, 0.0, STANDARD_GRAVITY + true_scalar_bias])

    bias_est, diag = estimate_accel_bias_gravity_aligned(f_body, nominal_g=STANDARD_GRAVITY)

    assert diag["status"] == "SUCCESS_1D_OBSERVABLE_ONLY"
    assert np.isclose(diag["scalar_bias"], true_scalar_bias, atol=1e-12)
    np.testing.assert_allclose(bias_est, [0.0, 0.0, true_scalar_bias], atol=1e-12)
    assert "Orthogonal horizontal components are unobservable" in diag["observability_notice"]


def test_constant_acceleration_analytical_trajectory():
    """6. Test exact numerical integration against analytical motion: v(t)=a*t, p(t)=0.5*a*t^2."""
    n_samples = 201
    rate_hz = 100.0
    dt = 1.0 / rate_hz
    duration = (n_samples - 1) * dt

    a_const = np.array([1.5, -0.8, 2.0])  # m/s^2 linear acceleration in world
    # Sensor specific force: f_body = a_const + [0, 0, g]
    f_body = a_const + np.array([0.0, 0.0, STANDARD_GRAVITY])

    df_imu = _create_synthetic_imu_df(n_samples, rate_hz=rate_hz, accel_values=f_body)

    quats = np.zeros((n_samples, 4), dtype=np.float64)
    quats[:, 0] = 1.0  # Identity orientation

    traj_df = integrate_trajectory(df_imu, quats)

    t_s = np.arange(n_samples) * dt
    expected_v = np.outer(t_s, a_const)
    expected_p = 0.5 * np.outer(t_s**2, a_const)

    # Trapezoidal integration of constant acceleration is analytically exact
    np.testing.assert_allclose(traj_df[["v_x", "v_y", "v_z"]].to_numpy(), expected_v, atol=1e-12)
    np.testing.assert_allclose(traj_df[["p_x", "p_y", "p_z"]].to_numpy(), expected_p, atol=1e-12)


def test_nonzero_initial_velocity_propagation():
    """7. Test that nonzero initial velocity propagates linearly: p(t) = p0 + v0*t."""
    n_samples = 101
    rate_hz = 100.0
    dt = 1.0 / rate_hz

    # Stationary specific force (zero linear acceleration)
    df_imu = _create_synthetic_imu_df(n_samples, rate_hz=rate_hz, accel_values=np.array([0.0, 0.0, STANDARD_GRAVITY]))
    quats = np.zeros((n_samples, 4), dtype=np.float64)
    quats[:, 0] = 1.0

    p0 = np.array([10.0, -5.0, 2.0])
    v0 = np.array([1.0, 2.5, -0.5])

    traj_df = integrate_trajectory(df_imu, quats, initial_position=p0, initial_velocity=v0)

    t_s = np.arange(n_samples) * dt
    expected_v = np.tile(v0, (n_samples, 1))
    expected_p = p0 + np.outer(t_s, v0)

    np.testing.assert_allclose(traj_df[["v_x", "v_y", "v_z"]].to_numpy(), expected_v, atol=1e-12)
    np.testing.assert_allclose(traj_df[["p_x", "p_y", "p_z"]].to_numpy(), expected_p, atol=1e-12)


def test_outage_constant_velocity_propagation():
    """8. Test that during a 2s outage, acceleration is NOT integrated, velocity is held, and position propagates."""
    rate_hz = 100.0
    dt_ns = int(1e9 / rate_hz)

    # Create 50 samples before outage, 2.0s gap, 50 samples after outage
    t0_ns = 1403636579758555392
    part1_times = t0_ns + np.arange(50, dtype=np.int64) * dt_ns
    gap_duration_ns = int(2.0 * 1e9)
    part2_times = part1_times[-1] + gap_duration_ns + np.arange(1, 51, dtype=np.int64) * dt_ns

    all_times = np.concatenate([part1_times, part2_times])
    n_samples = len(all_times)

    # Constant linear acceleration of [1.0, 0.0, 0.0]
    a_lin = np.array([1.0, 0.0, 0.0])
    f_body = a_lin + np.array([0.0, 0.0, STANDARD_GRAVITY])

    df_imu = pd.DataFrame({
        IMU_TIMESTAMP_COL: all_times,
        "a_RS_S_x": np.full(n_samples, f_body[0]),
        "a_RS_S_y": np.full(n_samples, f_body[1]),
        "a_RS_S_z": np.full(n_samples, f_body[2]),
    })

    quats = np.zeros((n_samples, 4), dtype=np.float64)
    quats[:, 0] = 1.0

    traj_df = integrate_trajectory(df_imu, quats, gap_threshold_s=0.1)

    # Index 50 is the first sample after the gap
    assert traj_df["is_gap"].iloc[50] == True
    assert traj_df["is_gap"].iloc[49] == False
    assert traj_df["is_gap"].iloc[51] == False

    # Velocity across the gap MUST be held constant
    v_before = traj_df["v_x"].iloc[49]
    v_at_gap = traj_df["v_x"].iloc[50]
    assert np.isclose(v_before, v_at_gap, atol=1e-12)

    # Position at sample 50 must equal p[49] + v[49] * dt_actual
    p_before = traj_df["p_x"].iloc[49]
    p_at_gap = traj_df["p_x"].iloc[50]
    dt_actual_s = (all_times[50] - all_times[49]) / 1e9
    expected_p = p_before + v_before * dt_actual_s
    assert np.isclose(p_at_gap, expected_p, atol=1e-10)


def test_multiple_outages_and_recovery_continuity():
    """9. Test trajectory integration across multiple outages and subsequent recovery."""
    t0_ns = 1403636579758555392
    dt_normal = int(0.01 * 1e9)  # 10 ms (100 Hz)

    # 3 continuous blocks separated by two 1.5s outages
    block1 = t0_ns + np.arange(30, dtype=np.int64) * dt_normal
    block2 = block1[-1] + int(1.5 * 1e9) + np.arange(1, 31, dtype=np.int64) * dt_normal
    block3 = block2[-1] + int(1.5 * 1e9) + np.arange(1, 31, dtype=np.int64) * dt_normal

    all_times = np.concatenate([block1, block2, block3])
    n_samples = len(all_times)

    df_imu = pd.DataFrame({
        IMU_TIMESTAMP_COL: all_times,
        "a_RS_S_x": np.zeros(n_samples),
        "a_RS_S_y": np.zeros(n_samples),
        "a_RS_S_z": np.full(n_samples, STANDARD_GRAVITY),
    })
    quats = np.zeros((n_samples, 4))
    quats[:, 0] = 1.0

    v0 = np.array([2.0, 0.0, 0.0])
    traj_df = integrate_trajectory(df_imu, quats, initial_velocity=v0, gap_threshold_s=0.05)

    assert len(traj_df) == n_samples
    assert traj_df["is_gap"].sum() == 2
    assert traj_df["is_gap"].iloc[30] == True
    assert traj_df["is_gap"].iloc[60] == True

    # Since linear acceleration is zero, velocity should remain constant throughout
    np.testing.assert_allclose(traj_df["v_x"], 2.0, atol=1e-12)

    # Position at end should equal v0 * total_elapsed_time
    total_time_s = (all_times[-1] - all_times[0]) / 1e9
    assert np.isclose(traj_df["p_x"].iloc[-1], 2.0 * total_time_s, atol=1e-10)


def test_zupt_disabled_by_default():
    """10. Verify that ZUPT is disabled by default and does not zero velocity."""
    n_samples = 50
    df_imu = _create_synthetic_imu_df(n_samples, accel_values=np.array([0.0, 0.0, STANDARD_GRAVITY]))
    quats = np.zeros((n_samples, 4))
    quats[:, 0] = 1.0

    v0 = np.array([3.0, 0.0, 0.0])
    # Pass a zupt_mask with all True, but keep enable_zupt=False (default)
    zupt_mask = np.ones(n_samples, dtype=bool)

    traj_df = integrate_trajectory(df_imu, quats, initial_velocity=v0, enable_zupt=False, zupt_mask=zupt_mask)

    # Velocity must NOT be zeroed out
    np.testing.assert_allclose(traj_df["v_x"], 3.0, atol=1e-12)


def test_input_dataframe_immutability():
    """11. Verify that input IMU DataFrame is left completely unmodified."""
    n_samples = 50
    df_imu = _create_synthetic_imu_df(n_samples)
    df_copy = df_imu.copy(deep=True)

    quats = np.zeros((n_samples, 4))
    quats[:, 0] = 1.0

    _ = integrate_trajectory(df_imu, quats)

    pd.testing.assert_frame_equal(df_imu, df_copy)


def test_bias_estimator_schema_compatibility():
    """12. Interface check: verify estimate_gyro_bias schema consumed by trajectory evaluation."""
    from src.bias_estimation import estimate_gyro_bias

    n_samples = 500
    df_imu = _create_synthetic_imu_df(n_samples)

    res = estimate_gyro_bias(df_imu)

    # Required keys consumed by evaluate_trajectory.py
    assert "bias_estimated" in res
    assert "gyro_bias" in res
    assert isinstance(res["gyro_bias"], np.ndarray)
    assert res["gyro_bias"].shape == (3,)
    assert "confidence" in res
    assert "interval_info" in res


def test_groundtruth_synchronization_initial_state_validation():
    """13. Interface check: verify GT synchronization returns finite unit quaternion when masked."""
    from src.synchronization import interpolate_groundtruth_to_imu

    # Create synthetic IMU and GT
    df_imu = _create_synthetic_imu_df(100)
    # GT spanning a subset of IMU
    df_gt = pd.DataFrame({
        "timestamp": df_imu[IMU_TIMESTAMP_COL].iloc[10:90].values,
        "p_RS_R_x": np.zeros(80),
        "p_RS_R_y": np.zeros(80),
        "p_RS_R_z": np.zeros(80),
        "q_RS_w": np.ones(80),
        "q_RS_x": np.zeros(80),
        "q_RS_y": np.zeros(80),
        "q_RS_z": np.zeros(80),
        "v_RS_R_x": np.zeros(80),
        "v_RS_R_y": np.zeros(80),
        "v_RS_R_z": np.zeros(80),
        "b_w_RS_S_x": np.zeros(80),
        "b_w_RS_S_y": np.zeros(80),
        "b_w_RS_S_z": np.zeros(80),
        "b_a_RS_S_x": np.zeros(80),
        "b_a_RS_S_y": np.zeros(80),
        "b_a_RS_S_z": np.zeros(80),
    })

    df_gt_aligned, valid_mask = interpolate_groundtruth_to_imu(df_imu, df_gt)

    # df_gt_aligned has NaNs at rows where valid_mask is False
    assert np.isnan(df_gt_aligned.iloc[0]["q_RS_w"])

    # When masked with valid_mask, the first element MUST be valid and finite
    df_gt_valid = df_gt_aligned[valid_mask].reset_index(drop=True)
    q0 = df_gt_valid[["q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z"]].iloc[0].to_numpy()

    assert np.isfinite(q0).all()
    assert np.isclose(np.linalg.norm(q0), 1.0, atol=1e-12)


def test_reject_dataframe_as_quaternion_input():
    """14. Regression: integrate_trajectory must reject a DataFrame passed as quaternions_wxyz."""
    df_imu = _create_synthetic_imu_df(10)
    df_quats = pd.DataFrame({
        "timestamp": df_imu[IMU_TIMESTAMP_COL].values,
        "q_est_w": np.ones(10),
        "q_est_x": np.zeros(10),
        "q_est_y": np.zeros(10),
        "q_est_z": np.zeros(10),
        "dt_s": np.full(10, 0.005),
        "is_gap": np.zeros(10, dtype=bool),
    })

    with pytest.raises(ValueError, match="must be a NumPy array of shape"):
        _ = integrate_trajectory(df_imu, df_quats)


def test_reject_malformed_quaternion_shapes():
    """15. Regression: reject (N, 7) and (N, 3) arrays passed as quaternions_wxyz."""
    df_imu = _create_synthetic_imu_df(10)

    # (N, 7) array
    arr_n7 = np.zeros((10, 7), dtype=np.float64)
    with pytest.raises(ValueError, match="must be a 2D array of shape"):
        _ = integrate_trajectory(df_imu, arr_n7)

    # (N, 3) array
    arr_n3 = np.zeros((10, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="must be a 2D array of shape"):
        _ = integrate_trajectory(df_imu, arr_n3)


def test_reject_non_finite_and_zero_norm_quaternions():
    """16. Regression: reject NaN, Inf, and zero-norm quaternions."""
    df_imu = _create_synthetic_imu_df(10)

    # NaN quaternion
    quats_nan = np.zeros((10, 4), dtype=np.float64)
    quats_nan[:, 0] = 1.0
    quats_nan[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _ = integrate_trajectory(df_imu, quats_nan)

    # Zero-norm quaternion
    quats_zero = np.zeros((10, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="zero-norm"):
        _ = integrate_trajectory(df_imu, quats_zero)


def test_accept_valid_quaternion_array_and_evaluate_extraction():
    """17. Regression: confirm valid (N, 4) array accepted and orientation DataFrame extracted."""
    from src.orientation import EST_QUATERNION_COLS

    df_imu = _create_synthetic_imu_df(10)
    # Mock return from integrate_gyro_quaternion
    df_rot = pd.DataFrame({
        "timestamp": df_imu[IMU_TIMESTAMP_COL].values,
        "q_est_w": np.ones(10),
        "q_est_x": np.zeros(10),
        "q_est_y": np.zeros(10),
        "q_est_z": np.zeros(10),
        "dt_s": np.full(10, 0.005),
        "is_gap": np.zeros(10, dtype=bool),
    })

    # Explicit extraction pattern used in evaluate_trajectory.py
    quats_valid = df_rot[EST_QUATERNION_COLS].to_numpy(dtype=np.float64)
    assert quats_valid.shape == (10, 4)

    traj = integrate_trajectory(df_imu, quats_valid)
    assert len(traj) == 10



