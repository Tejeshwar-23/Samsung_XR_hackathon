"""
Unit tests for src/orientation.py.

Tests cover:
- Zero angular velocity preserves initial orientation.
- Constant angular velocity produces expected rotation angle theta = omega * t.
- Independent rotations about X, Y, and Z axes.
- Quaternion normalization: unit norms maintained throughout integration.
- Irregular timestamp intervals (variable dt).
- Large timestamp gap handling (gap policy tests).
- Configurable initial quaternion support.
- Ground truth orientation error calculation (zero error when identical, exact angle error when rotated).
- Original IMU DataFrame immutability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from src.orientation import (
    integrate_gyro_quaternion,
    compute_angular_error,
    evaluate_orientation_against_groundtruth,
    EST_QUATERNION_COLS,
)
from src.data_loader import GT_QUATERNION_COLS


@pytest.fixture
def synthetic_constant_rate_imu() -> pd.DataFrame:
    """Generate 201 samples (1.0 second at 200 Hz) with constant yaw rotation (w_z = 1.0 rad/s)."""
    n_samples = 201
    dt_ns = 5_000_000  # 5 ms
    t0 = 1_000_000_000_000
    timestamps = np.arange(t0, t0 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    return pd.DataFrame({
        "timestamp": timestamps,
        "w_RS_S_x": np.zeros(n_samples),
        "w_RS_S_y": np.zeros(n_samples),
        "w_RS_S_z": np.full(n_samples, 1.0),
        "a_RS_S_x": np.zeros(n_samples),
        "a_RS_S_y": np.zeros(n_samples),
        "a_RS_S_z": np.full(n_samples, 9.81),
    })


# ==============================================================================
# TESTS
# ==============================================================================


def test_zero_angular_velocity_preserves_orientation():
    """Verify that zero angular velocity preserves initial orientation exactly."""
    n_samples = 100
    dt_ns = 5_000_000
    t0 = 100_000_000
    timestamps = np.arange(t0, t0 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    df_imu = pd.DataFrame({
        "timestamp": timestamps,
        "w_RS_S_x": np.zeros(n_samples),
        "w_RS_S_y": np.zeros(n_samples),
        "w_RS_S_z": np.zeros(n_samples),
    })

    # Non-trivial initial quaternion
    q0 = np.array([0.5, 0.5, 0.5, 0.5])
    df_est = integrate_gyro_quaternion(df_imu, initial_quaternion=q0)

    for col in EST_QUATERNION_COLS:
        # All rows should equal initial value
        np.testing.assert_allclose(df_est[col], q0[EST_QUATERNION_COLS.index(col)], atol=1e-12)


def test_constant_angular_velocity(synthetic_constant_rate_imu: pd.DataFrame):
    """Verify that constant angular rate integrates to theta = omega * t."""
    df_est = integrate_gyro_quaternion(synthetic_constant_rate_imu)

    # After 1.0 s with w_z = 1.0 rad/s, angle should be exactly 1.0 rad
    t0 = synthetic_constant_rate_imu["timestamp"].iloc[0]
    t_s = (synthetic_constant_rate_imu["timestamp"] - t0) * 1e-9

    quats_wxyz = df_est[EST_QUATERNION_COLS].to_numpy()
    # Expected quaternion for rotation about Z: [cos(theta/2), 0, 0, sin(theta/2)]
    expected_w = np.cos(0.5 * t_s)
    expected_z = np.sin(0.5 * t_s)

    np.testing.assert_allclose(quats_wxyz[:, 0], expected_w, atol=1e-5)
    np.testing.assert_allclose(quats_wxyz[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(quats_wxyz[:, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(quats_wxyz[:, 3], expected_z, atol=1e-5)


@pytest.mark.parametrize("axis,col", [("x", "w_RS_S_x"), ("y", "w_RS_S_y"), ("z", "w_RS_S_z")])
def test_single_axis_rotations(axis: str, col: str):
    """Test independent rotations about X, Y, and Z axes."""
    n_samples = 51
    dt_ns = 10_000_000  # 10 ms -> 0.5 seconds
    t0 = 500_000_000
    timestamps = np.arange(t0, t0 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    data = {
        "timestamp": timestamps,
        "w_RS_S_x": np.zeros(n_samples),
        "w_RS_S_y": np.zeros(n_samples),
        "w_RS_S_z": np.zeros(n_samples),
    }
    rate = 2.0  # 2.0 rad/s
    data[col] = np.full(n_samples, rate)
    df_imu = pd.DataFrame(data)

    df_est = integrate_gyro_quaternion(df_imu)
    q_final = df_est[EST_QUATERNION_COLS].iloc[-1].to_numpy()

    # After 0.5 s at 2 rad/s -> angle is 1.0 rad
    expected_rot = Rotation.from_euler(axis, 1.0)
    expected_xyzw = expected_rot.as_quat()
    expected_wxyz = np.array([expected_xyzw[3], expected_xyzw[0], expected_xyzw[1], expected_xyzw[2]])

    np.testing.assert_allclose(np.abs(np.dot(q_final, expected_wxyz)), 1.0, atol=1e-5)


def test_quaternion_normalization(synthetic_constant_rate_imu: pd.DataFrame):
    """Verify that all estimated quaternions have strict unit norm ||q|| == 1.0."""
    df_est = integrate_gyro_quaternion(synthetic_constant_rate_imu)
    quats = df_est[EST_QUATERNION_COLS].to_numpy()
    norms = np.linalg.norm(quats, axis=1)

    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_irregular_timestamp_intervals():
    """Verify integration works correctly on non-uniform, irregular sampling intervals."""
    # Irregular intervals varying from 2 ms to 15 ms
    rng = np.random.default_rng(42)
    dt_ms = rng.uniform(2.0, 15.0, size=100)
    dt_ns = (dt_ms * 1e6).astype(np.int64)
    timestamps = np.cumsum(np.insert(dt_ns, 0, 1_000_000_000))

    rate = 0.5  # rad/s around Y
    df_imu = pd.DataFrame({
        "timestamp": timestamps,
        "w_RS_S_x": np.zeros(len(timestamps)),
        "w_RS_S_y": np.full(len(timestamps), rate),
        "w_RS_S_z": np.zeros(len(timestamps)),
    })

    # Setting gap_threshold_s above maximum interval (20 ms) tests irregular dt integration
    df_est = integrate_gyro_quaternion(df_imu, gap_threshold_s=0.05)
    total_time_s = (timestamps[-1] - timestamps[0]) * 1e-9
    expected_angle = rate * total_time_s

    q_final = df_est[EST_QUATERNION_COLS].iloc[-1].to_numpy()
    expected_rot = Rotation.from_euler("y", expected_angle)
    expected_xyzw = expected_rot.as_quat()
    expected_wxyz = np.array([expected_xyzw[3], expected_xyzw[0], expected_xyzw[1], expected_xyzw[2]])

    np.testing.assert_allclose(np.abs(np.dot(q_final, expected_wxyz)), 1.0, atol=1e-5)


def test_large_timestamp_gap_handling():
    """Verify behavior when a large gap occurs with different gap policies."""
    # 5 normal steps (5 ms), 1 huge gap of 1.0 second, then 5 normal steps
    dt_list = [5_000_000] * 5 + [1_000_000_000] + [5_000_000] * 5
    timestamps = np.cumsum(np.insert(dt_list, 0, 1_000_000_000))

    df_imu = pd.DataFrame({
        "timestamp": timestamps,
        "w_RS_S_x": np.zeros(len(timestamps)),
        "w_RS_S_y": np.zeros(len(timestamps)),
        "w_RS_S_z": np.full(len(timestamps), 1.0),
    })

    # Policy 1: hold_orientation over the gap
    df_hold = integrate_gyro_quaternion(df_imu, gap_threshold_s=0.01, gap_policy="hold_orientation")
    gap_idx = 6  # index of delayed sample
    assert df_hold["is_gap"].iloc[gap_idx] is True or bool(df_hold["is_gap"].iloc[gap_idx])
    # Orientation right after the gap should match orientation right before
    np.testing.assert_allclose(
        df_hold[EST_QUATERNION_COLS].iloc[gap_idx],
        df_hold[EST_QUATERNION_COLS].iloc[gap_idx - 1],
        atol=1e-12,
    )

    # Policy 2: zero_order_hold integrates through the full gap
    df_zoh = integrate_gyro_quaternion(df_imu, gap_threshold_s=0.01, gap_policy="zero_order_hold")
    # Angle jumped significantly across the gap
    assert not np.allclose(
        df_zoh[EST_QUATERNION_COLS].iloc[gap_idx],
        df_zoh[EST_QUATERNION_COLS].iloc[gap_idx - 1],
    )


def test_configurable_initial_quaternion():
    """Verify that initial_quaternion is respected and normalized."""
    df_imu = pd.DataFrame({
        "timestamp": [100, 200],
        "w_RS_S_x": [0.0, 0.0],
        "w_RS_S_y": [0.0, 0.0],
        "w_RS_S_z": [0.0, 0.0],
    })

    # Unnormalized 90 deg rotation around X
    rot90 = Rotation.from_euler("x", 90, degrees=True)
    xyzw = rot90.as_quat()
    q_custom = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]) * 5.0  # scaled

    df_est = integrate_gyro_quaternion(df_imu, initial_quaternion=q_custom)
    q_out0 = df_est[EST_QUATERNION_COLS].iloc[0].to_numpy()

    # Must be normalized unit vector
    assert np.isclose(np.linalg.norm(q_out0), 1.0)
    np.testing.assert_allclose(q_out0, q_custom / 5.0, atol=1e-6)


def test_groundtruth_orientation_error_calculation():
    """Test angular error computation: identical gives 0, rotated gives exact angle."""
    # Identical quaternions
    q1 = np.array([[1.0, 0.0, 0.0, 0.0]])
    err_rad, err_deg = compute_angular_error(q1, q1)
    assert np.isclose(err_deg[0], 0.0, atol=1e-10)

    # 45-degree rotation around Y
    rot45 = Rotation.from_euler("y", 45, degrees=True)
    xyzw = rot45.as_quat()
    q_45 = np.array([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]])

    err_rad, err_deg = compute_angular_error(q1, q_45)
    assert np.isclose(err_deg[0], 45.0, atol=1e-6)
    assert np.isclose(err_rad[0], np.deg2rad(45.0), atol=1e-6)


def test_original_dataframe_immutability(synthetic_constant_rate_imu: pd.DataFrame):
    """Ensure that original input DataFrame is left completely unchanged."""
    df_orig = synthetic_constant_rate_imu.copy(deep=True)
    _ = integrate_gyro_quaternion(synthetic_constant_rate_imu)
    pd.testing.assert_frame_equal(synthetic_constant_rate_imu, df_orig)


def test_initial_timestamp_alignment_benchmark():
    """
    Verify that in evaluation benchmarks starting at the first synchronized overlap timestamp:
    1. The initial estimated quaternion matches the ground truth pose at that exact timestamp.
    2. The angular error at t0 is exactly 0.0 degrees.
    3. Integration starts at the exact synchronized IMU timestamp rather than earlier out-of-range samples.
    """
    from src.synchronization import interpolate_groundtruth_to_imu

    # Synthetic IMU: 5 samples before overlap, 10 samples inside overlap, 5 samples after
    ts_imu = np.arange(1_000, 1_000 + 20 * 10, 10, dtype=np.int64)  # 1000, 1010, ..., 1190
    df_imu = pd.DataFrame({
        "timestamp": ts_imu,
        "w_RS_S_x": np.zeros(len(ts_imu)),
        "w_RS_S_y": np.zeros(len(ts_imu)),
        "w_RS_S_z": np.full(len(ts_imu), 0.1),
    })

    # Synthetic Ground Truth: covers timestamps 1050 to 1150
    ts_gt = np.arange(1_050, 1_160, 10, dtype=np.int64)
    # Rotating around Z axis
    rot = Rotation.from_euler("z", np.linspace(0.1, 0.5, len(ts_gt))[:, np.newaxis])
    q_xyzw = rot.as_quat()
    df_gt = pd.DataFrame({
        "timestamp": ts_gt,
        "p_RS_R_x": np.zeros(len(ts_gt)),
        "p_RS_R_y": np.zeros(len(ts_gt)),
        "p_RS_R_z": np.zeros(len(ts_gt)),
        "q_RS_w": q_xyzw[:, 3],
        "q_RS_x": q_xyzw[:, 0],
        "q_RS_y": q_xyzw[:, 1],
        "q_RS_z": q_xyzw[:, 2],
        "v_RS_R_x": np.zeros(len(ts_gt)),
        "v_RS_R_y": np.zeros(len(ts_gt)),
        "v_RS_R_z": np.zeros(len(ts_gt)),
        "b_w_RS_S_x": np.zeros(len(ts_gt)),
        "b_w_RS_S_y": np.zeros(len(ts_gt)),
        "b_w_RS_S_z": np.zeros(len(ts_gt)),
        "b_a_RS_S_x": np.zeros(len(ts_gt)),
        "b_a_RS_S_y": np.zeros(len(ts_gt)),
        "b_a_RS_S_z": np.zeros(len(ts_gt)),
    })

    # Synchronize ground truth to IMU
    df_aligned_gt, valid_mask = interpolate_groundtruth_to_imu(df_imu, df_gt)
    first_valid_idx = int(np.where(valid_mask)[0][0])
    last_valid_idx = int(np.where(valid_mask)[0][-1])

    # Slice IMU strictly from first valid synchronized timestamp
    df_imu_sync = df_imu.iloc[first_valid_idx:last_valid_idx + 1].copy().reset_index(drop=True)
    t_start = df_imu_sync["timestamp"].iloc[0]

    # Ground-truth quaternion at that exact synchronized timestamp
    q_gt_at_t_start = df_aligned_gt.loc[first_valid_idx, GT_QUATERNION_COLS].to_numpy(dtype=np.float64)

    # Run estimator strictly over the synchronized range
    df_est = integrate_gyro_quaternion(df_imu_sync, initial_quaternion=q_gt_at_t_start)
    eval_res = evaluate_orientation_against_groundtruth(
        df_imu=df_imu_sync,
        df_gt=df_gt,
        df_est=df_est,
    )

    # 1. Start timestamp corresponds to the first valid timestamp (1050 ns)
    assert t_start == 1050
    # 2. Initial error at t0 must be zero
    assert np.isclose(eval_res["errors_deg"][0], 0.0, atol=1e-12)
    # 3. Estimated initial quaternion matches GT at t_start
    np.testing.assert_allclose(df_est[EST_QUATERNION_COLS].iloc[0], q_gt_at_t_start, atol=1e-12)


def test_hold_orientation_is_default_gap_policy():
    """Verify that hold_orientation is the default gap_policy in integrate_gyro_quaternion."""
    import inspect
    sig = inspect.signature(integrate_gyro_quaternion)
    assert sig.parameters["gap_policy"].default == "hold_orientation"


def test_zero_order_hold_remains_available_explicitly():
    """Verify that zero_order_hold remains accessible and extrapolates across gaps when requested."""
    dt_ns = 5_000_000
    ts_pre = np.arange(0, 10 * dt_ns, dt_ns, dtype=np.int64)
    # 1.0s gap
    ts_post = ts_pre[-1] + 1_000_000_000 + np.arange(dt_ns, 11 * dt_ns, dt_ns, dtype=np.int64)
    ts = np.concatenate([ts_pre, ts_post])

    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.zeros(len(ts)),
        "w_RS_S_y": np.zeros(len(ts)),
        "w_RS_S_z": np.full(len(ts), 1.0),
    })

    df_zoh = integrate_gyro_quaternion(df, gap_threshold_s=0.01, gap_policy="zero_order_hold")
    gap_idx = 10
    assert df_zoh["is_gap"].iloc[gap_idx] is True or bool(df_zoh["is_gap"].iloc[gap_idx])
    # Orientation jumps across the 1-second gap under ZOH
    assert not np.allclose(
        df_zoh[EST_QUATERNION_COLS].iloc[gap_idx],
        df_zoh[EST_QUATERNION_COLS].iloc[gap_idx - 1],
    )


def test_orientation_integration_signal_drop_2s():
    """
    Simulate a 2-second IMU signal drop.
    Verify that hold_orientation (default) holds the previous orientation at the recovery sample,
    and subsequent normal IMU frames resume integration properly.
    """
    dt_ns = 5_000_000
    # 100 samples pre-gap (0.5s) rotating around Z at 1.0 rad/s
    ts_pre = np.arange(0, 100 * dt_ns, dt_ns, dtype=np.int64)
    # 2.0-second blackout gap
    t_gap_end = ts_pre[-1] + 2_000_000_000
    # 100 samples post-gap (0.5s) continuing rotation at 1.0 rad/s
    ts_post = t_gap_end + np.arange(dt_ns, 101 * dt_ns, dt_ns, dtype=np.int64)
    ts = np.concatenate([ts_pre, ts_post])

    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.zeros(len(ts)),
        "w_RS_S_y": np.zeros(len(ts)),
        "w_RS_S_z": np.full(len(ts), 1.0),
    })

    # Default policy is now hold_orientation
    df_est = integrate_gyro_quaternion(df, gap_threshold_s=0.01)

    gap_idx = 100
    # 1. Gap correctly detected and flagged
    assert df_est["is_gap"].iloc[gap_idx] is True or bool(df_est["is_gap"].iloc[gap_idx])

    # 2. Orientation at the first recovery sample exactly equals the pre-gap orientation
    q_pre = df_est[EST_QUATERNION_COLS].iloc[gap_idx - 1].to_numpy()
    q_rec = df_est[EST_QUATERNION_COLS].iloc[gap_idx].to_numpy()
    np.testing.assert_allclose(q_rec, q_pre, atol=1e-12)

    # 3. Subsequent samples resume normal continuous integration
    q_post = df_est[EST_QUATERNION_COLS].iloc[gap_idx + 1].to_numpy()
    assert not np.allclose(q_post, q_rec)

    # 4. Total output rows match input exactly
    assert len(df_est) == len(df)
    # 5. All quaternions strictly unit-normalized
    norms = np.linalg.norm(df_est[EST_QUATERNION_COLS].to_numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_orientation_complete_failure_5s():
    """
    Simulate a complete 5.0-second IMU outage.
    Verify exact 1:1 frame preservation, no NaN values, and safe attitude freeze across the outage.
    """
    dt_ns = 5_000_000
    ts_1 = np.arange(0, 50 * dt_ns, dt_ns, dtype=np.int64)
    t_outage_end = ts_1[-1] + 5_000_000_000  # 5s blackout
    ts_2 = t_outage_end + np.arange(dt_ns, 51 * dt_ns, dt_ns, dtype=np.int64)
    ts = np.concatenate([ts_1, ts_2])

    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.full(len(ts), 0.2),
        "w_RS_S_y": np.zeros(len(ts)),
        "w_RS_S_z": np.zeros(len(ts)),
    })

    df_est = integrate_gyro_quaternion(df, gap_threshold_s=0.01)

    # Exactly 1:1 input to output mapping
    assert len(df_est) == len(df)
    assert (df_est["timestamp"].to_numpy() == ts).all()

    # Zero NaNs anywhere in output
    assert not df_est[EST_QUATERNION_COLS].isna().any().any()

    # Quaternions are normalized
    norms = np.linalg.norm(df_est[EST_QUATERNION_COLS].to_numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    # At gap sample (idx 50), orientation matches pre-gap sample
    np.testing.assert_allclose(
        df_est[EST_QUATERNION_COLS].iloc[50],
        df_est[EST_QUATERNION_COLS].iloc[49],
        atol=1e-12,
    )


def test_post_gap_recovery_continuity():
    """
    Verify that when resuming after a gap, the rotation step delta_theta
    corresponds only to subsequent single-step dt, introducing no artificial angular acceleration.
    """
    dt_ns = 5_000_000
    ts_pre = np.arange(0, 20 * dt_ns, dt_ns, dtype=np.int64)
    ts_post = ts_pre[-1] + 3_000_000_000 + np.arange(dt_ns, 21 * dt_ns, dt_ns, dtype=np.int64)
    ts = np.concatenate([ts_pre, ts_post])

    # Steady 1.0 rad/s yaw
    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.zeros(len(ts)),
        "w_RS_S_y": np.zeros(len(ts)),
        "w_RS_S_z": np.full(len(ts), 1.0),
    })

    df_est = integrate_gyro_quaternion(df, gap_threshold_s=0.01)

    gap_idx = 20
    # Step across gap is 0
    err_gap = compute_angular_error(
        df_est[EST_QUATERNION_COLS].iloc[gap_idx : gap_idx + 1].to_numpy(),
        df_est[EST_QUATERNION_COLS].iloc[gap_idx - 1 : gap_idx].to_numpy(),
    )[1][0]
    assert np.isclose(err_gap, 0.0, atol=1e-10)

    # Step immediately after gap is nominal (w * dt = 1.0 * 0.005 rad ~= 0.286 deg)
    err_next = compute_angular_error(
        df_est[EST_QUATERNION_COLS].iloc[gap_idx + 1 : gap_idx + 2].to_numpy(),
        df_est[EST_QUATERNION_COLS].iloc[gap_idx : gap_idx + 1].to_numpy(),
    )[1][0]
    nominal_deg = np.rad2deg(1.0 * 0.005)
    np.testing.assert_allclose(err_next, nominal_deg, atol=1e-3)


def test_orientation_multiple_outages_and_consecutive_drops():
    """
    Simulate multiple outages and consecutive frame drops within a single trajectory:
    1. Initial 0.5s rotation around Z at 1.0 rad/s.
    2. Consecutive frame drop lasting 75 ms (15 missing samples).
    3. Normal rotation for 0.5s.
    4. Complete signal outage of 2.0s.
    5. Normal rotation for 0.5s.
    6. Complete signal outage of 5.0s.
    7. Final 0.5s rotation.

    Verifies:
    - Exactly 1:1 frame count preservation (N_out == N_in).
    - Every gap transition holds attitude cleanly without runaway divergence.
    - Post-gap recoveries resume valid integration.
    - All output quaternions are finite and strictly unit-normalized.
    - Output timestamps strictly match input timestamps.
    """
    dt_ns = 5_000_000  # 5 ms (200 Hz)

    # Segment 1: 100 samples (0.5s)
    ts_1 = np.arange(0, 100 * dt_ns, dt_ns, dtype=np.int64)

    # Outage 1: 75 ms drop (15 dropped samples)
    t_out1 = ts_1[-1] + 75_000_000
    # Segment 2: 100 samples (0.5s)
    ts_2 = t_out1 + np.arange(dt_ns, 101 * dt_ns, dt_ns, dtype=np.int64)

    # Outage 2: 2.0s blackout
    t_out2 = ts_2[-1] + 2_000_000_000
    # Segment 3: 100 samples (0.5s)
    ts_3 = t_out2 + np.arange(dt_ns, 101 * dt_ns, dt_ns, dtype=np.int64)

    # Outage 3: 5.0s blackout
    t_out3 = ts_3[-1] + 5_000_000_000
    # Segment 4: 100 samples (0.5s)
    ts_4 = t_out3 + np.arange(dt_ns, 101 * dt_ns, dt_ns, dtype=np.int64)

    ts = np.concatenate([ts_1, ts_2, ts_3, ts_4])
    total_samples = len(ts)

    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.zeros(total_samples),
        "w_RS_S_y": np.zeros(total_samples),
        "w_RS_S_z": np.full(total_samples, 1.0),
    })

    # Integrate with default hold_orientation policy
    df_est = integrate_gyro_quaternion(df, gap_threshold_s=0.01)

    # 1. 1:1 input to output mapping
    assert len(df_est) == total_samples
    assert (df_est["timestamp"].to_numpy() == ts).all()

    # 2. Correct gap detections
    # Gap indices correspond to the starts of segments 2, 3, and 4
    gap_indices = [100, 200, 300]
    for idx in gap_indices:
        assert df_est["is_gap"].iloc[idx] is True or bool(df_est["is_gap"].iloc[idx])
        # Transition across each outage holds previous orientation
        q_pre = df_est[EST_QUATERNION_COLS].iloc[idx - 1].to_numpy()
        q_rec = df_est[EST_QUATERNION_COLS].iloc[idx].to_numpy()
        np.testing.assert_allclose(q_rec, q_pre, atol=1e-12)

        # Subsequent sample resumes normal integration
        q_next = df_est[EST_QUATERNION_COLS].iloc[idx + 1].to_numpy()
        assert not np.allclose(q_next, q_rec)

    # 3. Finite, strictly normalized quaternions
    quats = df_est[EST_QUATERNION_COLS].to_numpy()
    assert np.isfinite(quats).all()
    norms = np.linalg.norm(quats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_orientation_random_isolated_frame_drops():
    """
    Simulate random isolated frame drops throughout a continuous trajectory:
    Randomly drops 10% of samples (causing isolated 10 ms intervals).
    Verifies that when gap_threshold_s is set to nominal (e.g. 7.5 ms),
    isolated dropped samples trigger hold_orientation without throwing errors or corrupting integration.
    """
    dt_ns = 5_000_000
    n_total = 500
    ts = np.arange(0, n_total * dt_ns, dt_ns, dtype=np.int64)

    rng = np.random.default_rng(77)
    drop_mask = rng.random(n_total) > 0.10
    drop_mask[0] = True  # keep first sample
    drop_mask[-1] = True # keep last sample

    ts_survived = ts[drop_mask]
    n_survived = len(ts_survived)

    df = pd.DataFrame({
        "timestamp": ts_survived,
        "w_RS_S_x": np.zeros(n_survived),
        "w_RS_S_y": np.zeros(n_survived),
        "w_RS_S_z": np.full(n_survived, 0.5),
    })

    df_est = integrate_gyro_quaternion(df, gap_threshold_s=0.0075)

    assert len(df_est) == n_survived
    assert not df_est[EST_QUATERNION_COLS].isna().any().any()
    norms = np.linalg.norm(df_est[EST_QUATERNION_COLS].to_numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)



