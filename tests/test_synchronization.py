"""
Comprehensive unit tests for src/synchronization.py.

Tests cover:
- Overlapping interval detection with both overlapping and disjoint ranges.
- Linear interpolation of 3D Position.
- Linear interpolation of 3D Velocity.
- Spherical Linear Interpolation (SLERP) of Quaternions with w, x, y, z order and unit norm check.
- Non-overlapping timestamp ranges producing has_overlap=False and empty/all-False masks.
- Out-of-bounds IMU timestamps producing NaN and False in valid_mask (no extrapolation).
- High-precision timestamp preservation with large Unix epoch nanoseconds.
- Original DataFrame immutability (ensuring input data is not altered).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from src.synchronization import (
    get_overlapping_interval,
    interpolate_groundtruth_to_imu,
)
from src.data_loader import (
    GT_ALL_COLS,
    GT_POSITION_COLS,
    GT_QUATERNION_COLS,
    GT_VELOCITY_COLS,
    GT_GYRO_BIAS_COLS,
    GT_ACCEL_BIAS_COLS,
)


@pytest.fixture
def sample_gt_dataframe() -> pd.DataFrame:
    """
    Generate synthetic ground truth data with known analytical trajectories:
    - Position: p(t) = [1 + 2*t, -3 + 4*t, 5 - 6*t]
    - Velocity: v(t) = [2, 4, -6]
    - Orientation: Rotation around Z axis by theta(t) = 0.5 * t (rad)
    - Biases: constant values
    """
    t_start = 1_400_000_000_000_000_000  # Unix epoch in nanoseconds
    dt_ns = 10_000_000  # 10 ms (100 Hz)
    n_samples = 101  # 0.0 to 1.0 seconds

    timestamps = np.arange(t_start, t_start + n_samples * dt_ns, dt_ns, dtype=np.int64)
    t_s = (timestamps - t_start) * 1e-9

    p_x = 1.0 + 2.0 * t_s
    p_y = -3.0 + 4.0 * t_s
    p_z = 5.0 - 6.0 * t_s

    v_x = np.full(n_samples, 2.0)
    v_y = np.full(n_samples, 4.0)
    v_z = np.full(n_samples, -6.0)

    # Rotation around Z axis: angles = 0.5 * t_s (2D array shape (n_samples, 1))
    angles = (0.5 * t_s)[:, np.newaxis]
    rotations = Rotation.from_euler("z", angles)
    # SciPy returns [x, y, z, w]
    q_xyzw = rotations.as_quat()
    # Loader standard is [w, x, y, z]
    q_w = q_xyzw[:, 3]
    q_x = q_xyzw[:, 0]
    q_y = q_xyzw[:, 1]
    q_z = q_xyzw[:, 2]

    data = {
        "timestamp": timestamps,
        "p_RS_R_x": p_x,
        "p_RS_R_y": p_y,
        "p_RS_R_z": p_z,
        "q_RS_w": q_w,
        "q_RS_x": q_x,
        "q_RS_y": q_y,
        "q_RS_z": q_z,
        "v_RS_R_x": v_x,
        "v_RS_R_y": v_y,
        "v_RS_R_z": v_z,
        "b_w_RS_S_x": np.full(n_samples, 0.01),
        "b_w_RS_S_y": np.full(n_samples, -0.02),
        "b_w_RS_S_z": np.full(n_samples, 0.03),
        "b_a_RS_S_x": np.full(n_samples, 0.04),
        "b_a_RS_S_y": np.full(n_samples, -0.05),
        "b_a_RS_S_z": np.full(n_samples, 0.06),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_imu_dataframe(sample_gt_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic IMU DataFrame sampled at 200 Hz with timestamps:
    - Starting 0.2 s before ground truth
    - Ending 0.2 s after ground truth
    """
    t_gt_start = sample_gt_dataframe["timestamp"].iloc[0]
    dt_imu_ns = 5_000_000  # 5 ms (200 Hz)

    t_imu_start = t_gt_start - int(0.2 * 1e9)  # 200 ms before GT
    t_imu_end = sample_gt_dataframe["timestamp"].iloc[-1] + int(0.2 * 1e9)  # 200 ms after GT

    timestamps = np.arange(t_imu_start, t_imu_end + dt_imu_ns, dt_imu_ns, dtype=np.int64)

    data = {
        "timestamp": timestamps,
        "w_RS_S_x": np.zeros(len(timestamps)),
        "w_RS_S_y": np.zeros(len(timestamps)),
        "w_RS_S_z": np.full(len(timestamps), 0.5),
        "a_RS_S_x": np.zeros(len(timestamps)),
        "a_RS_S_y": np.zeros(len(timestamps)),
        "a_RS_S_z": np.full(len(timestamps), 9.81),
    }
    return pd.DataFrame(data)


# ==============================================================================
# TESTS
# ==============================================================================


def test_get_overlapping_interval(sample_imu_dataframe, sample_gt_dataframe):
    """Test overlapping interval calculation and sample masking."""
    res = get_overlapping_interval(sample_imu_dataframe, sample_gt_dataframe)

    assert res["has_overlap"] is True
    assert res["overlap_start_ns"] == sample_gt_dataframe["timestamp"].iloc[0]
    assert res["overlap_end_ns"] == sample_gt_dataframe["timestamp"].iloc[-1]
    assert np.isclose(res["overlap_duration_s"], 1.0, atol=1e-6)

    assert res["gt_overlap_count"] == len(sample_gt_dataframe)
    # GT is 1.0 s; IMU at 200 Hz over 1.0s overlap has 201 samples
    assert res["imu_overlap_count"] == 201
    assert np.sum(res["imu_overlap_mask"]) == 201


def test_non_overlapping_interval():
    """Test that disjoint ranges correctly report has_overlap=False."""
    df1 = pd.DataFrame({"timestamp": np.array([100, 200, 300], dtype=np.int64)})
    df2 = pd.DataFrame({"timestamp": np.array([400, 500, 600], dtype=np.int64)})

    res = get_overlapping_interval(df1, df2)
    assert res["has_overlap"] is False
    assert res["overlap_start_ns"] is None
    assert res["imu_overlap_count"] == 0
    assert not np.any(res["imu_overlap_mask"])


def test_position_and_velocity_interpolation(sample_imu_dataframe, sample_gt_dataframe):
    """Test that linear interpolation of position and velocity matches analytical truth."""
    df_aligned, valid_mask = interpolate_groundtruth_to_imu(
        sample_imu_dataframe, sample_gt_dataframe
    )

    t_ref = sample_gt_dataframe["timestamp"].iloc[0]
    valid_ts = df_aligned.loc[valid_mask, "timestamp"]
    t_s = (valid_ts - t_ref) * 1e-9

    expected_px = 1.0 + 2.0 * t_s
    expected_py = -3.0 + 4.0 * t_s
    expected_pz = 5.0 - 6.0 * t_s

    np.testing.assert_allclose(df_aligned.loc[valid_mask, "p_RS_R_x"], expected_px, atol=1e-8)
    np.testing.assert_allclose(df_aligned.loc[valid_mask, "p_RS_R_y"], expected_py, atol=1e-8)
    np.testing.assert_allclose(df_aligned.loc[valid_mask, "p_RS_R_z"], expected_pz, atol=1e-8)

    np.testing.assert_allclose(df_aligned.loc[valid_mask, "v_RS_R_x"], 2.0, atol=1e-8)
    np.testing.assert_allclose(df_aligned.loc[valid_mask, "v_RS_R_y"], 4.0, atol=1e-8)
    np.testing.assert_allclose(df_aligned.loc[valid_mask, "v_RS_R_z"], -6.0, atol=1e-8)


def test_quaternion_slerp_interpolation(sample_imu_dataframe, sample_gt_dataframe):
    """Test that SLERP quaternion interpolation preserves w,x,y,z order and unit norms."""
    df_aligned, valid_mask = interpolate_groundtruth_to_imu(
        sample_imu_dataframe, sample_gt_dataframe
    )

    valid_quats = df_aligned.loc[valid_mask, GT_QUATERNION_COLS].to_numpy()

    # 1. Unit norm check
    norms = np.linalg.norm(valid_quats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-7)

    # 2. Check against analytical rotation around Z axis
    t_ref = sample_gt_dataframe["timestamp"].iloc[0]
    valid_ts = df_aligned.loc[valid_mask, "timestamp"]
    t_s = (valid_ts - t_ref) * 1e-9
    expected_angles = (0.5 * t_s.to_numpy())[:, np.newaxis]
    expected_rot = Rotation.from_euler("z", expected_angles)
    expected_xyzw = expected_rot.as_quat()

    # Loader standard is [w, x, y, z]
    expected_wxyz = np.column_stack([
        expected_xyzw[:, 3],  # w
        expected_xyzw[:, 0],  # x
        expected_xyzw[:, 1],  # y
        expected_xyzw[:, 2],  # z
    ])

    # Account for quaternion double cover (+q and -q represent same rotation)
    dot_products = np.abs(np.sum(valid_quats * expected_wxyz, axis=1))
    np.testing.assert_allclose(dot_products, 1.0, atol=1e-6)


def test_no_extrapolation_outside_groundtruth(sample_imu_dataframe, sample_gt_dataframe):
    """Verify that IMU timestamps outside ground truth coverage are marked False and NaN."""
    df_aligned, valid_mask = interpolate_groundtruth_to_imu(
        sample_imu_dataframe, sample_gt_dataframe
    )

    t_gt_min = sample_gt_dataframe["timestamp"].iloc[0]
    t_gt_max = sample_gt_dataframe["timestamp"].iloc[-1]

    invalid_mask = ~valid_mask
    assert np.any(invalid_mask)

    # Outside timestamps should be before t_gt_min or after t_gt_max
    outside_ts = df_aligned.loc[invalid_mask, "timestamp"]
    assert np.all((outside_ts < t_gt_min) | (outside_ts > t_gt_max))

    # All interpolated fields outside range must be NaN
    non_ts_cols = [c for c in GT_ALL_COLS if c != "timestamp"]
    for col in non_ts_cols:
        assert df_aligned.loc[invalid_mask, col].isna().all()


def test_timestamp_precision_large_unix_nanoseconds():
    """Verify that using 19-digit Unix nanoseconds does not suffer floating-point roundoff."""
    # Two timestamps separated by exactly 1 ns
    t_base = 1_700_000_000_000_000_000  # Year ~2023 in ns
    gt_df = pd.DataFrame({
        "timestamp": [t_base, t_base + 10_000_000],  # 10 ms interval
        "p_RS_R_x": [0.0, 10.0],
        "p_RS_R_y": [0.0, 0.0],
        "p_RS_R_z": [0.0, 0.0],
        "q_RS_w": [1.0, 1.0],
        "q_RS_x": [0.0, 0.0],
        "q_RS_y": [0.0, 0.0],
        "q_RS_z": [0.0, 0.0],
        "v_RS_R_x": [0.0, 0.0],
        "v_RS_R_y": [0.0, 0.0],
        "v_RS_R_z": [0.0, 0.0],
        "b_w_RS_S_x": [0.0, 0.0],
        "b_w_RS_S_y": [0.0, 0.0],
        "b_w_RS_S_z": [0.0, 0.0],
        "b_a_RS_S_x": [0.0, 0.0],
        "b_a_RS_S_y": [0.0, 0.0],
        "b_a_RS_S_z": [0.0, 0.0],
    })

    # Query exactly midway: 5 ms (5,000,000 ns)
    query_ts = t_base + 5_000_000
    imu_df = pd.DataFrame({"timestamp": [query_ts]})

    df_aligned, valid_mask = interpolate_groundtruth_to_imu(imu_df, gt_df)

    assert valid_mask[0] is True or bool(valid_mask[0])
    assert df_aligned["timestamp"].iloc[0] == query_ts
    assert df_aligned["timestamp"].dtype == np.int64
    # Expected p_x at midway is exactly 5.0
    assert np.isclose(df_aligned["p_RS_R_x"].iloc[0], 5.0, atol=1e-10)


def test_dataframe_immutability(sample_imu_dataframe, sample_gt_dataframe):
    """Verify that get_overlapping_interval and interpolate_groundtruth_to_imu do not modify inputs."""
    imu_orig = sample_imu_dataframe.copy(deep=True)
    gt_orig = sample_gt_dataframe.copy(deep=True)

    _ = get_overlapping_interval(sample_imu_dataframe, sample_gt_dataframe)
    _ = interpolate_groundtruth_to_imu(sample_imu_dataframe, sample_gt_dataframe)

    pd.testing.assert_frame_equal(sample_imu_dataframe, imu_orig)
    pd.testing.assert_frame_equal(sample_gt_dataframe, gt_orig)
