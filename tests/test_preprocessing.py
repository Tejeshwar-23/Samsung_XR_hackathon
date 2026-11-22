"""
Unit tests for src/preprocessing.py.

Tests:
1. dt calculation and conversion to seconds.
2. Timestamp gap detection with synthetic gaps.
3. Sensor vector magnitude calculation (Euclidean norms).
4. Sensor statistics (min, max, mean, std) across channels.
5. Stationarity detection using synthetic stationary data (at rest).
6. Stationarity rejection using synthetic moving data (high rotation/acceleration variance).
7. Bias estimation succeeds on stationary data and is correctly aborted on dynamic data.
8. Gravity vector assessment returns expected values on stationary data.
9. Confirmation that input DataFrame and original measurements remain completely unchanged.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    compute_timestamp_intervals,
    compute_vector_magnitudes,
    compute_sensor_statistics,
    analyze_initial_stationarity,
    estimate_sensor_bias,
    assess_gravity_vector,
    GYRO_COLS,
    ACCEL_COLS,
    SENSOR_COLS,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def synthetic_stationary_imu() -> pd.DataFrame:
    """Generate 400 samples (2 seconds at 200 Hz) of stationary IMU data at rest."""
    n_samples = 400
    t0 = 1_000_000_000_000
    dt_ns = 5_000_000  # 5 ms (200 Hz)
    timestamps = np.arange(t0, t0 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    rng = np.random.default_rng(42)
    # Stationary gyro: zero mean with tiny noise (< 0.005 rad/s)
    gyro_noise = rng.normal(0, 0.002, size=(n_samples, 3))
    # True gyro bias = [0.01, -0.015, 0.02]
    gyro_bias = np.array([0.01, -0.015, 0.02])
    gyro_vals = gyro_bias + gyro_noise

    # Stationary accel: pure gravity [0, 0, 9.80665] with tiny noise (< 0.02 m/s^2)
    accel_noise = rng.normal(0, 0.01, size=(n_samples, 3))
    accel_vals = np.array([0.0, 0.0, 9.80665]) + accel_noise

    data = {
        "timestamp": timestamps,
        "w_RS_S_x": gyro_vals[:, 0],
        "w_RS_S_y": gyro_vals[:, 1],
        "w_RS_S_z": gyro_vals[:, 2],
        "a_RS_S_x": accel_vals[:, 0],
        "a_RS_S_y": accel_vals[:, 1],
        "a_RS_S_z": accel_vals[:, 2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_moving_imu() -> pd.DataFrame:
    """Generate 400 samples of highly dynamic/moving IMU data."""
    n_samples = 400
    t0 = 1_000_000_000_000
    dt_ns = 5_000_000
    timestamps = np.arange(t0, t0 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    rng = np.random.default_rng(123)
    # High rotational velocity (mean magnitude ~0.8 rad/s)
    gyro_vals = rng.normal(0.5, 0.2, size=(n_samples, 3))
    # Erratic accelerations with high variance
    accel_vals = rng.normal(12.0, 3.0, size=(n_samples, 3))

    data = {
        "timestamp": timestamps,
        "w_RS_S_x": gyro_vals[:, 0],
        "w_RS_S_y": gyro_vals[:, 1],
        "w_RS_S_z": gyro_vals[:, 2],
        "a_RS_S_x": accel_vals[:, 0],
        "a_RS_S_y": accel_vals[:, 1],
        "a_RS_S_z": accel_vals[:, 2],
    }
    return pd.DataFrame(data)


# ==============================================================================
# TESTS
# ==============================================================================


def test_compute_timestamp_intervals(synthetic_stationary_imu: pd.DataFrame):
    """Test dt calculation in seconds and nanoseconds."""
    res = compute_timestamp_intervals(synthetic_stationary_imu)

    assert len(res["dt_ns"]) == len(synthetic_stationary_imu) - 1
    assert len(res["dt_s"]) == len(synthetic_stationary_imu) - 1
    assert res["dt_ns"].dtype == np.int64
    assert res["dt_s"].dtype == np.float64
    assert np.isclose(res["mean_dt_s"], 0.005, atol=1e-6)
    assert np.isclose(res["estimated_freq_hz"], 200.0, atol=1e-2)
    assert res["num_gaps"] == 0


def test_timestamp_gap_detection(synthetic_stationary_imu: pd.DataFrame):
    """Test identifying unusually large timestamp gaps."""
    df_gap = synthetic_stationary_imu.copy()
    # Inject two artificial gaps (e.g. 20 ms and 50 ms instead of 5 ms)
    ts = df_gap["timestamp"].to_numpy().copy()
    ts[50:] += int(15e6)   # 20 ms dt at index 50
    ts[100:] += int(45e6)  # 50 ms dt at index 100
    df_gap["timestamp"] = ts

    res = compute_timestamp_intervals(df_gap, nominal_dt_s=0.005)
    assert res["num_gaps"] == 2
    assert 50 in res["gap_indices"]
    assert 100 in res["gap_indices"]


def test_compute_vector_magnitudes(synthetic_stationary_imu: pd.DataFrame):
    """Test Euclidean norm calculations for gyro and accel vectors."""
    gyro_mag, accel_mag = compute_vector_magnitudes(synthetic_stationary_imu)

    assert len(gyro_mag) == len(synthetic_stationary_imu)
    assert len(accel_mag) == len(synthetic_stationary_imu)
    assert gyro_mag.dtype == np.float64
    assert accel_mag.dtype == np.float64

    # Synthetic stationary data has ~9.81 m/s^2 accel norm
    assert np.allclose(accel_mag, 9.80665, atol=0.1)
    # Gyro norm is close to norm([0.01, -0.015, 0.02]) ~= 0.0269 rad/s
    assert np.all(gyro_mag < 0.05)


def test_compute_sensor_statistics(synthetic_stationary_imu: pd.DataFrame):
    """Test per-channel and magnitude statistics computation."""
    stats = compute_sensor_statistics(synthetic_stationary_imu)

    for col in SENSOR_COLS + ["gyro_magnitude", "accel_magnitude"]:
        assert col in stats
        s = stats[col]
        assert "min" in s and "max" in s and "mean" in s and "std" in s
        assert s["min"] <= s["mean"] <= s["max"]
        assert s["std"] >= 0.0


def test_stationarity_detection_on_synthetic_stationary(synthetic_stationary_imu: pd.DataFrame):
    """Test that truly stationary data passes stationarity analysis."""
    result = analyze_initial_stationarity(synthetic_stationary_imu, duration_s=2.0)

    assert result["is_stationary"] is True
    assert result["sample_count"] == 400
    assert result["evidence"]["checks"]["gyro_mean_passed"] is True
    assert result["evidence"]["checks"]["accel_std_passed"] is True
    assert result["evidence"]["checks"]["accel_mean_passed"] is True


def test_stationarity_rejection_on_synthetic_moving(synthetic_moving_imu: pd.DataFrame):
    """Test that dynamic/moving data is correctly rejected by stationarity checks."""
    result = analyze_initial_stationarity(synthetic_moving_imu, duration_s=2.0)

    assert result["is_stationary"] is False
    assert "Stationarity rejected" in result["reason"]
    # Gyro mean and accel std checks should fail
    assert (
        result["evidence"]["checks"]["gyro_mean_passed"] is False
        or result["evidence"]["checks"]["accel_std_passed"] is False
    )


def test_bias_estimation_stationary_vs_moving(
    synthetic_stationary_imu: pd.DataFrame, synthetic_moving_imu: pd.DataFrame
):
    """Test that bias estimation succeeds on stationary data and is aborted on moving data."""
    # Stationary: Should estimate biases
    bias_stat = estimate_sensor_bias(synthetic_stationary_imu, duration_s=2.0)
    assert bias_stat["bias_estimated"] is True
    assert bias_stat["gyro_bias"] is not None
    assert len(bias_stat["gyro_bias"]) == 3
    # Check estimated gyro bias close to injected bias [0.01, -0.015, 0.02]
    np.testing.assert_allclose(bias_stat["gyro_bias"], [0.01, -0.015, 0.02], atol=0.005)

    # Moving: Should abort bias estimation
    bias_mov = estimate_sensor_bias(synthetic_moving_imu, duration_s=2.0)
    assert bias_mov["bias_estimated"] is False
    assert bias_mov["gyro_bias"] is None
    assert bias_mov["accel_bias"] is None
    assert "aborted" in bias_mov["reason"]


def test_gravity_assessment(synthetic_stationary_imu: pd.DataFrame):
    """Test gravity vector assessment on stationary data."""
    grav = assess_gravity_vector(synthetic_stationary_imu, duration_s=2.0)

    assert grav["gravity_assessed"] is True
    assert np.isclose(grav["magnitude"], 9.80665, atol=0.05)
    # Unit direction should be along [0, 0, 1]
    np.testing.assert_allclose(grav["unit_direction"], [0.0, 0.0, 1.0], atol=0.02)


def test_original_data_immutability(synthetic_stationary_imu: pd.DataFrame):
    """Ensure that all preprocessing functions leave original data completely unmodified."""
    df_orig = synthetic_stationary_imu.copy(deep=True)

    _ = compute_timestamp_intervals(synthetic_stationary_imu)
    _ = compute_vector_magnitudes(synthetic_stationary_imu)
    _ = compute_sensor_statistics(synthetic_stationary_imu)
    stat = analyze_initial_stationarity(synthetic_stationary_imu)
    _ = estimate_sensor_bias(synthetic_stationary_imu, stationarity_result=stat)
    _ = assess_gravity_vector(synthetic_stationary_imu, stationarity_result=stat)

    pd.testing.assert_frame_equal(synthetic_stationary_imu, df_orig)
