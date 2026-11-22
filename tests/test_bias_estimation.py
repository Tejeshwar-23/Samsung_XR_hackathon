"""
Unit tests for src/bias_estimation.py.

Tests cover:
- Synthetic stationary IMU with known injected bias (accurate detection and recovery).
- Synthetic moving IMU with no valid stationary window (clean rejection, zero bias returned).
- Constant-rate rotation scenario demonstrating low-variance rotational ambiguity.
- Dataset with no qualifying window returns bias_estimated=False and descriptive reason.
- Enforcement of minimum continuous duration threshold (e.g., rejecting 1.5s window when 2.0s is required).
- Configurable thresholds and output structure validation.
- Original DataFrame immutability.
- Safe fallback when detection fails.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from src.bias_estimation import (
    detect_quasi_stationary_intervals,
    estimate_gyro_bias,
)


@pytest.fixture
def synthetic_stationary_with_bias() -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate 1000 samples (5.0s at 200 Hz) of stationary IMU with known injected gyro bias."""
    n_samples = 1000
    dt_ns = 5_000_000
    t0 = 1_000_000_000_000
    timestamps = np.arange(t0, t0 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    rng = np.random.default_rng(42)
    # True injected bias: [0.015, -0.025, 0.080] rad/s
    true_bias = np.array([0.015, -0.025, 0.080])
    gyro_noise = rng.normal(0, 0.002, size=(n_samples, 3))
    w_vals = true_bias + gyro_noise

    # Stationary accel: gravity along Z [0, 0, 9.80665] with tiny noise
    accel_noise = rng.normal(0, 0.01, size=(n_samples, 3))
    a_vals = np.array([0.0, 0.0, 9.80665]) + accel_noise

    df = pd.DataFrame({
        "timestamp": timestamps,
        "w_RS_S_x": w_vals[:, 0],
        "w_RS_S_y": w_vals[:, 1],
        "w_RS_S_z": w_vals[:, 2],
        "a_RS_S_x": a_vals[:, 0],
        "a_RS_S_y": a_vals[:, 1],
        "a_RS_S_z": a_vals[:, 2],
    })
    return df, true_bias


@pytest.fixture
def synthetic_continuous_moving() -> pd.DataFrame:
    """Generate continuous dynamic flight data with high variance and non-1g accelerations."""
    n_samples = 1000
    dt_ns = 5_000_000
    t0 = 1_000_000_000_000
    timestamps = np.arange(t0, t0 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    rng = np.random.default_rng(123)
    w_vals = rng.normal(0.5, 0.3, size=(n_samples, 3))
    a_vals = rng.normal(12.0, 2.5, size=(n_samples, 3))

    return pd.DataFrame({
        "timestamp": timestamps,
        "w_RS_S_x": w_vals[:, 0],
        "w_RS_S_y": w_vals[:, 1],
        "w_RS_S_z": w_vals[:, 2],
        "a_RS_S_x": a_vals[:, 0],
        "a_RS_S_y": a_vals[:, 1],
        "a_RS_S_z": a_vals[:, 2],
    })


# ==============================================================================
# TESTS
# ==============================================================================


def test_synthetic_stationary_bias_recovery(synthetic_stationary_with_bias):
    """Test that stationary IMU data accurately recovers known injected bias."""
    df, true_bias = synthetic_stationary_with_bias
    res = estimate_gyro_bias(df, min_continuous_duration_s=2.0)

    assert res["bias_estimated"] is True
    assert res["status"] == "CONFIRMED_QUASI_STATIONARY"
    assert res["confidence"] == "HIGH"
    assert res["interval_info"]["duration_s"] >= 2.0

    # Recovered bias should match injected bias within noise margin (< 0.001 rad/s)
    np.testing.assert_allclose(res["gyro_bias"], true_bias, atol=0.001)


def test_synthetic_moving_rejected(synthetic_continuous_moving):
    """Test that dynamic motion data is rejected and returns safe zero bias fallback."""
    res = estimate_gyro_bias(synthetic_continuous_moving, min_continuous_duration_s=2.0)

    assert res["bias_estimated"] is False
    assert res["status"] == "NO_RELIABLE_INTERVAL"
    assert res["confidence"] == "NONE"
    np.testing.assert_allclose(res["gyro_bias"], np.zeros(3), atol=1e-12)
    assert "No continuous quasi-stationary segment" in res["reason"]


def test_constant_rate_rotation_limitation():
    """
    Demonstrate the documented theoretical limitation:
    A pure constant-rate rotation around gravity (e.g., yaw spin at 0.5 rad/s)
    has low variance and constant 1g acceleration, illustrating why variance alone
    cannot mathematically separate constant rotational rate from sensor bias without external sensors.
    """
    n_samples = 600  # 3.0 s
    dt_ns = 5_000_000
    timestamps = np.arange(100, 100 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    # True motion is steady 0.5 rad/s yaw with NO bias
    w_vals = np.zeros((n_samples, 3))
    w_vals[:, 2] = 0.5

    # Accelerometer measures pure gravity along Z
    a_vals = np.zeros((n_samples, 3))
    a_vals[:, 2] = 9.80665

    df = pd.DataFrame({
        "timestamp": timestamps,
        "w_RS_S_x": w_vals[:, 0],
        "w_RS_S_y": w_vals[:, 1],
        "w_RS_S_z": w_vals[:, 2],
        "a_RS_S_x": a_vals[:, 0],
        "a_RS_S_y": a_vals[:, 1],
        "a_RS_S_z": a_vals[:, 2],
    })

    res = estimate_gyro_bias(df, min_continuous_duration_s=2.0)

    # The detector accepts this because variance is near zero and accel is 1g
    assert res["bias_estimated"] is True
    # The output estimate reflects the mean angular rate (0.5 rad/s)
    np.testing.assert_allclose(res["gyro_bias"], [0.0, 0.0, 0.5], atol=1e-6)
    # The limitation notice explicitly warns of this ambiguity
    assert "THEORETICAL LIMITATION NOTICE" in res["limitations_notice"]


def test_minimum_duration_enforcement():
    """Verify that candidate intervals shorter than min_continuous_duration_s are rejected."""
    # Create 1.5 seconds of stationary data, followed by 3 seconds of high motion
    dt_ns = 5_000_000
    n_stat = 300  # 1.5 s
    n_mov = 600   # 3.0 s
    total_n = n_stat + n_mov
    timestamps = np.arange(100, 100 + total_n * dt_ns, dt_ns, dtype=np.int64)

    rng = np.random.default_rng(99)
    # 1.5s stationary
    w_stat = rng.normal(0, 0.001, size=(n_stat, 3))
    a_stat = np.tile([0.0, 0.0, 9.80665], (n_stat, 1)) + rng.normal(0, 0.01, size=(n_stat, 3))

    # 3.0s moving
    w_mov = rng.normal(1.0, 0.5, size=(n_mov, 3))
    a_mov = rng.normal(15.0, 3.0, size=(n_mov, 3))

    w_all = np.vstack([w_stat, w_mov])
    a_all = np.vstack([a_stat, a_mov])

    df = pd.DataFrame({
        "timestamp": timestamps,
        "w_RS_S_x": w_all[:, 0],
        "w_RS_S_y": w_all[:, 1],
        "w_RS_S_z": w_all[:, 2],
        "a_RS_S_x": a_all[:, 0],
        "a_RS_S_y": a_all[:, 1],
        "a_RS_S_z": a_all[:, 2],
    })

    # Require 2.0s: The 1.5s segment must be rejected
    res_2s = estimate_gyro_bias(df, min_continuous_duration_s=2.0)
    assert res_2s["bias_estimated"] is False
    assert res_2s["status"] == "NO_RELIABLE_INTERVAL"

    # Require 1.0s: The 1.5s segment is accepted
    res_1s = estimate_gyro_bias(df, min_continuous_duration_s=1.0)
    assert res_1s["bias_estimated"] is True
    assert res_1s["status"] == "CONFIRMED_QUASI_STATIONARY"


def test_empty_or_tiny_dataset():
    """Verify safe fallback on empty or tiny dataset."""
    df_empty = pd.DataFrame(columns=["timestamp", "w_RS_S_x", "w_RS_S_y", "w_RS_S_z", "a_RS_S_x", "a_RS_S_y", "a_RS_S_z"])
    res = estimate_gyro_bias(df_empty)
    assert res["bias_estimated"] is False
    assert res["status"] == "NO_RELIABLE_INTERVAL"
    np.testing.assert_allclose(res["gyro_bias"], np.zeros(3))


def test_original_dataframe_immutability(synthetic_stationary_with_bias):
    """Ensure raw input DataFrame is never mutated by bias estimation."""
    df, _ = synthetic_stationary_with_bias
    df_orig = df.copy(deep=True)

    _ = estimate_gyro_bias(df)
    pd.testing.assert_frame_equal(df, df_orig)


def test_bias_detector_rejects_gap_bridging_window():
    """
    Verify that an individual candidate window bridging a 5-second gap is rejected,
    even if the combined samples look stationary when ignoring time.
    """
    dt_ns = 5_000_000  # 5 ms
    # 50 samples before gap (0.25s), then 5.0s gap, then 50 samples after gap (0.25s)
    # Total samples = 100 (which would form a 0.5s window if gap is ignored)
    ts_pre = np.arange(1_000_000, 1_000_000 + 50 * dt_ns, dt_ns, dtype=np.int64)
    t_gap_end = ts_pre[-1] + 5_000_000_000  # 5 second outage
    ts_post = t_gap_end + np.arange(dt_ns, 51 * dt_ns, dt_ns, dtype=np.int64)
    ts = np.concatenate([ts_pre, ts_post])

    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.zeros(len(ts)),
        "w_RS_S_y": np.zeros(len(ts)),
        "w_RS_S_z": np.zeros(len(ts)),
        "a_RS_S_x": np.zeros(len(ts)),
        "a_RS_S_y": np.zeros(len(ts)),
        "a_RS_S_z": np.full(len(ts), 9.80665),
    })

    # Window of 0.5s (100 samples) would encompass the gap
    res = detect_quasi_stationary_intervals(
        df,
        window_duration_s=0.5,
        min_continuous_duration_s=0.4,
        max_allowed_gap_s=0.01,
    )
    # All windows that bridge the gap must be rejected
    assert res["qualifying_window_count"] == 0
    assert len(res["detected_intervals"]) == 0


def test_bias_detector_does_not_count_outage_duration():
    """
    Verify that a 5-second outage between two short stationary intervals
    is NOT credited toward the minimum continuous duration threshold.
    """
    dt_ns = 5_000_000
    # Segment 1: 1.0s stationary (200 samples)
    ts_1 = np.arange(1_000_000, 1_000_000 + 200 * dt_ns, dt_ns, dtype=np.int64)
    # 5.0-second gap
    gap_ns = 5_000_000_000
    ts_2 = ts_1[-1] + gap_ns + np.arange(dt_ns, 201 * dt_ns, dt_ns, dtype=np.int64)
    ts = np.concatenate([ts_1, ts_2])

    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.zeros(len(ts)),
        "w_RS_S_y": np.zeros(len(ts)),
        "w_RS_S_z": np.zeros(len(ts)),
        "a_RS_S_x": np.zeros(len(ts)),
        "a_RS_S_y": np.zeros(len(ts)),
        "a_RS_S_z": np.full(len(ts), 9.80665),
    })

    # min_continuous_duration_s = 2.0s
    # Each segment is only 1.0s. If outage time were counted (1.0 + 5.0 + 1.0 = 7.0s),
    # it would falsely qualify. With continuity protection, both segments fail (1.0s < 2.0s).
    res = estimate_gyro_bias(df, min_continuous_duration_s=2.0, max_allowed_gap_s=0.01)
    assert res["bias_estimated"] is False
    assert res["status"] == "NO_RELIABLE_INTERVAL"


def test_bias_detector_accepts_continuous_stationary_segment():
    """Verify that a genuine, continuous 2.5s stationary segment without gaps is accepted."""
    dt_ns = 5_000_000
    n_samples = 500  # 2.5s at 200 Hz
    ts = np.arange(1_000_000, 1_000_000 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    true_bias = np.array([0.01, -0.02, 0.03])
    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.full(n_samples, true_bias[0]),
        "w_RS_S_y": np.full(n_samples, true_bias[1]),
        "w_RS_S_z": np.full(n_samples, true_bias[2]),
        "a_RS_S_x": np.zeros(n_samples),
        "a_RS_S_y": np.zeros(n_samples),
        "a_RS_S_z": np.full(n_samples, 9.80665),
    })

    res = estimate_gyro_bias(df, min_continuous_duration_s=2.0, max_allowed_gap_s=0.01)
    assert res["bias_estimated"] is True
    assert res["status"] == "CONFIRMED_QUASI_STATIONARY"
    np.testing.assert_allclose(res["gyro_bias"], true_bias, atol=1e-10)


def test_bias_detector_respects_configurable_gap_threshold():
    """Verify that user-configured max_allowed_gap_s is strictly enforced."""
    dt_ns = 5_000_000
    # Two samples with 20ms gap (0.02s)
    ts = np.arange(0, 400 * dt_ns, dt_ns, dtype=np.int64)
    ts[200:] += int(20e6)  # inject 20ms gap

    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.zeros(len(ts)),
        "w_RS_S_y": np.zeros(len(ts)),
        "w_RS_S_z": np.zeros(len(ts)),
        "a_RS_S_x": np.zeros(len(ts)),
        "a_RS_S_y": np.zeros(len(ts)),
        "a_RS_S_z": np.full(len(ts), 9.80665),
    })

    # Strict threshold 0.01s: gap of 0.02s splits/rejects
    res_strict = detect_quasi_stationary_intervals(df, min_continuous_duration_s=1.5, max_allowed_gap_s=0.01)
    # Neither split subsegment reaches 1.5s (each is ~1.0s)
    assert len(res_strict["detected_intervals"]) == 0

    # Permissive threshold 0.05s: gap of 0.02s is tolerated
    res_permissive = detect_quasi_stationary_intervals(df, min_continuous_duration_s=1.5, max_allowed_gap_s=0.05)
    assert len(res_permissive["detected_intervals"]) > 0


def test_bias_detector_outage_inside_candidate_stationary_interval():
    """
    Test scenario: An outage (e.g. 500 ms drop) occurs right in the middle of
    an otherwise stationary candidate interval.
    The detector must split the interval into two disjoint sub-intervals and ensure
    that the outage duration is not counted, preventing false certification.
    """
    dt_ns = 5_000_000  # 5 ms
    # 300 samples (1.5s) stationary
    ts_1 = np.arange(1_000_000, 1_000_000 + 300 * dt_ns, dt_ns, dtype=np.int64)
    # 500 ms outage (100 missing frames)
    t_gap = 500_000_000
    ts_2 = ts_1[-1] + t_gap + np.arange(dt_ns, 301 * dt_ns, dt_ns, dtype=np.int64)
    ts = np.concatenate([ts_1, ts_2])

    df = pd.DataFrame({
        "timestamp": ts,
        "w_RS_S_x": np.zeros(len(ts)),
        "w_RS_S_y": np.zeros(len(ts)),
        "w_RS_S_z": np.zeros(len(ts)),
        "a_RS_S_x": np.zeros(len(ts)),
        "a_RS_S_y": np.zeros(len(ts)),
        "a_RS_S_z": np.full(len(ts), 9.80665),
    })

    # Total nominal duration across span is 1.5s + 0.5s + 1.5s = 3.5s.
    # However, each continuous sub-interval is only 1.5s.
    # When min_continuous_duration_s=2.0s is required, the dataset must be rejected!
    res_reject = estimate_gyro_bias(df, min_continuous_duration_s=2.0, max_allowed_gap_s=0.01)
    assert res_reject["bias_estimated"] is False
    assert res_reject["status"] == "NO_RELIABLE_INTERVAL"

    # If min_continuous_duration_s=1.0s, the detector should find 2 separate intervals
    det = detect_quasi_stationary_intervals(df, min_continuous_duration_s=1.0, max_allowed_gap_s=0.01)
    assert len(det["detected_intervals"]) == 2
    # Neither interval should exceed ~1.5s
    for iv in det["detected_intervals"]:
        assert iv[4] <= 1.55


def test_bias_detector_tolerates_isolated_single_frame_drops():
    """
    Test scenario: Isolated dropped frames that cause dt to rise to e.g. 10 ms (2x nominal 5 ms).
    When max_allowed_gap_s is set to tolerate isolated drops (e.g. 15 ms),
    the stationary window remains qualified and accurately recovers bias.
    """
    dt_ns = 5_000_000
    n_samples = 600  # 3.0s
    ts = np.arange(1_000_000, 1_000_000 + n_samples * dt_ns, dt_ns, dtype=np.int64)

    # Drop 5 isolated frames (e.g. at sample 50, 120, 200, 310, 450)
    drop_indices = [50, 120, 200, 310, 450]
    keep_mask = np.ones(n_samples, dtype=bool)
    keep_mask[drop_indices] = False
    ts_dropped = ts[keep_mask]

    true_bias = np.array([0.005, -0.010, 0.020])
    df = pd.DataFrame({
        "timestamp": ts_dropped,
        "w_RS_S_x": np.full(len(ts_dropped), true_bias[0]),
        "w_RS_S_y": np.full(len(ts_dropped), true_bias[1]),
        "w_RS_S_z": np.full(len(ts_dropped), true_bias[2]),
        "a_RS_S_x": np.zeros(len(ts_dropped)),
        "a_RS_S_y": np.zeros(len(ts_dropped)),
        "a_RS_S_z": np.full(len(ts_dropped), 9.80665),
    })

    # max_allowed_gap_s = 0.015 (15 ms) tolerates single dropped frame (dt = 10 ms)
    res = estimate_gyro_bias(df, min_continuous_duration_s=2.0, max_allowed_gap_s=0.015)
    assert res["bias_estimated"] is True
    assert res["status"] == "CONFIRMED_QUASI_STATIONARY"
    np.testing.assert_allclose(res["gyro_bias"], true_bias, atol=1e-10)


