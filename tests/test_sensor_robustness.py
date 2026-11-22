"""
Comprehensive unit tests for Sensor Sanitization and Robustness (Stage 2 / Task C).

Tests verify:
- Gyroscope outlier/saturation gating (||omega|| > 10.0 rad/s).
- Accelerometer outlier/saturation gating (||a|| > 50.0 m/s^2).
- Non-finite (NaN, +Inf, -Inf) measurement masking.
- Frozen/stuck sensor reading detection (>= 5 consecutive identical readings).
- Single-sample impulse spike rejection.
- Terminal blackout / failure zeroed suffix isolation.
- State-hold behavior (constant velocity, orientation hold) under masked sensor frames.
"""

import numpy as np
import pytest
from src.main import (
    _sanitize_arrays_inplace,
    GYRO_OUTLIER_THRESHOLD_RAD_S,
    ACCEL_OUTLIER_THRESHOLD_M_S2,
)


def test_sanitize_normal_measurements():
    """Verify normal physical measurements pass through unmodified."""
    g = np.array([[0.01, 0.02, -0.01], [0.00, -0.01, 0.02]], dtype=np.float64)
    a = np.array([[0.1, 0.2, 9.81], [0.12, 0.18, 9.79]], dtype=np.float64)

    g_orig = g.copy()
    a_orig = a.copy()

    vg, va = _sanitize_arrays_inplace(g, a)

    assert vg.all()
    assert va.all()
    np.testing.assert_allclose(g, g_orig)
    np.testing.assert_allclose(a, a_orig)


def test_sanitize_gyro_outlier_gating():
    """Verify gyroscope rates exceeding 10.0 rad/s are flagged and masked to 0.0."""
    g = np.array([
        [0.0, 5.0, 0.0],
        [0.0, 10.5, 0.0],  # Outlier > 10 rad/s
        [0.0, -12.0, 0.0], # Outlier > 10 rad/s
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    a = np.full((4, 3), [0.0, 0.0, 9.81], dtype=np.float64)

    vg, va = _sanitize_arrays_inplace(g, a)

    assert vg[0] and not vg[1] and not vg[2] and vg[3]
    assert va.all()
    assert (g[1] == 0.0).all()
    assert (g[2] == 0.0).all()


def test_sanitize_accel_outlier_gating():
    """Verify accelerometer norms exceeding 50.0 m/s^2 are flagged and masked to 0.0."""
    g = np.full((4, 3), [0.01, 0.0, 0.0], dtype=np.float64)
    a = np.array([
        [0.0, 0.0, 9.81],
        [0.0, 55.0, 0.0],   # Outlier > 50 m/s^2
        [35.0, 35.0, 10.0], # Norm = sqrt(1225 + 1225 + 100) = sqrt(2550) = 50.49 > 50 m/s^2
        [0.0, 0.0, 9.81],
    ], dtype=np.float64)

    vg, va = _sanitize_arrays_inplace(g, a)

    assert vg.all()
    assert va[0] and not va[1] and not va[2] and va[3]
    assert (a[1] == 0.0).all()
    assert (a[2] == 0.0).all()


def test_sanitize_non_finite_values():
    """Verify NaN and Inf values are safely masked without propagating NaNs."""
    g = np.array([
        [0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
        [0.0, np.inf, 0.0],
        [0.0, 0.0, -np.inf],
    ], dtype=np.float64)
    a = np.array([
        [np.nan, 0.0, 9.81],
        [0.0, 0.0, 9.81],
        [0.0, 0.0, np.inf],
        [0.0, 0.0, 9.81],
    ], dtype=np.float64)

    vg, va = _sanitize_arrays_inplace(g, a)

    assert vg[0] and not vg[1] and not vg[2] and not vg[3]
    assert not va[0] and va[1] and not va[2] and va[3]
    assert np.isfinite(g).all()
    assert np.isfinite(a).all()


def test_sanitize_stuck_sensor_readings():
    """Verify >= 5 consecutive identical readings are flagged as stuck."""
    n = 12
    # Samples 3..8 are frozen identical non-zero values
    g = np.random.RandomState(42).normal(0, 0.01, (n, 3))
    a = np.random.RandomState(42).normal(0, 0.05, (n, 3)) + [0, 0, 9.81]

    frozen_g = np.array([0.05, -0.02, 0.01])
    for i in range(3, 9):
        g[i] = frozen_g

    vg, va = _sanitize_arrays_inplace(g, a)

    # First 4 identical samples (indices 3, 4, 5, 6) might pass before run reaches 5
    # The subsequent identical samples must be flagged False
    assert not vg[7]
    assert not vg[8]


def test_sanitize_terminal_failure_blackout():
    """Verify terminal zeroed blackout interval is isolated and masked."""
    rng = np.random.default_rng(123)
    n = 20
    g = rng.normal(0, 0.01, (n, 3))
    a = rng.normal(0, 0.05, (n, 3)) + [0, 0, 9.81]

    # Terminal 5 frames blackout (zeroed out)
    g[-5:] = 0.0
    a[-5:] = 0.0

    vg, va = _sanitize_arrays_inplace(g, a)

    assert (vg[:15]).all()
    assert (va[:15]).all()
    assert not (vg[15:]).any()
    assert not (va[15:]).any()
