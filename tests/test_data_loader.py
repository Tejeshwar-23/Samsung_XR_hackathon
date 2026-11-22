"""
Comprehensive unit tests for src/data_loader.py.

Tests cover:
- Sample counts and column extraction from actual MH_01_easy dataset.
- Preservation of int64 nanosecond timestamps.
- Finite float64 conversion for sensor readings.
- Monotonicity checks (strictly increasing timestamps).
- Quaternion w, x, y, z order and unit norm validation.
- Error handling for missing files, missing columns, non-monotonic timestamps,
  duplicate timestamps, non-finite values, and invalid quaternion norms using temporary CSVs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    load_imu_data,
    load_groundtruth_data,
    IMU_ALL_COLS,
    GT_ALL_COLS,
    GT_QUATERNION_COLS,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def imu_actual_path(project_root) -> Path:
    return project_root / "data/raw/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv"


@pytest.fixture
def gt_actual_path(project_root) -> Path:
    return (
        project_root
        / "data/raw/euroc/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    )


# ==============================================================================
# TESTS ON ACTUAL MH_01_easy DATASET
# ==============================================================================


def test_load_imu_actual_dataset(imu_actual_path: Path):
    """Test loading and validating the actual EuRoC MH_01_easy IMU data."""
    df = load_imu_data(imu_actual_path)

    # 1. Correct sample count
    assert len(df) == 36820

    # 2. Correct column extraction
    assert list(df.columns) == IMU_ALL_COLS

    # 3. Timestamp dtype is int64
    assert df["timestamp"].dtype == np.int64

    # 4. Sensor values are float64 and all finite
    sensor_cols = [c for c in IMU_ALL_COLS if c != "timestamp"]
    for col in sensor_cols:
        assert df[col].dtype == np.float64
        assert np.isfinite(df[col]).all()

    # 5. Timestamps are strictly increasing
    dt = df["timestamp"].diff().iloc[1:]
    assert (dt > 0).all()


def test_load_groundtruth_actual_dataset(gt_actual_path: Path):
    """Test loading and validating the actual EuRoC MH_01_easy Ground Truth data."""
    df = load_groundtruth_data(gt_actual_path)

    # 1. Correct sample count
    assert len(df) == 36382

    # 2. Correct column extraction
    assert list(df.columns) == GT_ALL_COLS

    # 3. Timestamp dtype is int64
    assert df["timestamp"].dtype == np.int64

    # 4. Float64 conversion and finite check
    non_ts_cols = [c for c in GT_ALL_COLS if c != "timestamp"]
    for col in non_ts_cols:
        assert df[col].dtype == np.float64
        assert np.isfinite(df[col]).all()

    # 5. Strictly increasing timestamps
    dt = df["timestamp"].diff().iloc[1:]
    assert (dt > 0).all()

    # 6. Quaternion columns retained in w, x, y, z order and normalized
    assert GT_QUATERNION_COLS == ["q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z"]
    for col in GT_QUATERNION_COLS:
        assert col in df.columns

    q_vals = df[GT_QUATERNION_COLS].to_numpy()
    norms = np.linalg.norm(q_vals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


# ==============================================================================
# TESTS WITH SYNTHETIC & MALFORMED DATA
# ==============================================================================


def test_missing_files():
    """Verify FileNotFoundError when input files do not exist."""
    with pytest.raises(FileNotFoundError, match="IMU file not found"):
        load_imu_data(Path("non_existent_imu.csv"))

    with pytest.raises(FileNotFoundError, match="Ground truth file not found"):
        load_groundtruth_data(Path("non_existent_gt.csv"))


def test_imu_missing_columns(tmp_path: Path):
    """Verify ValueError when required IMU columns are missing."""
    file = tmp_path / "bad_imu.csv"
    # Missing accelerometer columns
    file.write_text("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z\n100,0.1,0.2,0.3\n")

    with pytest.raises(ValueError, match="Missing required IMU column"):
        load_imu_data(file)


def test_imu_non_monotonic_timestamps(tmp_path: Path):
    """Verify ValueError when timestamps are non-monotonic."""
    file = tmp_path / "non_monotonic_imu.csv"
    data = (
        "#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n"
        "1000,0,0,0,0,0,9.81\n"
        "999,0,0,0,0,0,9.81\n"  # decreased timestamp
    )
    file.write_text(data)

    with pytest.raises(ValueError, match="not strictly increasing"):
        load_imu_data(file)


def test_imu_duplicate_timestamps(tmp_path: Path):
    """Verify ValueError when timestamps contain duplicates."""
    file = tmp_path / "dup_imu.csv"
    data = (
        "#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n"
        "1000,0,0,0,0,0,9.81\n"
        "1000,0,0,0,0,0,9.81\n"  # duplicate timestamp
    )
    file.write_text(data)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_imu_data(file)


def test_imu_non_finite_values(tmp_path: Path):
    """Verify ValueError when sensor columns contain NaN or Inf."""
    file = tmp_path / "nan_imu.csv"
    data = (
        "#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n"
        "1000,0,0,0,NaN,0,9.81\n"
        "2000,0,0,0,0,0,9.81\n"
    )
    file.write_text(data)

    with pytest.raises(ValueError, match="(missing values|non-finite)"):
        load_imu_data(file)


def test_groundtruth_invalid_quaternion_norm(tmp_path: Path):
    """Verify ValueError when quaternion norm deviates significantly from 1.0."""
    file = tmp_path / "bad_quat_gt.csv"
    # q = (2.0, 0, 0, 0) -> norm is 2.0 != 1.0
    header = (
        "#timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,"
        "q_RS_w,q_RS_x,q_RS_y,q_RS_z,"
        "v_RS_R_x,v_RS_R_y,v_RS_R_z,"
        "b_w_RS_S_x,b_w_RS_S_y,b_w_RS_S_z,"
        "b_a_RS_S_x,b_a_RS_S_y,b_a_RS_S_z\n"
    )
    row = "1000,0,0,0,2.0,0.0,0.0,0.0,0,0,0,0,0,0,0,0,0\n"
    file.write_text(header + row)

    with pytest.raises(ValueError, match="violate unit norm condition"):
        load_groundtruth_data(file, validate_quat_norm=True)


# ==============================================================================
# TESTS FOR FLEXIBLE HEADER NORMALIZATION & DUPLICATE PRESERVATION (TASK A & B)
# ==============================================================================


def test_imu_flexible_header_normalization(tmp_path: Path):
    """Verify loading CSV with alternative header names (e.g. gx, gy, gz, ax, ay, az)."""
    file = tmp_path / "tum_vi_style.csv"
    data = (
        "timestamp,ax,ay,az,gx,gy,gz\n"
        "1520530308000000000,0.1,0.2,9.8,0.01,0.02,0.03\n"
        "1520530308005000000,0.1,0.2,9.8,0.01,0.02,0.03\n"
    )
    file.write_text(data)

    df = load_imu_data(file)
    assert len(df) == 2
    assert list(df.columns) == IMU_ALL_COLS
    assert df["w_RS_S_x"].iloc[0] == 0.01
    assert df["a_RS_S_z"].iloc[0] == 9.8


def test_imu_headers_with_whitespace(tmp_path: Path):
    """Verify headers with leading/trailing whitespace are stripped cleanly."""
    file = tmp_path / "whitespace_headers.csv"
    data = (
        " timestamp , w_RS_S_x  ,  w_RS_S_y ,w_RS_S_z, a_RS_S_x ,a_RS_S_y,  a_RS_S_z \n"
        "1000,0.0,0.0,0.0,0.0,0.0,9.81\n"
        "2000,0.0,0.0,0.0,0.0,0.0,9.81\n"
    )
    file.write_text(data)

    df = load_imu_data(file)
    assert len(df) == 2
    assert list(df.columns) == IMU_ALL_COLS


def test_imu_allow_duplicates_preserves_rows(tmp_path: Path):
    """Verify allow_duplicates=True permits non-decreasing timestamps without dropping."""
    file = tmp_path / "dup_allowed.csv"
    data = (
        "timestamp,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n"
        "1000,0.0,0.0,0.0,0.0,0.0,9.81\n"
        "1000,0.0,0.0,0.0,0.0,0.0,9.81\n"  # duplicate
        "2000,0.0,0.0,0.0,0.0,0.0,9.81\n"
    )
    file.write_text(data)

    df = load_imu_data(file, allow_duplicates=True)
    assert len(df) == 3
    assert (df["timestamp"].to_numpy() == [1000, 1000, 2000]).all()
