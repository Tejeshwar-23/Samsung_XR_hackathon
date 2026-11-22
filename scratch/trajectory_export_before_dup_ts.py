"""
Trajectory Export and Validation Module for Samsung XR Hackathon Track 3.
"""

from decimal import Decimal
import math
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd


TUM_COLUMN_NAMES = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]


def _format_timestamp_exact(ts_val: Union[int, float, np.integer, np.floating, str]) -> Tuple[str, Decimal]:
    if isinstance(ts_val, (int, np.integer)):
        val_int = int(ts_val)
        if abs(val_int) > 100_000_000_000:
            sec = val_int // 1_000_000_000
            nsec = val_int % 1_000_000_000
            str_rep = f"{sec}.{nsec:09d}"
            return str_rep, Decimal(str_rep)
        else:
            str_rep = f"{val_int}.000000000"
            return str_rep, Decimal(str_rep)

    val_float = float(ts_val)
    if abs(val_float) > 1e11:
        val_int = int(round(val_float))
        sec = val_int // 1_000_000_000
        nsec = val_int % 1_000_000_000
        str_rep = f"{sec}.{nsec:09d}"
        return str_rep, Decimal(str_rep)
    else:
        dec = Decimal(str(val_float)) if isinstance(ts_val, str) else Decimal(f"{val_float:.9f}")
        str_rep = f"{dec:.9f}"
        return str_rep, dec


def export_tum_trajectory(
    df_trajectory: Optional[pd.DataFrame] = None,
    output_path: Union[str, Path] = "outputs/estimated_trajectory.txt",
    time_col: str = "timestamp",
    pos_cols: Tuple[str, str, str] = ("p_x", "p_y", "p_z"),
    quat_cols_wxyz: Tuple[str, str, str, str] = ("q_w", "q_x", "q_y", "q_z"),
    quat_norm_tolerance: float = 1e-4,
    write_header_comment: bool = True,
    timestamps_ns: Optional[np.ndarray] = None,
    pos_arr: Optional[np.ndarray] = None,
    quats_wxyz: Optional[np.ndarray] = None,
) -> Path:
    if timestamps_ns is not None and pos_arr is not None and quats_wxyz is not None:
        raw_ts = np.asarray(timestamps_ns)
        pos_data = np.asarray(pos_arr, dtype=np.float64)
        q_in = np.asarray(quats_wxyz, dtype=np.float64)
        n_samples = len(raw_ts)
        if n_samples == 0:
            raise ValueError("Cannot export empty trajectory arrays.")
        if len(pos_data) != n_samples or len(q_in) != n_samples or q_in.ndim != 2 or q_in.shape[1] != 4:
            raise ValueError(f"Invalid array shapes: ts={n_samples}, pos={pos_data.shape}, quats={q_in.shape}")
        if not np.isfinite(raw_ts).all():
            raise ValueError("Trajectory contains non-finite timestamp values.")
        if not np.isfinite(pos_data).all() or not np.isfinite(q_in).all():
            raise ValueError("Trajectory position or quaternion arrays contain non-finite values (NaN or Inf).")
        q_w = q_in[:, 0].copy()
        q_x = q_in[:, 1].copy()
        q_y = q_in[:, 2].copy()
        q_z = q_in[:, 3].copy()
    elif df_trajectory is not None:
        if df_trajectory.empty:
            raise ValueError("Cannot export empty trajectory DataFrame.")
        required_cols = [time_col] + list(pos_cols) + list(quat_cols_wxyz)
        for col in required_cols:
            if col not in df_trajectory.columns:
                raise ValueError(f"Required column '{col}' missing from trajectory DataFrame.")
        n_samples = len(df_trajectory)
        raw_ts = df_trajectory[time_col].to_numpy()
        pos_data = df_trajectory[list(pos_cols)].to_numpy(dtype=np.float64)
        q_w = df_trajectory[quat_cols_wxyz[0]].to_numpy(dtype=np.float64, copy=True)
        q_x = df_trajectory[quat_cols_wxyz[1]].to_numpy(dtype=np.float64, copy=True)
        q_y = df_trajectory[quat_cols_wxyz[2]].to_numpy(dtype=np.float64, copy=True)
        q_z = df_trajectory[quat_cols_wxyz[3]].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(raw_ts).all():
            raise ValueError("Trajectory contains non-finite timestamp values.")
        if not np.isfinite(pos_data).all():
            raise ValueError("Trajectory position columns contain non-finite values (NaN or Inf).")
        if not (np.isfinite(q_w).all() and np.isfinite(q_x).all() and np.isfinite(q_y).all() and np.isfinite(q_z).all()):
            raise ValueError("Trajectory quaternion columns contain non-finite values (NaN or Inf).")
    else:
        raise ValueError("Must provide either (timestamps_ns, pos_arr, quats_wxyz) or df_trajectory.")

    norms = np.sqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z)
    if (norms < 1e-6).any():
        raise ValueError("Found zero-norm or near-zero quaternion in trajectory.")
    inv_norms = 1.0 / norms
    q_w *= inv_norms
    q_x *= inv_norms
    q_y *= inv_norms
    q_z *= inv_norms
    del norms, inv_norms

    ts_is_int_or_ns = isinstance(raw_ts[0], (int, np.integer)) or (
        isinstance(raw_ts[0], (float, np.floating)) and abs(float(raw_ts[0])) > 1e11
    )
    if ts_is_int_or_ns:
        ts_cmp = raw_ts.astype(np.int64, copy=False)
        if n_samples > 1 and np.any(ts_cmp[1:] <= ts_cmp[:-1]):
            raise ValueError("Trajectory timestamps are not strictly increasing.")
    else:
        ts_cmp = raw_ts.astype(np.float64, copy=False)
        if n_samples > 1 and np.any(ts_cmp[1:] <= ts_cmp[:-1]):
            raise ValueError("Trajectory timestamps are not strictly increasing.")

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = 1024
    with open(out_file, "w", encoding="utf-8") as f:
        if write_header_comment:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            px_s = pos_data[start:end, 0].tolist()
            py_s = pos_data[start:end, 1].tolist()
            pz_s = pos_data[start:end, 2].tolist()
            qw_s = q_w[start:end].tolist()
            qx_s = q_x[start:end].tolist()
            qy_s = q_y[start:end].tolist()
            qz_s = q_z[start:end].tolist()
            if ts_is_int_or_ns:
                ts_c = ts_cmp[start:end].tolist()
                lines_chunk = [
                    f"{v // 1_000_000_000}.{v % 1_000_000_000:09d} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                    for v, x, y, z, qw, qx, qy, qz in zip(ts_c, px_s, py_s, pz_s, qw_s, qx_s, qy_s, qz_s)
                ]
                del ts_c
            else:
                ts_c = ts_cmp[start:end].tolist()
                lines_chunk = [
                    f"{v:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                    for v, x, y, z, qw, qx, qy, qz in zip(ts_c, px_s, py_s, pz_s, qw_s, qx_s, qy_s, qz_s)
                ]
                del ts_c
            f.writelines(lines_chunk)
            del px_s, py_s, pz_s, qw_s, qx_s, qy_s, qz_s, lines_chunk

    return out_file


def validate_tum_file(
    file_path: Union[str, Path],
    expected_row_count: Optional[int] = None,
    quat_norm_tolerance: float = 1e-3,
) -> Dict[str, Any]:
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TUM trajectory file not found: {path}")

    row_count = 0
    first_ts_str = None
    last_ts_str = None
    prev_ts_val = None
    max_dev = 0.0

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) != 8:
                raise ValueError(f"Line {line_idx+1} in TUM file has {len(parts)} columns; expected exactly 8.")

            try:
                ts_val = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            except Exception as exc:
                raise ValueError(f"Failed to parse numeric values on line {line_idx+1}: {exc}") from exc

            if not (math.isfinite(ts_val) and math.isfinite(tx) and math.isfinite(ty) and math.isfinite(tz) and
                    math.isfinite(qx) and math.isfinite(qy) and math.isfinite(qz) and math.isfinite(qw)):
                raise ValueError(f"Line {line_idx+1} contains non-finite values (NaN or Inf).")

            q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            dev = abs(q_norm - 1.0)
            if dev > max_dev:
                max_dev = dev
            if dev > quat_norm_tolerance:
                raise ValueError(
                    f"Quaternion norm error {dev:.6e} exceeds tolerance {quat_norm_tolerance} at line {line_idx+1}."
                )

            if prev_ts_val is not None and ts_val <= prev_ts_val:
                raise ValueError(
                    f"TUM trajectory timestamps are not strictly increasing at line {line_idx+1}: "
                    f"{prev_ts_val} -> {ts_val}."
                )

            if first_ts_str is None:
                first_ts_str = parts[0]
            last_ts_str = parts[0]
            prev_ts_val = ts_val
            row_count += 1

    if row_count == 0:
        raise ValueError(f"TUM trajectory file is empty or contains only comments: {path}")

    if expected_row_count is not None and row_count != expected_row_count:
        raise ValueError(
            f"Row count mismatch in TUM file: expected {expected_row_count} rows, found {row_count} rows."
        )

    first_ts_dec = Decimal(first_ts_str)
    last_ts_dec = Decimal(last_ts_str)
    duration_dec = last_ts_dec - first_ts_dec
    duration_s = float(duration_dec)

    return {
        "valid": True,
        "row_count": row_count,
        "duration_s": duration_s,
        "duration_decimal_s": str(duration_dec),
        "timestamp_start_s": float(first_ts_dec),
        "timestamp_end_s": float(last_ts_dec),
        "max_quat_norm_deviation": max_dev,
    }


def extrapolate_terminal_trajectory(
    df_trajectory: pd.DataFrame,
    duration_s: float = 5.0,
    nominal_rate_hz: float = 200.0,
    time_col: str = "timestamp",
    pos_cols: Tuple[str, str, str] = ("p_x", "p_y", "p_z"),
    vel_cols: Tuple[str, str, str] = ("v_x", "v_y", "v_z"),
    quat_cols: Tuple[str, str, str, str] = ("q_w", "q_x", "q_y", "q_z"),
) -> pd.DataFrame:
    if df_trajectory.empty:
        raise ValueError("Cannot extrapolate empty trajectory DataFrame.")
    if duration_s <= 0.0:
        raise ValueError(f"Extrapolation duration must be strictly positive, got {duration_s}.")
    if nominal_rate_hz <= 0.0:
        raise ValueError(f"Nominal sampling rate must be strictly positive, got {nominal_rate_hz}.")

    required = [time_col] + list(pos_cols) + list(vel_cols) + list(quat_cols)
    for col in required:
        if col not in df_trajectory.columns:
            raise ValueError(f"Required column '{col}' missing from trajectory DataFrame for extrapolation.")

    dt_step_s = 1.0 / float(nominal_rate_hz)
    dt_step_ns = int(round(dt_step_s * 1e9))
    num_steps = int(round(duration_s * nominal_rate_hz))

    last_row = df_trajectory.iloc[-1]
    last_ts_ns = int(last_row[time_col])

    last_pos = last_row[list(pos_cols)].to_numpy(dtype=np.float64)
    last_vel = last_row[list(vel_cols)].to_numpy(dtype=np.float64)
    last_quat = last_row[list(quat_cols)].to_numpy(dtype=np.float64)

    future_ts_ns = last_ts_ns + np.arange(1, num_steps + 1, dtype=np.int64) * dt_step_ns
    dt_future_s = np.arange(1, num_steps + 1, dtype=np.float64) * dt_step_s
    future_pos = last_pos + np.outer(dt_future_s, last_vel)
    future_vel = np.tile(last_vel, (num_steps, 1))
    future_quat = np.tile(last_quat, (num_steps, 1))

    extrap_data = {
        time_col: future_ts_ns,
        pos_cols[0]: future_pos[:, 0],
        pos_cols[1]: future_pos[:, 1],
        pos_cols[2]: future_pos[:, 2],
        vel_cols[0]: future_vel[:, 0],
        vel_cols[1]: future_vel[:, 1],
        vel_cols[2]: future_vel[:, 2],
        quat_cols[0]: future_quat[:, 0],
        quat_cols[1]: future_quat[:, 1],
        quat_cols[2]: future_quat[:, 2],
        quat_cols[3]: future_quat[:, 3],
    }

    for col in df_trajectory.columns:
        if col not in extrap_data:
            extrap_data[col] = [last_row[col]] * num_steps

    df_extrap = pd.DataFrame(extrap_data)
    return pd.concat([df_trajectory, df_extrap], ignore_index=True)
