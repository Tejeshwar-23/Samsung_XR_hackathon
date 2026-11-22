"""
Trajectory estimation module for Samsung XR Hackathon Track 3 (IMU Dead-Reckoning).

Provides modular, high-precision strapdown inertial navigation algorithms:
1. Coordinate Frames & Conventions:
   - World frame R: Cartesian frame with Z-axis pointing vertically UPWARD opposing gravity.
     Nominal gravity vector: g_R = [0, 0, -9.80665]^T (m/s^2).
   - Sensor/body frame S: Right-handed coordinate frame of the IMU.
   - Specific force measurement: f_S = R_{RS}^T (a_R - g_R) + b_a + eta_a.
     When stationary at level attitude: f_S = [0, 0, +9.80665]^T.
   - Body-to-world rotation: a_R = R_{RS} * a_S.

2. Initial Gravity Alignment:
   - Analytical roll (phi) and pitch (theta) from stationary mean specific force.
   - Yaw (psi) is fundamentally unobservable from gravity alone; explicitly defaulted to 0.0.
   - Executed only when stationarity is sufficiently supported.

3. Accelerometer Bias Observability & Fallback:
   - In a single static/quasi-stationary attitude, only the 1D bias component along
     the gravity vector is observable (deviation of ||f_S|| from 9.80665 m/s^2).
   - Horizontal orthogonal components are indistinguishable from sensor tilt.
   - Safe fallback: Zero bias correction (default). Full 3-axis bias recovery is not claimed.

4. Zero-Velocity Update (ZUPT):
   - Quasi-stationarity does NOT prove zero translational velocity.
   - ZUPT is DISABLED BY DEFAULT (enable_zupt=False).

5. Outage Handling & Sensor Degradation:
   - Detects timestamp gaps (dt > gap_threshold_s).
   - Does NOT integrate acceleration across the missing interval.
   - Holds velocity constant: v_k = v_{k-1}.
   - Linearly propagates position using pre-gap velocity: p_k = p_{k-1} + v_{k-1} * dt.
   - Resumes normal trapezoidal integration upon recovery.
   - Exactly one output row per input IMU sample (1:1 frame count).
"""

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

try:
    from src.data_loader import IMU_TIMESTAMP_COL
    from src.preprocessing import ACCEL_COLS
except ModuleNotFoundError:
    from data_loader import IMU_TIMESTAMP_COL
    from preprocessing import ACCEL_COLS


STANDARD_GRAVITY = 9.80665  # m/s^2

TRAJ_POSITION_COLS = ["p_x", "p_y", "p_z"]
TRAJ_VELOCITY_COLS = ["v_x", "v_y", "v_z"]
TRAJ_ACCEL_WORLD_COLS = ["a_lin_x", "a_lin_y", "a_lin_z"]


def compute_gravity_alignment_quaternion(
    mean_accel: np.ndarray,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    """
    Computes initial orientation quaternion [w, x, y, z] from stationary accelerometer measurements.

    Mathematical Formulation & Derivation:
    - In the level world frame (R) with +Z pointing UP opposing gravity:
        f_R = [0, 0, +||f_S||]^T
    - The body-to-world rotation R_{RS} maps measured specific force f_S to f_R:
        f_R = R_{RS} * f_S
    - Let u_S = f_S / ||f_S|| be the unit body vector along measured reaction force.
    - Let z_R = [0, 0, 1]^T be the world +Z unit vector.
    - We want a rotation R_align such that:
        z_R = R_align * u_S
      Using the shortest arc rotation (axis v = u_S x z_R, angle alpha = acos(u_S . z_R)):
      Or using half-angle quaternion representation:
        q_align has scalar w = 1 + u_S . z_R, vector xyz = u_S x z_R (then normalized).
      If u_S is anti-parallel to z_R (pointing straight down [0, 0, -1]), rotate 180 deg around X.
    - Unobservable Heading (Yaw):
      Any subsequent rotation around the world Z axis (z_R) leaves the vertical alignment
      completely invariant:
        R_{RS} = R_z(yaw) * R_align
      Since R_z(yaw) rotates around z_R = [0, 0, 1]^T, and R_align * u_S = [0, 0, 1]^T:
        R_{RS} * u_S = R_z(yaw) * (R_align * u_S) = R_z(yaw) * [0, 0, 1]^T = [0, 0, 1]^T.
      Therefore:
        R_{RS} * f_S = [0, 0, ||f_S||]^T  (IDENTICALLY FOR ANY YAW ANGLE).

    Parameters
    ----------
    mean_accel : np.ndarray
        3-element array of measured stationary specific force [f_x, f_y, f_z] in m/s^2.
    yaw_rad : float
        Initial yaw angle in radians (default: 0.0, as yaw is unobservable from gravity).

    Returns
    -------
    np.ndarray
        Unit quaternion [w, x, y, z] representing body-to-world rotation R_{RS}(0).
    """
    f = np.asarray(mean_accel, dtype=np.float64)
    norm_f = float(np.linalg.norm(f))
    if norm_f < 1e-6:
        raise ValueError("Cannot perform gravity alignment: accelerometer magnitude is near zero.")

    u_S = f / norm_f
    z_R = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    dot = float(np.dot(u_S, z_R))

    if dot > 1.0 - 1e-12:
        # Already aligned with +Z
        q_align_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    elif dot < -1.0 + 1e-12:
        # Anti-parallel: 180 deg flip around X axis
        q_align_xyzw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        # Rotation vector from cross product u_S x z_R
        # Axis of rotation: u_S x z_R = [u_y, -u_x, 0]
        # Half-angle formula: w = 1 + dot, xyz = cross(u_S, z_R)
        cross_vec = np.cross(u_S, z_R)
        q_align_xyzw = np.array([cross_vec[0], cross_vec[1], cross_vec[2], 1.0 + dot], dtype=np.float64)
        q_align_xyzw /= np.linalg.norm(q_align_xyzw)

    r_align = Rotation.from_quat(q_align_xyzw)

    # Apply yaw rotation around world Z: R_total = R_z(yaw) * R_align
    if abs(yaw_rad) > 1e-12:
        r_yaw = Rotation.from_euler("z", yaw_rad)
        r_total = r_yaw * r_align
    else:
        r_total = r_align

    q_total_xyzw = r_total.as_quat()  # [x, y, z, w]
    # Return canonical [w, x, y, z]
    q_wxyz = np.array([q_total_xyzw[3], q_total_xyzw[0], q_total_xyzw[1], q_total_xyzw[2]], dtype=np.float64)
    q_wxyz /= np.linalg.norm(q_wxyz)
    return q_wxyz


def estimate_accel_bias_gravity_aligned(
    mean_accel: np.ndarray,
    nominal_g: float = STANDARD_GRAVITY,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Estimates the 1D gravity-parallel accelerometer bias component from a single stationary interval.

    Observability Explanation:
    - From a single stationary pose, the true 3-axis accelerometer bias is mathematically
      unobservable without independent attitude reference.
    - Only the scalar projection along the gravity vector alters the magnitude of specific force:
        ||f_S|| ≈ g + b_{a, parallel}
    - Orthogonal components b_{a, perp} alter the direction of specific force and are
      indistinguishable from a physical tilt angle error.
    - Full 3-axis bias estimation requires multiple non-coplanar static attitudes (e.g. 6-position tumble).
    - This function computes ONLY the observable scalar norm deviation and projects it along the measured gravity direction.

    Parameters
    ----------
    mean_accel : np.ndarray
        Mean measured specific force vector during stationary interval [3,].
    nominal_g : float
        Nominal local gravity magnitude (default: 9.80665 m/s^2).

    Returns
    -------
    bias_vec : np.ndarray
        Estimated 3D bias vector [b_x, b_y, b_z] (parallel component only).
    diagnostics : dict
        Details on measured norm, scalar bias, and observability notice.
    """
    f = np.asarray(mean_accel, dtype=np.float64)
    norm_f = np.linalg.norm(f)
    if norm_f < 1e-6:
        return np.zeros(3, dtype=np.float64), {"status": "FAILED_ZERO_NORM"}

    unit_f = f / norm_f
    scalar_bias = float(norm_f - nominal_g)
    bias_vec = scalar_bias * unit_f

    diagnostics = {
        "status": "SUCCESS_1D_OBSERVABLE_ONLY",
        "measured_norm": float(norm_f),
        "nominal_g": float(nominal_g),
        "scalar_bias": scalar_bias,
        "bias_vector": bias_vec.copy(),
        "observability_notice": (
            "Only 1D gravity-parallel bias is observable from a single stationary pose. "
            "Orthogonal horizontal components are unobservable and assumed zero."
        ),
    }
    return bias_vec, diagnostics


def rotate_and_compensate_gravity(
    accel_body: np.ndarray,
    quaternions_wxyz: np.ndarray,
    accel_bias: Optional[np.ndarray] = None,
    gravity_world: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rotates body-frame specific force into the world frame and cancels gravity.

    Formulation:
    - Given body-to-world rotation R_{RS} (represented by quaternions_wxyz):
      f_R = R_{RS} * (a_body - b_a)
    - Linear acceleration in world frame:
      a_lin_R = f_R + g_R (where g_R = [0, 0, -9.80665]^T)
      Note: When at rest, f_R = [0, 0, +g]^T, so a_lin_R = [0, 0, 0]^T.

    Parameters
    ----------
    accel_body : np.ndarray
        Shape (N, 3) body-frame accelerometer measurements.
    quaternions_wxyz : np.ndarray
        Shape (N, 4) body-to-world unit quaternions in [w, x, y, z] convention.
        Must be a 2D NumPy array with shape (N, 4). DataFrames or other shapes are rejected.
    accel_bias : np.ndarray, optional
        Shape (3,) accelerometer bias vector. Defaults to zero vector.
    gravity_world : np.ndarray, optional
        Shape (3,) gravity vector in world frame. Defaults to [0, 0, -9.80665] m/s^2.

    Returns
    -------
    np.ndarray
        Shape (N, 3) linear acceleration in world frame (m/s^2).
    """
    if isinstance(quaternions_wxyz, pd.DataFrame):
        raise ValueError(
            "quaternions_wxyz must be a NumPy array of shape (N, 4), not a pandas DataFrame. "
            "Please explicitly extract the quaternion columns (e.g. df[EST_QUATERNION_COLS].to_numpy())."
        )

    quats = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(
            f"quaternions_wxyz must be a 2D array of shape (N, 4). "
            f"Got shape {quats.shape} (ndim={quats.ndim})."
        )

    if not np.isfinite(quats).all():
        raise ValueError("quaternions_wxyz contains non-finite values (NaN or Inf).")

    quat_norms = np.linalg.norm(quats, axis=1)
    if (quat_norms < 1e-6).any():
        raise ValueError("Found zero-norm or near-zero-norm quaternions in quaternions_wxyz.")

    accel_body = np.asarray(accel_body, dtype=np.float64)
    if accel_body.ndim != 2 or accel_body.shape[1] != 3:
        raise ValueError(
            f"accel_body must be a 2D array of shape (N, 3). Got shape {accel_body.shape}."
        )

    if len(accel_body) != len(quats):
        raise ValueError(
            f"Length mismatch: accel_body has {len(accel_body)} rows, but quats has {len(quats)} rows."
        )

    if accel_bias is None:
        accel_bias = np.zeros(3, dtype=np.float64)
    else:
        accel_bias = np.asarray(accel_bias, dtype=np.float64)

    if gravity_world is None:
        gravity_world = np.array([0.0, 0.0, -STANDARD_GRAVITY], dtype=np.float64)
    else:
        gravity_world = np.asarray(gravity_world, dtype=np.float64)

    # Convert quats [w, x, y, z] to SciPy [x, y, z, w]
    quats_xyzw = np.column_stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]])
    rot = Rotation.from_quat(quats_xyzw)

    # Specific force with bias subtracted
    f_body = accel_body - accel_bias

    # Rotate into world frame: f_R = R_{RS} * f_body
    f_world = rot.apply(f_body)

    # Linear acceleration: a_lin_R = f_world + g_world
    # Since g_world = [0, 0, -g], this adds -g in Z: a_lin = f_world - [0, 0, g]
    a_lin_world = f_world + gravity_world
    return a_lin_world


def integrate_trajectory(
    df_imu: pd.DataFrame,
    quaternions_wxyz: np.ndarray,
    initial_position: Optional[np.ndarray] = None,
    initial_velocity: Optional[np.ndarray] = None,
    accel_bias: Optional[np.ndarray] = None,
    gap_threshold_s: Optional[float] = None,
    enable_zupt: bool = False,
    zupt_mask: Optional[np.ndarray] = None,
    gravity_world: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Integrates linear acceleration to compute 3D velocity and position over time.

    Sensor Degradation & Outage Policy:
    - Uses actual timestamps from df_imu['timestamp'].
    - For normal continuous samples (dt <= gap_threshold_s):
        * Trapezoidal integration of linear acceleration to velocity:
          v_k = v_{k-1} + 0.5 * (a_{k-1} + a_k) * dt
        * Trapezoidal integration of velocity to position:
          p_k = p_{k-1} + 0.5 * (v_{k-1} + v_k) * dt
    - For outage transitions (dt > gap_threshold_s):
        * Acceleration is NOT integrated across the missing gap.
        * Velocity is held constant: v_k = v_{k-1}.
        * Position is propagated linearly using pre-gap velocity:
          p_k = p_{k-1} + v_{k-1} * dt.
        * Resumes normal trapezoidal integration starting with sample k+1.
    - Zero-Velocity Update (ZUPT):
        * Controlled by enable_zupt (False by default).
        * Low variance is NOT assumed to mean zero velocity.

    Parameters
    ----------
    df_imu : pd.DataFrame
        IMU DataFrame containing 'timestamp' (ns) and ['a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z'].
    quaternions_wxyz : np.ndarray
        Array of body-to-world unit quaternions [w, x, y, z] matching df_imu length.
    initial_position : np.ndarray, optional
        Initial 3D position [p_x0, p_y0, p_z0]. Defaults to [0, 0, 0].
    initial_velocity : np.ndarray, optional
        Initial 3D velocity [v_x0, v_y0, v_z0]. Defaults to [0, 0, 0].
    accel_bias : np.ndarray, optional
        3D accelerometer bias vector to subtract. Defaults to [0, 0, 0].
    gap_threshold_s : float, optional
        Outage detection threshold in seconds. Defaults to 1.5 * median(dt).
    enable_zupt : bool
        Whether ZUPT is active (default: False).
    zupt_mask : np.ndarray, optional
        Boolean array of length N where True indicates active ZUPT.
    gravity_world : np.ndarray, optional
        World gravity vector. Defaults to [0, 0, -9.80665].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['timestamp', 'p_x', 'p_y', 'p_z', 'v_x', 'v_y', 'v_z',
         'a_lin_x', 'a_lin_y', 'a_lin_z', 'is_gap']
    """
    n_samples = len(df_imu)
    if n_samples == 0:
        raise ValueError("Cannot integrate trajectory on empty DataFrame.")

    if isinstance(quaternions_wxyz, pd.DataFrame):
        raise ValueError(
            "quaternions_wxyz must be a NumPy array of shape (N, 4), not a pandas DataFrame. "
            "Please explicitly extract the quaternion columns (e.g. df[EST_QUATERNION_COLS].to_numpy())."
        )

    quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4:
        raise ValueError(
            f"quaternions_wxyz must be a 2D array of shape (N, 4). "
            f"Got shape {quaternions_wxyz.shape} (ndim={quaternions_wxyz.ndim})."
        )

    if len(quaternions_wxyz) != n_samples:
        raise ValueError(
            f"Length mismatch: IMU DataFrame has {n_samples} rows, "
            f"but quaternions has {len(quaternions_wxyz)} rows."
        )

    if not np.isfinite(quaternions_wxyz).all():
        raise ValueError("quaternions_wxyz contains non-finite values (NaN or Inf).")

    quat_norms = np.linalg.norm(quaternions_wxyz, axis=1)
    if (quat_norms < 1e-6).any():
        raise ValueError("Found zero-norm or near-zero-norm quaternions in quaternions_wxyz.")

    timestamps_ns = df_imu[IMU_TIMESTAMP_COL].to_numpy(dtype=np.int64)
    dt_ns = np.diff(timestamps_ns)
    if (dt_ns < 0).any():
        raise ValueError("IMU timestamps must be non-decreasing (decreasing timestamp detected).")

    dt_s = dt_ns / 1e9

    if gap_threshold_s is None:
        # Compute median dt ignoring duplicate zero deltas if possible
        pos_dt_s = dt_s[dt_s > 0]
        median_dt = float(np.median(pos_dt_s)) if len(pos_dt_s) > 0 else 0.005
        gap_threshold_s = 1.5 * median_dt

    p0 = np.zeros(3, dtype=np.float64) if initial_position is None else np.asarray(initial_position, dtype=np.float64)
    v0 = np.zeros(3, dtype=np.float64) if initial_velocity is None else np.asarray(initial_velocity, dtype=np.float64)

    # Accelerometer values
    accel_body = df_imu[ACCEL_COLS].to_numpy(dtype=np.float64)

    # Rotate and compensate gravity
    a_lin_world = rotate_and_compensate_gravity(
        accel_body=accel_body,
        quaternions_wxyz=quaternions_wxyz,
        accel_bias=accel_bias,
        gravity_world=gravity_world,
    )

    positions = np.zeros((n_samples, 3), dtype=np.float64)
    velocities = np.zeros((n_samples, 3), dtype=np.float64)
    is_gap = np.zeros(n_samples, dtype=bool)

    positions[0] = p0
    velocities[0] = v0

    if enable_zupt and zupt_mask is not None and zupt_mask[0]:
        velocities[0] = np.zeros(3, dtype=np.float64)

    for i in range(n_samples - 1):
        dt = dt_s[i]
        curr_is_gap = dt > gap_threshold_s
        is_gap[i + 1] = curr_is_gap

        if curr_is_gap:
            # Outage policy:
            # Do NOT integrate unknown acceleration across the gap.
            # Hold velocity constant, propagate position linearly using pre-gap velocity.
            velocities[i + 1] = velocities[i]
            positions[i + 1] = positions[i] + velocities[i] * dt
        else:
            # Normal continuous trapezoidal integration
            v_next = velocities[i] + 0.5 * (a_lin_world[i] + a_lin_world[i + 1]) * dt
            if enable_zupt and zupt_mask is not None and zupt_mask[i + 1]:
                v_next = np.zeros(3, dtype=np.float64)
            velocities[i + 1] = v_next
            positions[i + 1] = positions[i] + 0.5 * (velocities[i] + v_next) * dt

    out_df = pd.DataFrame({
        IMU_TIMESTAMP_COL: timestamps_ns,
        "p_x": positions[:, 0],
        "p_y": positions[:, 1],
        "p_z": positions[:, 2],
        "v_x": velocities[:, 0],
        "v_y": velocities[:, 1],
        "v_z": velocities[:, 2],
        "a_lin_x": a_lin_world[:, 0],
        "a_lin_y": a_lin_world[:, 1],
        "a_lin_z": a_lin_world[:, 2],
        "is_gap": is_gap,
    })
    return out_df
