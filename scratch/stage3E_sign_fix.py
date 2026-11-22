"""Quick fix check for Section 5 gravity sign."""
import sys; sys.path.insert(0,'src')
import numpy as np
from scipy.spatial.transform import Rotation
from data_loader import load_imu_data, load_groundtruth_data, IMU_TIMESTAMP_COL
from preprocessing import ACCEL_COLS
from degradation import sanitize_imu_measurements
from trajectory import STANDARD_GRAVITY

imu_df = load_imu_data('data/raw/euroc/machine_hall/MH_02_easy/mav0/imu0/data.csv')
df_san, _ = sanitize_imu_measurements(imu_df.copy())
gt_df = load_groundtruth_data('data/raw/euroc/machine_hall/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv')
gt_ts = gt_df['timestamp'].to_numpy(np.int64)

imu_ts = df_san[IMU_TIMESTAMP_COL].to_numpy(np.int64)
imu_a  = df_san[ACCEL_COLS].to_numpy(np.float64)

STAT_S, STAT_E = 5180, 5979
a_stat = imu_a[STAT_S:STAT_E+1]
a_stat_mean = a_stat.mean(0)
a_stat_std  = a_stat.std(0)

stat_ts_start = imu_ts[STAT_S]; stat_ts_end = imu_ts[STAT_E]
gt_mask = (gt_ts >= stat_ts_start) & (gt_ts <= stat_ts_end)
gt_bax = gt_df['b_a_RS_S_x'].to_numpy()[gt_mask].mean()
gt_bay = gt_df['b_a_RS_S_y'].to_numpy()[gt_mask].mean()
gt_baz = gt_df['b_a_RS_S_z'].to_numpy()[gt_mask].mean()
gt_ba_stat = np.array([gt_bax, gt_bay, gt_baz])
print(f"GT b_a at stationary interval: {gt_ba_stat}")

gt_qw = gt_df['q_RS_w'].to_numpy()[gt_mask].mean()
gt_qx = gt_df['q_RS_x'].to_numpy()[gt_mask].mean()
gt_qy = gt_df['q_RS_y'].to_numpy()[gt_mask].mean()
gt_qz = gt_df['q_RS_z'].to_numpy()[gt_mask].mean()
q_xyzw = np.array([gt_qx, gt_qy, gt_qz, gt_qw])
q_xyzw /= np.linalg.norm(q_xyzw)
R_WS = Rotation.from_quat(q_xyzw).as_matrix()

# EuRoC measurement model: f_S = R_WS^T * (a_W - g_W) + b_a
# g_W = [0, 0, -9.80665]  (Z-up, gravity negative Z)
# At rest (a_W=0): f_S = R_WS^T * (0 - g_W) + b_a = R_WS^T * [0,0,+g] + b_a
# So b_a = f_S - R_WS^T * [0, 0, g]

# Gravity contribution to specific force in body frame (at rest):
# -R_WS^T * g_W = -R_WS^T * [0,0,-g] = R_WS^T * [0,0,g]
g_body_component = R_WS.T @ np.array([0.0, 0.0, STANDARD_GRAVITY])  # R_WS^T * [0,0,g]
print(f"\ng_body_component (R_WS^T * [0,0,g]) = {g_body_component}")
print(f"a_stat_mean = {a_stat_mean}")
print(f"||a_stat_mean|| = {np.linalg.norm(a_stat_mean):.6f} vs g = {STANDARD_GRAVITY:.5f}")

ba_recovered_correct = a_stat_mean - g_body_component
print(f"\nb_a recovered (3-axis, known orientation): {ba_recovered_correct}")
print(f"GT b_a:                                    {gt_ba_stat}")
recovery_err = ba_recovered_correct - gt_ba_stat
print(f"Recovery error: {recovery_err}  ||={np.linalg.norm(recovery_err):.6f} m/s^2")
print(f"a_stat std: {a_stat_std}  ||={np.linalg.norm(a_stat_std):.4f} m/s^2")
print(f"Note: recovery error should ≈ noise std ({np.linalg.norm(a_stat_std):.4f} m/s^2)")

# Also verify: ||g_body_component|| should == g (rotation preserves norm)
print(f"\n||g_body_component|| = {np.linalg.norm(g_body_component):.6f} (should be {STANDARD_GRAVITY:.5f})")
