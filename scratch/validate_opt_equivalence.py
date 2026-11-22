"""
Validate bit-for-bit trajectory equivalence between original baseline and optimized output.
Compare rows, columns, exact numeric differences, and evaluate ATE/RPE.
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np

BASE_PATH = "outputs/MH_02_easy_est.txt"
OPT_PATH  = "outputs/MH_02_easy_opt.txt"
GT_PATH   = "data/raw/euroc/machine_hall/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"

def load_tum(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)

base_arr = load_tum(BASE_PATH)
opt_arr  = load_tum(OPT_PATH)

print(f"Base shape: {base_arr.shape}")
print(f"Opt  shape: {opt_arr.shape}")

assert base_arr.shape == opt_arr.shape, "Shape mismatch between baseline and optimized trajectories!"

diff = np.abs(base_arr - opt_arr)
max_ts_diff   = np.max(diff[:, 0])
max_pos_diff  = np.max(diff[:, 1:4])
max_quat_diff = np.max(diff[:, 4:8])

print(f"Max timestamp difference:  {max_ts_diff:.2e} s")
print(f"Max position difference:   {max_pos_diff:.2e} m")
print(f"Max quaternion difference: {max_quat_diff:.2e}")

# Exact text equality check
exact_match = True
with open(BASE_PATH, "r", encoding="utf-8") as f1, open(OPT_PATH, "r", encoding="utf-8") as f2:
    for l1, l2 in zip(f1, f2):
        if l1 != l2:
            exact_match = False
            break

print(f"Exact bit-for-bit text match: {exact_match}")

# Evaluate SE(3) ATE on both
from data_loader import load_groundtruth_data

gt_df = load_groundtruth_data(GT_PATH)
gt_ts = gt_df["timestamp"].to_numpy(np.int64)
gt_p  = gt_df[["p_RS_R_x", "p_RS_R_y", "p_RS_R_z"]].to_numpy(np.float64)

# Load IMU timestamps
ts_sec = opt_arr[:, 0]
ts_ns = (ts_sec * 1e9).round().astype(np.int64)

MAX_INTERP_GAP_NS = int(0.1 * 1e9)
valid_mask = np.zeros(len(ts_ns), dtype=bool)
gt_pos_interp = np.full((len(ts_ns), 3), np.nan)

for i in range(len(ts_ns)):
    t = ts_ns[i]
    idx_lo = int(np.searchsorted(gt_ts, t, side='right')) - 1
    if idx_lo < 0 or idx_lo >= len(gt_ts) - 1:
        continue
    idx_hi = idx_lo + 1
    if (gt_ts[idx_hi] - gt_ts[idx_lo]) > MAX_INTERP_GAP_NS:
        continue
    alpha = float(t - gt_ts[idx_lo]) / float(gt_ts[idx_hi] - gt_ts[idx_lo])
    gt_pos_interp[i] = (1 - alpha) * gt_p[idx_lo] + alpha * gt_p[idx_hi]
    valid_mask[i] = True

def umeyama_se3(p_est, p_gt):
    mu_e = p_est.mean(0); mu_g = p_gt.mean(0)
    E = p_est - mu_e; G = p_gt - mu_g
    H = (G.T @ E) / len(p_est)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R_opt = U @ D @ Vt
    t_opt = mu_g - R_opt @ mu_e
    p_aligned = (R_opt @ p_est.T).T + t_opt
    return R_opt, t_opt, p_aligned

p_gt_v   = gt_pos_interp[valid_mask]
p_base_v = base_arr[valid_mask, 1:4]
p_opt_v  = opt_arr[valid_mask, 1:4]

_, _, p_base_al = umeyama_se3(p_base_v, p_gt_v)
_, _, p_opt_al  = umeyama_se3(p_opt_v, p_gt_v)

ate_base = float(np.sqrt(np.mean(np.sum((p_base_al - p_gt_v)**2, axis=1))))
ate_opt  = float(np.sqrt(np.mean(np.sum((p_opt_al  - p_gt_v)**2, axis=1))))

print(f"Baseline  SE(3) ATE RMSE: {ate_base:.6f} m")
print(f"Optimized SE(3) ATE RMSE: {ate_opt:.6f} m")
print(f"ATE Delta: {abs(ate_base - ate_opt):.2e} m")
