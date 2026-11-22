import numpy as np
import pandas as pd
import copy
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface

for seq in ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]:
    gt_file = f"data/raw/euroc/machine_hall/{seq}/mav0/state_groundtruth_estimate0/data.csv"
    df_gt = pd.read_csv(gt_file)
    ts_gt = df_gt.iloc[:, 0].to_numpy() * 1e-9
    pos_gt = df_gt.iloc[:, 1:4].to_numpy()
    qw = df_gt.iloc[:, 4].to_numpy()
    qx = df_gt.iloc[:, 5].to_numpy()
    qy = df_gt.iloc[:, 6].to_numpy()
    qz = df_gt.iloc[:, 7].to_numpy()
    gt_traj = trajectory.PoseTrajectory3D(pos_gt, np.column_stack([qx, qy, qz, qw]), ts_gt)

    base_traj = file_interface.read_tum_trajectory_file(f"scratch/baseline_py_statq0_{seq}.txt")
    gt_sync, base_sync = sync.associate_trajectories(gt_traj, base_traj, max_diff=0.02)
    base_aligned = copy.deepcopy(base_sync)
    base_aligned.align(gt_sync, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt_sync, base_aligned))
    rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=200, delta_unit=metrics.Unit.frames, all_pairs=False)
    rpe.process_data((gt_sync, base_aligned))
    ate_val = ape.get_all_statistics()["rmse"]
    rpe_val = rpe.get_all_statistics()["rmse"]
    print(f"{seq} RAW BASELINE: ATE={ate_val:.2f}m, RPE={rpe_val:.2f}m, EstPath={base_aligned.path_length:.1f}m, GTPath={gt_sync.path_length:.1f}m")
