"""
Build a strictly compliant stepstack.zip for Samsung XR Hackathon Track 3 submission.
- Team name: stepstack
- Follows Slide 17 & Official FAQ:
  * Exclusively Python (NO c_src/, NO bin/imu_dr.exe)
  * estimated_trajectory.txt (Config A MH_01, 35,629 poses, 1:1)
  * metadata.json
  * src/ with main.py, README.md, and all Python modules
  * Top-level main.py and README.md included for maximum runner convenience
  * Zero __pycache__, zero nested duplicates
"""
import zipfile
import json
import os
from pathlib import Path

ROOT = Path("C:/Users/senth/Samsung_XR_Hackathon")
OUT_ZIP = ROOT / "stepstack.zip"

if OUT_ZIP.exists():
    OUT_ZIP.unlink()
    print(f"Removed previous zip: {OUT_ZIP}")

# Load verified benchmark metadata
with open(ROOT / "outputs/production_config_a/final_benchmark_results.json") as f:
    bench = json.load(f)

mh01 = next(r for r in bench if r["id"] == "MH_01")

metadata = {
    "team_name": "stepstack",
    "track": "Track 3 - Path Finder: Trajectory Estimation Under Sensor Degradation",
    "submission_language": "Python 3 (Standalone)",
    "pipeline_version": "Production Config A (V_MAX=10.0 m/s, OUTAGE_DECAY_RATE=1.5 s^-1)",
    "entrypoint": "src/main.py",
    "dataset": "EuRoC MAV MH_01_easy (Degraded)",
    "input_path": "data/degraded/MH_01_easy_degraded.csv",
    "output_path": "estimated_trajectory.txt",
    "input_frames": mh01["frames_in"],
    "output_tum_poses": mh01["frames_out"],
    "one_to_one_row_preservation": mh01["frames_in"] == mh01["frames_out"],
    "duplicate_timestamps_preserved": mh01["duplicates"],
    "terminal_failure_frames_preserved": True,
    "telemetry_duration_s": 184.095000064,
    "compute_latency_ms_per_s_untraced": round(mh01["latency_untraced_ms_s"], 4),
    "compute_latency_budget_ms_per_s": 20.0,
    "meets_compute_latency_budget": mh01["latency_untraced_ms_s"] <= 20.0,
    "python_heap_peak_mb": round(mh01["heap_mb"], 4),
    "python_heap_budget_mb": 8.0,
    "meets_python_heap_budget": mh01["heap_mb"] <= 8.0,
    "single_thread_enforced": mh01["single_thread"],
    "tum_format_status": mh01["tum_status"],
    "max_quat_norm_deviation": 8.84e-7,
    "all_positions_finite": mh01["all_finite"],
    "metrics": {
        "ate_rmse_m": round(mh01["ate_rmse"], 4),
        "rpe_1s_rmse_m": round(mh01["rpe_rmse"], 4),
        "checkpoint_25pct_m": round(mh01["cp_25"], 4),
        "checkpoint_50pct_m": round(mh01["cp_50"], 4),
        "checkpoint_75pct_m": round(mh01["cp_75"], 4),
        "blind_5s_trajectory_error_m": round(mh01["blind_5s_traj"], 4),
        "blind_5s_endpoint_error_m": round(mh01["blind_5s_end"], 4),
        "dropout_recovery_error_m": round(mh01["dropout_rec"], 4),
        "estimated_path_length_m": round(mh01["est_path"], 4),
        "gt_path_length_m": round(mh01["gt_path"], 4)
    },
    "bias_estimation": {
        "gyro_bias_rad_s": [-0.003029672, 0.022073285, 0.081076095],
        "confidence": "HIGH",
        "status": "CONFIRMED_QUASI_STATIONARY",
        "duration_s": 10.025
    },
    "execution_instructions": {
        "standard_run": "python src/main.py --input <imu_csv> --output estimated_trajectory.txt",
        "alternative_run": "python main.py --input <imu_csv> --output estimated_trajectory.txt"
    }
}

meta_tmp = ROOT / "scratch/stepstack_metadata.json"
with open(meta_tmp, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

traj_file = ROOT / "outputs/production_config_a/MH_01_config_a.txt"
readme_file = ROOT / "src/README.md"
main_file = ROOT / "src/main.py"

# Gather clean Python source files from src/
src_files = []
for p in (ROOT / "src").rglob("*.py"):
    if "__pycache__" not in str(p) and "src\\src" not in str(p):
        rel = p.relative_to(ROOT)
        src_files.append((p, str(rel).replace("\\", "/")))

# Also include src/README.md
src_files.append((readme_file, "src/README.md"))

print(f"Assembling {OUT_ZIP.name}...")
with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    # 1. Root deliverables (Slide 17)
    zf.write(traj_file, "estimated_trajectory.txt")
    print(f"  + estimated_trajectory.txt ({traj_file.stat().st_size:,} bytes)")
    
    zf.write(meta_tmp, "metadata.json")
    print("  + metadata.json")
    
    # Root convenience copies (in case evaluators run from unzipped root)
    zf.write(readme_file, "README.md")
    print("  + README.md (root)")
    
    zf.write(main_file, "main.py")
    print("  + main.py (root alias)")
    
    # 2. Complete src/ tree
    for abs_p, arc_n in sorted(src_files):
        zf.write(abs_p, arc_n)
        print(f"  + {arc_n}")

print(f"\nSuccessfully created {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1024:.1f} KB)")
