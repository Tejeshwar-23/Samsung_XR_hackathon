# Samsung XR Hackathon 2026 — Track 3: Path Finder
## Autonomous 6-DoF Inertial Dead-Reckoning Under Sensor Degradation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-89%20%2F%2089%20passed-success.svg)]()
[![Heap Peak](https://img.shields.io/badge/heap-3.01--5.38%20MB%20(%E2%89%A4%208%20MB)-brightgreen.svg)]()
[![Untraced Latency](https://img.shields.io/badge/latency-2.48--2.79%20ms%2Fs%20(%E2%89%A4%2020%20ms%2Fs)-brightgreen.svg)]()
[![TUM Format](https://img.shields.io/badge/TUM%20format-100%25%20compliant-blue.svg)]()
[![Single-Threaded](https://img.shields.io/badge/single--threaded-enforced-orange.svg)]()

> **Team Name:** `stepstack`  
> **Challenge:** Samsung XR Hackathon 2026 — Track 03: Path Finder  
> **Official Submission Bundle:** `stepstack.zip`

---

## 📌 Executive Summary

Extended Reality (XR) headsets demand continuous, ultra-low-latency 6-DoF spatial pose estimates to preserve perceptual coherence. In practical wearable deployments, visual and LiDAR sensors suffer from occlusions, low light, and feature tracking losses, forcing the system to fall back on the **Inertial Measurement Unit (IMU)**. 

This repository contains our production-grade, autonomous strapdown inertial navigation system (INS) engineered specifically for **high-stress, corrupted IMU telemetry streams**. Operating **purely on IMU inputs** (without external camera, LiDAR, or GNSS aiding), the system incorporates:
1. **Saturation & Blackout Sanitization:** Channel-level validity isolation without erroneous forward-filling.
2. **Quasi-Stationary Gyroscope Bias Estimation (ZARU):** Multi-criterion sliding-window detection recovering 3-axis bias without ground truth (0.107°/s error on MH_01).
3. **World-Frame Gravity Alignment:** Analytical shortest-arc leveling initializing orientation in the level navigation frame.
4. **Adaptive Interval-Specific Outage Dynamics (Production Configuration A):** Differentiated policies for zero-duration duplicate timestamps, short packet dropouts, multi-second dropouts, and terminal 5-second complete sensor blackouts with exponential velocity decay and a 10.0 m/s physical velocity cap.
5. **Exact 1:1 TUM Trajectory Export:** Precision-safe 8-column formatting preserving every input frame, timestamp, and duplicate pair.

---

## 🏆 Production Results & Findings (Slide 11)

All metrics correspond to **Production Configuration A** (`V_MAX = 10.0 m/s`, `OUTAGE_DECAY_RATE = 1.5 s⁻¹`), evaluated against ground truth using `evo` with SE(3) Umeyama rigid-body alignment (`scale = 1.0`) and a 1.0-second rolling RPE window:

| Dataset / Sequence | Input Frames | Output Poses | ATE RMSE (m) | 1s RPE RMSE (m) | Peak Heap (MB) | Untraced Latency (ms/s) | TUM Status |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| **EuRoC MH_01_easy** | 35,629 | 35,629 | **374.74** | **9.08** | **5.38** | **2.48** | **PASS** |
| **EuRoC MH_02_easy** | 29,480 | 29,480 | **332.63** | **10.02** | **4.44** | **2.53** | **PASS** |
| **EuRoC MH_03_medium** | 26,189 | 26,189 | **309.47** | **8.93** | **3.96** | **2.56** | **PASS** |
| **EuRoC MH_04_difficult** | 19,753 | 19,753 | **156.77** | **7.15** | **3.01** | **2.64** | **PASS** |
| **EuRoC MH_05_difficult** | 22,137 | 22,137 | **202.62** | **8.22** | **3.37** | **2.68** | **PASS** |
| **TUM-VI Room 1 (Canonical)**\* | 27,330 | 27,330 | **168.51** | **12.97** | **4.13** | **2.79** | **PASS** |
| **TUM-VI Room 1 (Corrected)**\* | 27,330 | 27,330 | **168.51** | **12.97** | **4.13** | **2.53** | **PASS** |

\* *Canonical and corrected variants of TUM-VI were evaluated independently; both produced bit-identical trajectories (`filecmp.cmp = True`), preserving all 27,330 rows and 35 duplicate timestamp pairs.*

### Budget Compliance Highlights

```
+-------------------------+-------------------------+-------------------------+-------------------------+
|         89 / 89         |      3.01 - 5.38 MB     |     2.48 - 2.79 ms/s    |       100% (7 / 7)      |
|  Regression Suite Pass  |     Peak Python Heap    |     Untraced Latency    |  TUM Format Validation  |
|    (10.03s runtime)     |   (Budget: <= 8.0 MB)   |   (Budget: <= 20 ms/s)  |   (Exact 1:1 Mapping)   |
+-------------------------+-------------------------+-------------------------+-------------------------+
```

---

## 🏗️ System Architecture

```
                                  [ Degraded IMU Telemetry (CSV) ]
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │    Lean Nanosecond Stream Parser    │  src/data_loader.py
                             │  - Resolves column header schemas   │
                             │  - Preserves exact int64 timestamps │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │   Sensor Preprocessing & Sanitizer  │  src/degradation.py
                             │  - Non-finite (NaN/Inf) gating      │
                             │  - Physical saturation (>35 rad/s)  │
                             │  - Terminal blackout isolation      │
                             │  - NEVER forward-fills stale values │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │  Quasi-Stationary Bias Estimation   │  src/bias_estimation.py
                             │  - Multi-criterion sliding ZARU    │
                             │  - Low gyro norm & accel variance   │
                             │  - Zero ground-truth reference used │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │     World-Frame Gravity Alignment   │  src/orientation.py
                             │  - Shortest-arc roll/pitch leveling │
                             │  - World-frame specific force proj. │
                             │  - Midpoint quaternion integration  │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │    Strapdown Trajectory Integrator  │  src/trajectory.py
                             │  - Trapezoidal position propagation │  src/main.py
                             │  - Level world frame (Z-up vs g)    │
                             │  - Differentiated Outage Policies   │
                             │  - Physical velocity cap (10.0 m/s) │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │       TUM Trajectory Formatter      │  src/trajectory_export.py
                             │  - Exact integer ns -> sec decimal  │
                             │  - 8-column format: t tx ty tz q..  │
                             │  - Strict unit quaternion norm      │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                             [ Official TUM Trajectory (.txt) ]
```

---

## 🔬 Core Methodological Innovations

### 1. Differentiated Outage Policy (Interval Dynamics)
A primary challenge in degraded dead-reckoning is avoiding runaway velocity divergence during sensor loss while not damping real vehicle motion:

| Interval Condition | Operational Detection | Downstream Estimator Behavior |
|:---|:---|:---|
| **Zero-Duration Duplicate** | `dt == 0 s` | Zero state advancement (`p_k = p_{k-1}`, `v_k = v_{k-1}`); zero decay. Retains exact row mapping. |
| **Nominal Flight** | `0 < dt <= 0.025 s`, valid sensor | Trapezoidal acceleration integration, velocity capped at `V_MAX = 10.0 m/s`. |
| **Short Packet Dropout** | `dt > 0.025 s` or invalid sensor | Acceleration zeroed; exponential velocity decay `v_k = v_{k-1} * exp(-1.5 * dt)`; position propagated linearly. |
| **Multi-Second Blackout** | `dt >= 2.0 s` (e.g. S4-06/S4-07) | Sustained exponential velocity damping preventing explosive quadratic trajectory expansion. |
| **Terminal 5-Second Failure** | Contiguous 6-axis zeros extending to end | Consistent velocity decay policy across the blind prediction interval; exact frame count preserved. |

### 2. Velocity Cap Headroom Analysis
Ground-truth drone speeds across all EuRoC sequences peak at 2.87 m/s. Our `V_MAX = 10.0 m/s` threshold provides **>3.4× headroom**, ensuring genuine high-acceleration maneuvers are never clipped while arresting spurious integration noise during dropouts.

### 3. Memory & Latency Optimization
- **Memoryview Indexing:** Both orientation and position integration loops utilize direct Python `memoryview` indexing over pre-allocated NumPy buffers, eliminating temporary slice allocations.
- **Single-Thread Pinning:** Environment variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`) are explicitly pinned to `"1"` before loading numerical libraries.
- **Dual Timing Protocol:** Traced latency (72–180 ms/s) reflects `tracemalloc` profiling overhead; official untraced runtime is measured cleanly at **2.48–2.79 ms/s**.

---

## 📁 Repository Structure

```text
Samsung_XR_Hackathon/
├── stepstack.zip                           # Official competition submission bundle (Slide 17 & FAQ compliant)
├── requirements.txt                        # Minimal Python dependencies
├── degrade_imu.py                          # Official degradation harness script
├── src/                                    # Autonomous dead-reckoning pipeline modules
│   ├── main.py                             # CLI entrypoint and orchestrator (Config A)
│   ├── README.md                           # Package-level documentation
│   ├── data_loader.py                      # Robust CSV loader with header schema resolution
│   ├── degradation.py                      # Sensor saturation, blackout detection, and sanitization
│   ├── bias_estimation.py                  # Sliding-window quasi-stationary ZARU bias estimator
│   ├── orientation.py                      # Gravity-alignment quaternion leveling and RK4 integration
│   ├── trajectory.py                       # Gravity-compensated position integration
│   ├── trajectory_export.py                # 8-column TUM trajectory exporter & validator
│   ├── preprocessing.py                    # Channel definitions and validation helpers
│   ├── synchronization.py                  # Interpolation and temporal alignment
│   ├── evaluate_trajectory.py              # Offline EVO-based ground truth evaluation
│   ├── evaluate_bias_corrected_orientation.py
│   ├── analyze_imu_outliers.py
│   ├── inspect_dataset.py
│   └── visualize_imu.py
├── c_src/                                  # High-performance native C99 implementation
│   ├── src/ (main.c, dr_math.c, dr_io.c, dr_detector.c)
│   └── include/ (dr_types.h, dr_math.h, dr_io.h, dr_detector.h)
├── bin/
│   └── imu_dr.exe                          # Compiled native executable
├── tests/                                  # 89-test regression suite
│   ├── test_bias_estimation.py             # ZARU detector validation
│   ├── test_data_loader.py                 # Header sanitization & duplicate timestamp tests
│   ├── test_orientation.py                 # Quaternion integration & leveling checks
│   ├── test_preprocessing.py               # Outlier & NaN detection checks
│   ├── test_sensor_robustness.py           # Blackout isolation & frame recovery
│   ├── test_synchronization.py             # Temporal sync tests
│   └── test_trajectory.py                  # Trapezoidal integration & budget checks
├── outputs/                                # Benchmark outputs and evaluation data
│   ├── production_config_a/                # Final Production Configuration A deliverables
│   │   ├── final_benchmark_results.json    # Primary source of truth benchmark JSON
│   │   ├── slide11_results_table.md        # Formatted presentation Slide 11
│   │   ├── slide11_detailed_metrics.md     # 23-metric breakdown across all 7 runs
│   │   ├── slide11_verified_values.json    # Machine-readable verified values
│   │   ├── MH_01_config_a.txt              # Trajectory: MH_01_easy (35,629 poses)
│   │   ├── MH_02_config_a.txt              # Trajectory: MH_02_easy (29,480 poses)
│   │   ├── MH_03_config_a.txt              # Trajectory: MH_03_medium (26,189 poses)
│   │   ├── MH_04_config_a.txt              # Trajectory: MH_04_difficult (19,753 poses)
│   │   ├── MH_05_config_a.txt              # Trajectory: MH_05_difficult (22,137 poses)
│   │   ├── TUM_VI_room1_canonical_config_a.txt
│   │   └── TUM_VI_room1_corrected_config_a.txt
│   └── plots/                              # Visualization figures
└── scratch/                                # Verification and audit scripts
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Tejeshwar-23/Samsung_XR_hackathon.git
cd Samsung_XR_hackathon

# Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.\.venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Autonomous Estimator

```bash
# Basic run on any degraded IMU CSV
python src/main.py --input data/degraded/MH_01_easy_degraded.csv --output estimated_trajectory.txt

# Run with full performance telemetry JSON export
python src/main.py \
    --input outputs/MH_01_easy_degraded.csv \
    --output outputs/estimated_trajectory.txt \
    --metadata-output outputs/metadata.json
```

#### CLI Parameters:
- `--input`, `-i` *(required)*: Path to degraded IMU CSV file.
- `--output`, `-o` *(default: `outputs/estimated_trajectory.txt`)*: Output path for TUM trajectory.
- `--metadata-output`, `-m` *(optional)*: Path to save performance telemetry and diagnostics.
- `--log-level` *(default: `INFO`)*: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

### 3. Run the Regression Test Suite

```bash
# Run all 89 unit and integration tests
pytest tests/ -v
```

---

## 📦 Submission Bundle Details

Per **Slide 17** and the **Official Track 03 FAQ**, the official submission archive is [`stepstack.zip`](./stepstack.zip) located in the repository root. It contains:

```text
stepstack.zip
├── estimated_trajectory.txt     # 35,629 poses (TUM format, 1:1 frame match)
├── metadata.json                # Runtime, memory, and telemetry specifications
├── README.md                    # Top-level setup and run instructions
├── main.py                      # Top-level execution alias
└── src/                         # Complete standalone Python source
    ├── main.py                  # Entrypoint
    ├── README.md
    └── [supporting modules]
```

---

## 📖 Theoretical Limitations of Pure IMU Dead-Reckoning

Pure strapdown inertial dead-reckoning without external updates (vision, LiDAR, or GNSS) is subject to well-known physical error progression:
1. **Gyroscope Bias Residuals ($\delta b_\omega$):** In open-loop navigation, a constant residual bias induces an orientation error $\delta \theta \approx \delta b_\omega t$. Misprojecting the gravity vector $g$ results in apparent horizontal acceleration residuals, yielding **cubic position error growth** $\frac{1}{6} g \delta b_\omega t^3$.
2. **Accelerometer Bias ($b_a$):** Unobservable during steady-state motion, double integration results in **quadratic position drift** $\frac{1}{2} b_a t^2$.

Over extended flight durations (e.g. 184 seconds on `MH_01_easy`), unassisted dead-reckoning inevitably drifts. This pipeline prioritizes **physical correctness, numerical stability, exact row preservation, and budget compliance**.

---

## 📚 References

1. Burri, M., et al. (2016). *The EuRoC micro aerial vehicle datasets*. The International Journal of Robotics Research, 35(10), 1157–1163.
2. Schubert, D., et al. (2018). *The TUM VI Benchmark for Evaluating Visual-Inertial Odometry*. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).
3. Sturm, J., et al. (2012). *A Benchmark for the Evaluation of RGB-D SLAM Systems*. IEEE/RSJ IROS.
4. Titterton, D. H., & Weston, J. L. (2004). *Strapdown Inertial Navigation Technology* (2nd ed.). IET.
5. Farrell, J. A. (2008). *Aided Navigation: GPS with High Rate Sensors*. McGraw-Hill.
6. Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems* (2nd ed.). Artech House.
7. Foxlin, E. (2005). *Pedestrian Tracking with Shoe-Mounted Inertial Sensors*. IEEE Computer Graphics and Applications, 25(6), 38–46.

---

## 📄 License & Team

Developed for **Samsung XR Hackathon 2026 — Track 03: Path Finder**.  
**Team stepstack:** Tejeshwar Senthilkumar & Contributors.
