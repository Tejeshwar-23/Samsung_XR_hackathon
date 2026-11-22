# Samsung XR Hackathon 2026 - Track 3: Path Finder
## Trajectory Estimation Under Sensor Degradation

### 1. Overview & System Architecture

This package provides an autonomous, real-time 6-DoF inertial navigation dead-reckoning pipeline designed for high-stress IMU telemetry under sensor degradation (sensor dropouts, burst packet loss, dynamic/stationary blackouts, additive noise, and progressive bias drift).

The architecture implements a modular, feed-forward strapdown inertial navigation system (INS):
- **Sensor Preprocessing & Sanitization (`src/degradation.py`)**: Sanitizes incoming IMU telemetry by detecting physical sensor saturation, invalid measurements, and non-finite values. Corrupted channels are isolated with zero placeholders (never forward-filled).
- **Quasi-Stationary Gyro Bias Estimation (`src/bias_estimation.py`)**: Multi-criterion sliding-window ZARU detector that identifies stationary and quasi-stationary hover phases (e.g., $21.20\text{ s} - 31.60\text{ s}$ on `MH_01_easy`) and recovers 3-axis gyroscope bias without ground-truth assistance.
- **Attitude Integration (`src/orientation.py`)**: Analytical shortest-arc gravity alignment initializes roll and pitch; trapezoidal midpoint quaternion integration propagates orientation with strict unit normalization and orientation-holding across outages.
- **Strapdown Trajectory Integration (`src/trajectory.py`)**: Compensates gravity in the level world frame ($Z$-up opposing gravity), propagates translational velocity and position, and applies pre-gap velocity holding across timestamp outages.
- **TUM Trajectory Formatting (`src/trajectory_export.py`)**: Precision-safe trajectory exporter adhering to the official 8-column TUM format (`timestamp tx ty tz qx qy qz qw`) with exact integer arithmetic for nanosecond-to-second conversion.

---

### 2. Standalone Submission Entrypoint & CLI Contract

The official autonomous submission entrypoint is `src/main.py`.

```bash
# Basic Autonomous Execution
python src/main.py --input <path_to_imu_csv> --output <path_to_estimated_trajectory.txt>

# Full Execution with Provenance & Performance Telemetry Output
python src/main.py \
    --input data/raw/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv \
    --output outputs/estimated_trajectory.txt \
    --metadata-output outputs/metadata.json \
    --scenario S4-01 \
    --seed 42
```

#### Command-Line Arguments:
- `--input`, `-i` *(required)*: Path to input IMU CSV (EuRoC MAV format or TUM-VI format).
- `--output`, `-o` *(default: `outputs/estimated_trajectory.txt`)*: Output destination for the 8-column TUM trajectory file.
- `--scenario`, `-s` *(default: `S4-01`)*: Provisional degradation scenario ID (`S4-01` through `S4-08`).
- `--seed` *(default: `42`)*: Integer random seed for reproducible synthetic degradation generation.
- `--extrapolate-terminal-seconds` *(default: `0.0`)*: Optional duration (seconds) to append constant-velocity terminal prediction. Quarantined and disabled by default.
- `--metadata-output`, `-m` *(default: `None`)*: Path to save comprehensive execution telemetry JSON.
- `--log-level` *(default: `INFO`)*: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

---

### 3. File Formats & I/O Specifications

#### Input Telemetry (EuRoC MAV CSV Format)
The loader expects space- or comma-separated CSV with header:
```csv
#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
1403636579758555392,0.01254,-0.00241,0.08123,8.4234,1.2341,-4.9812
```

#### Output Trajectory (Official 8-Column TUM Format)
Exported pose files strictly adhere to the TUM format without index or column headers (optional `# timestamp ...` comment permitted):
```text
# timestamp tx ty tz qx qy qz qw
1403636579.758555392 0.000000 0.000000 0.000000 -0.039244 -0.820397 0.000000 0.570446
1403636579.763555584 -0.000007 0.000000 -0.000017 -0.039275 -0.820227 -0.000282 0.570689
...
```
- **Columns**: `timestamp` (Unix seconds with exact 9-decimal precision), `tx`, `ty`, `tz` (meters), `qx`, `qy`, `qz`, `qw` (normalized unit quaternion).
- **1:1 Frame Rule**: Output rows exactly match input IMU frame count (unless terminal extrapolation is explicitly enabled).

---

### 4. Sensor Degradation & Outage Handling Policies

All synthetic degradation testing in development is marked:
`PROVISIONAL DEVELOPMENT TESTING - NOT OFFICIAL ORGANIZER DEGRADATION`

#### Supported Provisional Scenarios:
| Scenario ID | Name | Operational Description |
|:---|:---|:---|
| **S4-01** | Baseline | Unmodified input telemetry stream. |
| **S4-02** | Progressive Noise | Discrete sampled Gaussian noise scaling linearly $1\times \to 5\times$ over flight ($\sigma_{\omega,0} = 0.005\text{ rad/s}$, $\sigma_{a,0} = 0.05\text{ m/s}^2$). |
| **S4-03** | Progressive Drift | Linear bias drift: Gyro $[+0.05, -0.03, +0.04]^\circ/\text{s/min}$, Accel $[+0.05, +0.10, -0.05]\text{ m/s}^2/\text{min}$. |
| **S4-04** | Random Drops | Deterministic pseudo-random 3.0% frame removal (without replacement, preserving edge frames). |
| **S4-05** | Burst Drops | Five 75 ms outages (15 consecutive frames dropped at 200 Hz) at $t=15, 45, 75, 105, 145\text{ s}$. |
| **S4-06** | Dynamic Blackout | Complete 2.0-second sensor loss during active flight ($t=50.0\text{ s} - 52.0\text{ s}$). |
| **S4-07** | Stationary Blackout | Complete 2.0-second sensor loss during quasi-stationary hover ($t=24.0\text{ s} - 26.0\text{ s}$). |
| **S4-08** | Terminal Failure | Complete 5.0-second sensor loss for the final 5.0 seconds of recording. |

#### Measurement Sanitization & Outage Policy:
1. **Invalid Gyro Readings**: If gyro readings contain NaN/Inf or exceed physical saturation ($>35.0\text{ rad/s}$), `valid_gyro` is set to `False`. The estimator holds the previous orientation ($q_k = q_{k-1}$) rather than integrating corrupted rates.
2. **Invalid Accelerometer Readings**: If accelerometer readings contain NaN/Inf or exceed saturation ($>180.0\text{ m/s}^2$), `valid_accel` is set to `False`. The estimator forces world linear acceleration $\mathbf{a}_{\text{lin}, R} = \mathbf{0}$, holding velocity ($\mathbf{v}_k = \mathbf{v}_{k-1}$) and propagating position linearly ($\mathbf{p}_k = \mathbf{p}_{k-1} + \mathbf{v}_{k-1} \Delta t_k$).
3. **No Forward-Fill**: Corrupted sensor measurements are never forward-filled. Raw measurements are preserved in `w_raw_*` and `a_raw_*`.
4. **Timestamp Gaps**: When $\Delta t > 1.5 \times \text{median}(\Delta t)$, acceleration is not integrated across the outage. Velocity is held constant and position is propagated using pre-gap velocity.

---

### 5. Official Competition Resource Budgets & Measurement Methodology

The competition rules specify strict operational bounds:
- **RAM Budget**: $\le 8.0\text{ MB}$.
- **Compute Latency**: $\le 20.0\text{ ms}$ per second of sensor telemetry.
- **Concurrency**: Strict single-threaded execution.

#### Dual Memory Measurement Protocol:
The entrypoint implements dual memory profiling:
1. **Python Traced Heap (`tracemalloc`)**: Measures the peak allocation of internal Python data structures and NumPy arrays during pipeline execution.
2. **Total OS Process Working Set (`PeakWorkingSetSize` / RSS)**: Queries the operating system directly (via `GetProcessMemoryInfo` on Windows, or `psutil` peak working set).

> **Transparent Evaluation Note**: While Python heap usage for numerical processing is compact, standard CPython runtime environments, dynamically linked DLLs (`mkl`, `openblas`, C runtimes), and compiler heaps impose an unavoidable base memory footprint ($\sim 25 - 45\text{ MB}$ at startup before code execution). In `metadata.json`, both values are reported transparently:
> - `python_heap_peak_mb` vs. budget $\le 8.0\text{ MB}$
> - `process_peak_rss_mb` vs. budget $\le 8.0\text{ MB}$

#### Single-Thread Enforcement:
All threading environment variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`) are explicitly pinned to `"1"` at the very top of `src/main.py` before NumPy, SciPy, or pandas are loaded. In addition, the runtime dynamically verifies actual active BLAS/OpenMP thread pools using `threadpoolctl.threadpool_info()`, recording full pool introspection inside `metadata.json`.

---

### 6. Ground-Truth Isolation & Autonomous Guarantee

To ensure total competition integrity:
- Ground-truth files are strictly isolated to evaluation scripts (`src/evaluate_trajectory.py`).
- The autonomous pipeline (`src/main.py`), loader, orientation integrator, bias estimator, and trajectory integrator do not accept, load, or reference ground-truth data in any code path.
- Initial orientation is derived strictly from stationary accelerometer specific force (gravity alignment). Initial yaw is defaulted to $0.0\text{ rad}$ as unobservable from IMU alone.

---

### 7. Known Physical Limitations of Pure IMU Dead-Reckoning

Pure strapdown inertial navigation without external aiding (exteroceptive vision, LiDAR, or GNSS) is subject to fundamental physical error accumulation:
1. **Gyroscope Bias Residuals & Error Growth Dynamics**:
   - Under the specific idealized conditions of a constant residual gyroscope bias $\delta \mathbf{b}_\omega$ during near-level, non-accelerating flight, the accumulated tilt angle error $\delta \boldsymbol{\theta} \sim \delta \mathbf{b}_\omega t$ misprojects the nominal gravity vector $\mathbf{g}$, producing an apparent horizontal acceleration residual $\sim g \delta \boldsymbol{\theta}$. Double integration of this term produces a characteristic cubic position error growth component $\frac{1}{6} g \delta \mathbf{b}_\omega t^3$.
   - **Crucial Qualification**: This cubic progression is a canonical theoretical model for constant tilt drift under gravity alignment, **not** an empirical universal lower bound for all arbitrary trajectories. During complex, multi-axis high-dynamic maneuvers, time-varying body rotations and cross-axis kinematic coupling modulate the projection of gravity, leading to complex multi-rate divergence.
2. **Accelerometer Bias Unobservability**: In a constant attitude or hover, only the 1D gravity-parallel accelerometer bias component is observable. The horizontal bias components are indistinguishable from sensor tilt, leading to quadratic position drift $\frac{1}{2} \mathbf{b}_a t^2$.
3. **Long-Flight Drift**: Over full multi-minute trajectories (e.g. 184 seconds of `MH_01_easy`), unassisted dead-reckoning will exhibit rapid kinematic drift. The pipeline prioritizes physical correctness, numerical stability, outage recovery, and format compliance.
