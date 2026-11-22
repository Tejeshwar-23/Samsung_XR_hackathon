# Supplementary Detailed Production Metrics (Configuration A)

Primary Source of Truth: `outputs/production_config_a/final_benchmark_results.json`  
Configuration: $V_{\max} = 10.0\text{ m/s}$, $\text{decay\_rate} = 1.5\text{ s}^{-1}$, Single-Threaded, Memoryview Integration.

---

## 1. Complete Seven-Sequence Metric Breakdown

| Metric | MH_01_easy | MH_02_easy | MH_03_medium | MH_04_difficult | MH_05_difficult | TUM-VI Canonical | TUM-VI Corrected |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Dataset Source** | EuRoC MAV | EuRoC MAV | EuRoC MAV | EuRoC MAV | EuRoC MAV | TUM-VI Room 1 | TUM-VI Room 1 |
| **Input Frames** | 35,629 | 29,480 | 26,189 | 19,753 | 22,137 | 27,330 | 27,330 |
| **Output TUM Poses** | 35,629 | 29,480 | 26,189 | 19,753 | 22,137 | 27,330 | 27,330 |
| **Preserved Duplicates** | 49 | 30 | 29 | 19 | 27 | 35 | 35 |
| **All Poses Finite** | True | True | True | True | True | True | True |
| **Max Quat. Norm Err.** | $8.84 \times 10^{-7}$ | $8.66 \times 10^{-7}$ | $8.82 \times 10^{-7}$ | $8.78 \times 10^{-7}$ | $8.97 \times 10^{-7}$ | $7.61 \times 10^{-7}$ | $7.61 \times 10^{-7}$ |
| **ATE RMSE (m)** | **374.74** | **332.63** | **309.47** | **156.77** | **202.62** | **168.51** | **168.51** |
| **1s RPE RMSE (m)** | **9.08** | **10.02** | **8.93** | **7.15** | **8.22** | **12.97** | **12.97** |
| **Checkpoint 25% (m)** | 336.71 | 311.03 | 330.75 | 141.85 | 182.33 | 169.65 | 169.65 |
| **Checkpoint 50% (m)** | 161.48 | 57.20 | 33.17 | 54.24 | 105.57 | 112.98 | 112.98 |
| **Checkpoint 75% (m)** | 346.56 | 350.00 | 284.02 | 142.22 | 187.65 | 201.69 | 201.69 |
| **Blind 5s Traj. Err. (m)** | 617.97 | 473.28 | 549.78 | 305.39 | 379.04 | 170.72 | 170.72 |
| **Blind 5s End. Err. (m)** | 620.93 | 474.50 | 552.74 | 307.94 | 382.12 | 171.07 | 171.07 |
| **Dropout Rec. Err. (m)** | 403.45 | 302.88 | 296.89 | 139.00 | 187.45 | 184.50 | 184.50 |
| **Estimated Path (m)** | 1,510.56 | 1,426.61 | 1,011.23 | 538.01 | 735.13 | 947.45 | 947.45 |
| **Ground-Truth Path (m)** | 80.62 | 73.47 | 130.91 | 91.73 | 97.59 | 146.79 | 146.79 |
| **Python Heap Peak (MB)** | **5.38** | **4.44** | **3.96** | **3.01** | **3.37** | **4.13** | **4.13** |
| **Untraced Latency (ms/s)** | **2.48** | **2.53** | **2.56** | **2.64** | **2.68** | **2.79** | **2.53** |
| **Traced Latency (ms/s)** | 179.88 | 73.00 | 98.50 | 74.62 | 76.32 | 75.51 | 72.58 |
| **Gyro Bias Status** | CONFIRMED | NO_INTERVAL | CONFIRMED | CONFIRMED | CONFIRMED | NO_INTERVAL | NO_INTERVAL |
| **Gyro Bias Error (deg/s)** | **0.107**\*\* | N/A | N/A | N/A | N/A | N/A | N/A |
| **Single-Threaded** | True | True | True | True | True | True | True |
| **TUM Format Status** | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

\*\* *MH_01 bias error is computed against the verified raw IMU golden baseline `[-0.003206, 0.020793, 0.079723]` rad/s ($0.001871\text{ rad/s} = 0.1072^\circ/\text{s}$). For sequences without an authoritative reference in the dataset specification, values are reported as N/A.*

---

## 2. In-Depth Operational & Algorithmic Notes

1. **SE(3) Alignment Protocol:**
   All ATE (Absolute Trajectory Error) RMSE metrics are evaluated using the standard `evo` SE(3) Umeyama rigid-body transformation (rotation + translation, strictly preserving scale $= 1.0$).
2. **Relative Pose Error (RPE) Standard:**
   RPE evaluates local translation drift over a fixed rolling temporal delta of $1.0\text{ s}$ ($200$ consecutive frames at $200\text{ Hz}$).
3. **Canonical vs. Corrected TUM-VI Verification:**
   The canonical and corrected versions of TUM-VI Room 1 were verified to produce **bit-exact identical outputs** (`filecmp.cmp = True`). The untraced runtime variation ($2.79\text{ ms/s}$ vs. $2.53\text{ ms/s}$) reflects minor OS CPU scheduling jitter during standalone timing passes.
4. **Traced vs. Untraced Timing:**
   - **Untraced Latency ($2.48 - 2.79\text{ ms/s}$):** Represents official production latency without `tracemalloc` instrumentation hooks. Comfortably meets the $\le 20.0\text{ ms/s}$ budget.
   - **Traced Latency ($72.58 - 179.88\text{ ms/s}$):** Represents profiling overhead under Python's line-by-line memory allocation hooks; strictly used for auditing internal data-structure heap.
