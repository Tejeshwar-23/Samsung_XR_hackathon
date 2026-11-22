# Slide 11: Results and Findings

## Production Configuration A: $V_{\max} = 10.0\text{ m/s}$, Outage Decay $= 1.5\text{ s}^{-1}$

---

### Main Production Evaluation

| Dataset / Sequence | Frames | ATE RMSE (m) | 1s RPE RMSE (m) | Heap Peak (MB) | Untraced Latency (ms/s) | Validation |
|:---|---:|---:|---:|---:|---:|:---:|
| **MH_01_easy** | 35,629 | 374.74 | 9.08 | 5.38 | 2.48 | PASS |
| **MH_02_easy** | 29,480 | 332.63 | 10.02 | 4.44 | 2.53 | PASS |
| **MH_03_medium** | 26,189 | 309.47 | 8.93 | 3.96 | 2.56 | PASS |
| **MH_04_difficult** | 19,753 | 156.77 | 7.15 | 3.01 | 2.64 | PASS |
| **MH_05_difficult** | 22,137 | 202.62 | 8.22 | 3.37 | 2.68 | PASS |
| **TUM-VI Room 1**\* | 27,330 | 168.51 | 12.97 | 4.13 | 2.79 | PASS |

\* *Canonical and corrected variants were independently evaluated; both produced bit-identical trajectories. Each preserved all 27,330 input rows and 35 duplicate timestamp pairs.*

---

### Key Research Findings

- **F1 — Output Compliance & Integrity**
  - Seven production runs completed with $100\%$ pass rate on automated TUM validation.
  - Strict 1:1 input/output frame correspondence maintained without dropping rows or inserting synthetic timestamps.
  - All duplicate timestamp pairs (149 in EuRoC, 35 in TUM-VI) and 5-second terminal blackout frames preserved.
- **F2 — Resource Budget Compliance**
  - Peak algorithm Python heap usage spanned **3.01 – 5.38 MB**, strictly adhering to the $\le 8.0\text{ MB}$ limit ($32.8\% - 62.4\%$ headroom).
  - Untraced compute latency was **2.48 – 2.79 ms/s**, outperforming the $\le 20.0\text{ ms/s}$ competition budget by **$7.2\times - 8.1\times$**.
- **F3 — Software Reliability & Pipeline Integrity**
  - **89 / 89** unit and regression tests passed across all submodules (sanitization, bias, attitude, trajectory, and formatting).
  - Strict single-thread execution enforced across BLAS, LAPACK, and runtime libraries (`OMP/MKL/OPENBLAS = 1`).
  - Zero ground-truth leakage: runtime estimation is strictly feed-forward and unassisted.
- **F4 — Trajectory Accuracy & Physical Dead-Reckoning Limits**
  - Minimum ATE RMSE achieved on **MH_04_difficult** (**156.77 m**); cross-dataset verification on **TUM-VI Room 1** reached **168.51 m**.
  - ATE scales with sequence duration (MH_01_easy at 374.74 m over 184 s) due to fundamental open-loop cubic drift dynamics ($\frac{1}{6} g \delta \mathbf{b}_\omega t^3$) under unassisted inertial navigation.

---

### Achievement Highlights

```
+-------------------------+-------------------------+-------------------------+-------------------------+
|         89 / 89         |      3.01 - 5.38 MB     |     2.48 - 2.79 ms/s    |       100% (7 / 7)      |
|  Regression Suite Pass  |     Peak Python Heap    |     Untraced Latency    |  TUM Format Validation  |
|    (10.03s runtime)     |   (Budget: <= 8.0 MB)   |   (Budget: <= 20 ms/s)  |   (Exact 1:1 Mapping)   |
+-------------------------+-------------------------+-------------------------+-------------------------+
```
