"""
Visualize EuRoC MH_01_easy IMU signals.
Generates 6 vertically stacked subplots for gyroscope and accelerometer axes.
Saves the resulting plot to outputs/plots/mh01_imu_signals.png.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent


def load_imu_data(csv_path: Path) -> pd.DataFrame:
    """Load IMU CSV dataset and validate required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"IMU data file not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read IMU CSV file '{csv_path}': {e}") from e

    # Clean whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    expected_cols = [
        df.columns[0],  # #timestamp [ns]
        "w_RS_S_x [rad s^-1]",
        "w_RS_S_y [rad s^-1]",
        "w_RS_S_z [rad s^-1]",
        "a_RS_S_x [m s^-2]",
        "a_RS_S_y [m s^-2]",
        "a_RS_S_z [m s^-2]",
    ]

    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from IMU CSV.")

    return df


def compute_imu_metrics(df: pd.DataFrame) -> dict:
    """
    Preserve timestamps as int64 nanoseconds and compute dataset duration,
    sample count, sampling frequency, and relative time in seconds.
    """
    ts_col = df.columns[0]
    # Ensure int64 nanoseconds precision
    timestamps_ns = df[ts_col].astype(np.int64)

    sample_count = len(df)
    if sample_count < 2:
        raise ValueError("Insufficient IMU data points for analysis.")

    # Time differences in int64 nanoseconds
    dt_ns = timestamps_ns.diff().dropna().astype(np.int64)
    mean_dt_s = float(dt_ns.mean()) * 1e-9
    sampling_freq_hz = (1.0 / mean_dt_s) if mean_dt_s > 0 else 0.0

    # Total duration
    t_start_ns = timestamps_ns.iloc[0]
    t_end_ns = timestamps_ns.iloc[-1]
    duration_s = (t_end_ns - t_start_ns) * 1e-9

    # Relative time array in seconds from the first measurement
    relative_time_s = (timestamps_ns - t_start_ns).to_numpy(dtype=np.float64) * 1e-9

    return {
        "sample_count": sample_count,
        "duration_s": duration_s,
        "sampling_freq_hz": sampling_freq_hz,
        "relative_time_s": relative_time_s,
    }


def plot_imu_signals(
    df: pd.DataFrame,
    relative_time_s: np.ndarray,
    output_path: Path,
) -> None:
    """Create 6 vertically stacked subplots with shared time axis and save figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=6,
        ncols=1,
        sharex=True,
        figsize=(14, 12),
        dpi=150,
        constrained_layout=True,
    )

    plot_configs = [
        {
            "col": "w_RS_S_x [rad s^-1]",
            "title": "Gyroscope X",
            "ylabel": r"$\omega_x$ [rad/s]",
            "color": "#1f77b4",
        },
        {
            "col": "w_RS_S_y [rad s^-1]",
            "title": "Gyroscope Y",
            "ylabel": r"$\omega_y$ [rad/s]",
            "color": "#ff7f0e",
        },
        {
            "col": "w_RS_S_z [rad s^-1]",
            "title": "Gyroscope Z",
            "ylabel": r"$\omega_z$ [rad/s]",
            "color": "#2ca02c",
        },
        {
            "col": "a_RS_S_x [m s^-2]",
            "title": "Accelerometer X",
            "ylabel": r"$a_x$ [m/s²]",
            "color": "#d62728",
        },
        {
            "col": "a_RS_S_y [m s^-2]",
            "title": "Accelerometer Y",
            "ylabel": r"$a_y$ [m/s²]",
            "color": "#9467bd",
        },
        {
            "col": "a_RS_S_z [m s^-2]",
            "title": "Accelerometer Z",
            "ylabel": r"$a_z$ [m/s²]",
            "color": "#8c564b",
        },
    ]

    for ax, cfg in zip(axes, plot_configs):
        y_data = df[cfg["col"]].to_numpy(dtype=np.float64)
        ax.plot(relative_time_s, y_data, color=cfg["color"], linewidth=0.8, alpha=0.9)
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold", loc="left", pad=3)
        ax.set_ylabel(cfg["ylabel"], fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=9)

    axes[-1].set_xlabel("Time from start [s]", fontsize=11, fontweight="bold")
    fig.suptitle(
        "EuRoC MH_01_easy - IMU 6-Axis Signals (Gyroscope & Accelerometer)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Main execution entry point."""
    project_root = get_project_root()
    imu_rel_path = Path("data/raw/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv")
    output_rel_path = Path("outputs/plots/mh01_imu_signals.png")

    imu_path = project_root / imu_rel_path
    output_path = project_root / output_rel_path

    print("=" * 80)
    print("EUROC MAV IMU SIGNAL VISUALIZATION: MH_01_easy")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Input IMU CSV:  {imu_rel_path}")
    print(f"Output Plot:    {output_rel_path}\n")

    try:
        df_imu = load_imu_data(imu_path)
        metrics = compute_imu_metrics(df_imu)

        print("-" * 80)
        print("DATASET METRICS")
        print("-" * 80)
        print(f"Total IMU Samples:           {metrics['sample_count']:,}")
        print(f"Dataset Duration:            {metrics['duration_s']:.3f} s ({metrics['duration_s'] / 60:.2f} min)")
        print(f"Estimated Sampling Frequency: {metrics['sampling_freq_hz']:.2f} Hz")

        print("\nGenerating 6-axis IMU visualization...")
        plot_imu_signals(
            df=df_imu,
            relative_time_s=metrics["relative_time_s"],
            output_path=output_path,
        )

        print(f"Plot successfully saved to: {output_rel_path}")
        print("=" * 80)
        print("VISUALIZATION COMPLETE (Original dataset unmodified)")
        print("=" * 80)

    except Exception as exc:
        print(f"Error during IMU visualization: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
