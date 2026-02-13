"""
Visualize the distribution of error step times from complete_step_annotations.json.

Extracts start_time (and duration) for steps with has_errors=True, then plots
histogram and KDE. Invalid times (e.g. -1.0) are excluded.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANNOTATIONS_PATH = Path(__file__).resolve().parent / "metadata" / "complete_step_annotations.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "metadata" / "error_step_times_distribution.png"


def load_error_step_times(path: Path) -> tuple[list[float], list[float]]:
    """Load annotations and return (start_times, durations) for steps with has_errors=True."""
    with open(path, "r") as f:
        data = json.load(f)

    start_times: list[float] = []
    durations: list[float] = []

    for recording in data.values():
        for step in recording.get("steps", []):
            if not step.get("has_errors"):
                continue
            st, et = step.get("start_time"), step.get("end_time")
            if st is None or et is None or st < 0 or et < 0:
                continue
            start_times.append(st)
            durations.append(et - st)

    return start_times, durations


def main() -> None:
    start_times, durations = load_error_step_times(ANNOTATIONS_PATH)
    if not start_times:
        print("No valid error step times found.")
        return

    print(f"Error steps: {len(start_times)}")
    print(f"Start time (s): min={min(start_times):.1f}, max={max(start_times):.1f}, mean={np.mean(start_times):.1f}")
    print(f"Duration (s):   min={min(durations):.1f}, max={max(durations):.1f}, mean={np.mean(durations):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Start time distribution (when in the video errors occur)
    ax = axes[0]
    ax.hist(start_times, bins=min(50, len(set(np.round(np.array(start_times)))) or 20), density=True, alpha=0.7, color="steelblue", edgecolor="white")
    try:
        from scipy import stats
        kde = stats.gaussian_kde(start_times)
        x = np.linspace(min(start_times), max(start_times), 200)
        ax.plot(x, kde(x), color="darkblue", linewidth=2, label="KDE")
    except ImportError:
        pass
    ax.set_xlabel("Error step start time (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("When do error steps start? (start_time)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Duration distribution
    ax = axes[1]
    ax.hist(durations, bins=min(40, max(10, len(start_times) // 5)), density=True, alpha=0.7, color="coral", edgecolor="white")
    try:
        from scipy import stats
        kde = stats.gaussian_kde(durations)
        x = np.linspace(max(0, min(durations)), max(durations), 200)
        ax.plot(x, kde(x), color="darkred", linewidth=2, label="KDE")
    except ImportError:
        pass
    ax.set_xlabel("Error step duration (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("How long do error steps last? (duration)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Distribution of error step times (complete_step_annotations.json)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
