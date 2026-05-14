"""CSV exporter for ECG signal data."""

import csv
import json
from typing import Any

import numpy as np


def export_csv(
    signal: np.ndarray,
    fields: dict[str, Any],
    out_path: str,
    measurements: dict[str, Any] | None = None,
) -> None:
    """Export ECG signal to CSV format.

    Writes an optional ``# ecg_metrics:{JSON}`` comment line first (when
    *measurements* is provided), then a header row with lead names, followed
    by one sample per row in mV with 6 decimal places.

    Args:
        signal: Signal array of shape (samples, leads) in mV.
        fields: Metadata dict; must contain 'sig_name'.
        out_path: Output .csv file path.
        measurements: Optional dict with keys hr, pr, qrs, qt, qtc,
            p_axis, r_axis, t_axis (int or None, values in BPM/ms/deg).

    Returns:
        None

    Raises:
        OSError: If the file cannot be written.
    """
    sig_names = fields["sig_name"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if measurements:
            f.write(f"# ecg_metrics:{json.dumps(measurements)}\n")
        writer = csv.writer(f)
        writer.writerow(sig_names)
        for row in signal:
            writer.writerow([f"{v:.6f}" for v in row])
