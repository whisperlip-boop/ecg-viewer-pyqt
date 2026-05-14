"""MAT + HEA exporter for ECG signal data."""

import json
import os
from typing import Any

import numpy as np
from scipy.io import savemat


_ADC_GAIN = 1000.0  # adu/mV — 1 µV per ADC unit
_METRICS_PREFIX = "# ecg_metrics:"


def export_mat_hea(
    signal: np.ndarray,
    fields: dict[str, Any],
    out_path: str,
    measurements: dict[str, Any] | None = None,
) -> None:
    """Export ECG signal to MATLAB (.mat) + WFDB header (.hea) format.

    The .mat file stores a 'val' variable of shape (n_leads, n_samples) as int16
    at 1000 adu/mV, matching the convention expected by the existing mat_hea loader.
    The companion .hea is written alongside the .mat file.  When *measurements*
    is provided it is embedded as a ``# ecg_metrics:{JSON}`` comment line in the .hea.

    Args:
        signal: Signal array of shape (samples, leads) in mV.
        fields: Metadata dict; must contain 'fs' and 'sig_name'.
        out_path: Output .mat file path.
        measurements: Optional dict with keys hr, pr, qrs, qt, qtc,
            p_axis, r_axis, t_axis (int or None, values in BPM/ms/deg).

    Returns:
        None

    Raises:
        OSError: If files cannot be written.
    """
    base = os.path.splitext(out_path)[0]
    record_name = os.path.basename(base)
    mat_filename = os.path.basename(out_path)

    fs = fields["fs"]
    sig_name = [str(n) for n in fields["sig_name"]]
    n_sig = len(sig_name)
    n_samples = signal.shape[0]

    # WFDB MAT convention: shape is (n_leads, n_samples)
    val = np.round(signal.T * _ADC_GAIN).astype(np.int16)
    savemat(out_path, {"val": val})

    hea_path = base + ".hea"
    with open(hea_path, "w", encoding="utf-8") as f:
        f.write(f"{record_name} {n_sig} {int(fs)} {n_samples}\n")
        if measurements:
            f.write(f"{_METRICS_PREFIX}{json.dumps(measurements)}\n")
        for name in sig_name:
            f.write(
                f"{mat_filename} 16+24 {int(_ADC_GAIN)}/mV 16 0 0 0 0 {name}\n"
            )
