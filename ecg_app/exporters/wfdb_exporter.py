"""WFDB (.dat + .hea) exporter for ECG signal data."""

import json
import os
from typing import Any

import numpy as np
import wfdb


_ADC_GAIN = 1000.0  # adu/mV — 1 µV per ADC unit
_METRICS_PREFIX = "ecg_metrics:"


def export_wfdb(
    signal: np.ndarray,
    fields: dict[str, Any],
    out_path: str,
    measurements: dict[str, Any] | None = None,
) -> None:
    """Export ECG signal to WFDB format (.dat + .hea).

    Strips the extension from out_path to derive the record name,
    then writes a 16-bit .dat file and matching .hea header in the
    same directory.  When *measurements* is provided it is embedded as
    a ``# ecg_metrics:{JSON}`` comment in the .hea file.

    Args:
        signal: Signal array of shape (samples, leads) in mV.
        fields: Metadata dict; must contain 'fs' and 'sig_name'.
        out_path: Output .hea file path (extension stripped for record name).
        measurements: Optional dict with keys hr, pr, qrs, qt, qtc,
            p_axis, r_axis, t_axis (int or None, values in BPM/ms/deg).

    Returns:
        None

    Raises:
        OSError: If files cannot be written.
    """
    base = os.path.splitext(out_path)[0]
    record_name = os.path.basename(base)
    write_dir = os.path.dirname(out_path) or "."

    fs = int(fields["fs"])
    sig_name = [str(n) for n in fields["sig_name"]]
    n_leads = signal.shape[1]

    comments = [f"{_METRICS_PREFIX}{json.dumps(measurements)}"] if measurements else None

    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["mV"] * n_leads,
        sig_name=sig_name,
        p_signal=signal.astype(np.float64),
        fmt=["16"] * n_leads,
        adc_gain=[_ADC_GAIN] * n_leads,
        baseline=[0] * n_leads,
        write_dir=write_dir,
        comments=comments,
    )
