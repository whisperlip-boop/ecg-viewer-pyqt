import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ._utils import find_companion_file

_METRICS_PREFIX = "# ecg_metrics:"


def _safe_float(text, default=None):
    try:
        return float(text)
    except Exception:
        return default


def _parse_hea_for_mat(hea_path: str) -> dict:
    """Parse a WFDB-style .hea file written for a .mat record.

    Also extracts an optional ``# ecg_metrics:{JSON}`` comment line
    embedded by the MAT exporter and returns it under the
    ``measurements`` key when present.

    Args:
        hea_path: Path to the .hea file.

    Returns:
        Dict with record metadata and optional 'measurements' key.

    Raises:
        ValueError: If the file is empty or has an unexpected format.
    """
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as file:
        all_lines = [line.strip() for line in file if line.strip()]

    if not all_lines:
        raise ValueError(f"Empty HEA file: {hea_path}")

    measurements = None
    lines = []
    for line in all_lines:
        if line.startswith(_METRICS_PREFIX):
            try:
                measurements = json.loads(line[len(_METRICS_PREFIX):])
            except Exception:
                pass
        else:
            lines.append(line)

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("HEA first line format is invalid.")

    record_name = first[0]
    n_sig = int(first[1])
    fs = float(first[2])
    sig_len = int(first[3])

    sig_name = []
    units = []
    gains = []
    baselines = []

    sig_lines = lines[1:1 + n_sig]
    if len(sig_lines) < n_sig:
        raise ValueError("HEA channel line count is insufficient.")

    for line in sig_lines:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"HEA channel line parse error: {line}")

        gain_unit = parts[2]
        lead_name = parts[-1]

        if "/" in gain_unit:
            gain_text, unit = gain_unit.split("/", 1)
        else:
            gain_text, unit = gain_unit, "adu"

        gain = _safe_float(gain_text, 1.0)
        if gain == 0:
            gain = 1.0

        baseline = 0.0
        if len(parts) >= 5:
            baseline = _safe_float(parts[4], 0.0)

        sig_name.append(lead_name)
        units.append(unit)
        gains.append(float(gain))
        baselines.append(float(baseline))

    result = {
        "record_name": record_name,
        "n_sig": n_sig,
        "fs": fs,
        "sig_len": sig_len,
        "sig_name": sig_name,
        "units": units,
        "gains": np.asarray(gains, dtype=np.float64),
        "baselines": np.asarray(baselines, dtype=np.float64),
    }
    if measurements:
        result["measurements"] = measurements
    return result


def _robust_amp(signal):
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), 99))


def load_mat_hea(base_path):
    hea_path = find_companion_file(base_path, [".hea"])
    mat_path = find_companion_file(base_path, [".mat"])

    if not hea_path or not mat_path:
        raise FileNotFoundError("MAT+HEA file pair not found.")

    meta = _parse_hea_for_mat(hea_path)

    mat = loadmat(mat_path)
    if "val" not in mat:
        raise ValueError("MAT file has no 'val' variable.")

    val = np.asarray(mat["val"])
    if val.ndim != 2:
        raise ValueError(f"MAT val shape is not 2D: {val.shape}")

    n_sig = meta["n_sig"]
    sig_len = meta["sig_len"]

    if val.shape == (n_sig, sig_len):
        digital = val.astype(np.float64).T
    elif val.shape == (sig_len, n_sig):
        digital = val.astype(np.float64)
    else:
        raise ValueError(
            f"MAT/HEA shape mismatch: mat={val.shape}, hea=({n_sig}, {sig_len}) expected"
        )

    gains = meta["gains"].copy()
    baselines = meta["baselines"].copy()
    gains[gains == 0] = 1.0

    signal = (digital - baselines[np.newaxis, :]) / gains[np.newaxis, :]
    amp = _robust_amp(signal)

    scale_mode = "hea_gain"
    if amp < 0.05:
        signal = digital / 1000.0
        scale_mode = "digital_div_1000"

    fields: dict = {
        "fs": meta["fs"],
        "sig_name": meta["sig_name"],
        "units": ["mV"] * len(meta["sig_name"]),
        "source": "mat",
        "layout": "lead-major",
        "channel_count": len(meta["sig_name"]),
        "sample_count": signal.shape[0],
        "scale_mode": scale_mode,
        "record_name": meta["record_name"],
        "hea_path": hea_path,
        "mat_path": mat_path,
    }
    if meta.get("measurements"):
        fields["measurements"] = meta["measurements"]
    return signal, fields
