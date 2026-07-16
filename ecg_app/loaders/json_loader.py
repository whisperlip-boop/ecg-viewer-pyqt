import json
from datetime import datetime

import numpy as np

from ..constants import STANDARD_12_ORDER

_METRICS_PREFIX = "# ecg_metrics:"


def load_json_waveform(json_path: str):
    """Load a 12-lead ECG from this app's JSON format.

    Expected structure::

        {
          "patient_id": "...",
          "measure_time": "2025-07-02T11:56:36.611031+09:00",
          "samplerate": 500.0,
          "leads": {"I": [...], "II": [...], ..., "V6": [...]},
          "measurements": {...}   # optional
        }

    Lead keys are matched case-insensitively (e.g. "AVR" == "aVR") so the
    format tolerates either casing convention.

    Args:
        json_path: Path to the .json file.

    Returns:
        Tuple of (signal, fields) where signal has shape (samples, 12) in mV.

    Raises:
        ValueError: If required lead keys are missing or data is unparseable.
    """
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    leads = data.get("leads")
    if not isinstance(leads, dict):
        raise ValueError("JSON file has no 'leads' object.")

    expected_leads = STANDARD_12_ORDER.copy()
    leads_upper = {str(key).strip().upper(): value for key, value in leads.items()}

    missing = [lead for lead in expected_leads if lead.upper() not in leads_upper]
    if missing:
        raise ValueError(
            "JSON is missing required 12-lead entries.\n"
            f"Missing: {missing}\n"
            f"Available leads: {list(leads.keys())}"
        )

    try:
        columns = [
            np.asarray(leads_upper[lead.upper()], dtype=np.float64)
            for lead in expected_leads
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("JSON lead data contains unparseable values.") from exc

    lengths = [len(col) for col in columns]
    n = min(lengths) if lengths else 0
    if n <= 0:
        raise ValueError("JSON waveform length is 0.")

    signal = np.column_stack([col[:n] for col in columns])

    amp99 = float(np.percentile(np.abs(signal), 99))
    scale_mode = "json_raw"
    if amp99 > 20.0:
        signal = signal / 1000.0
        scale_mode = "json_div_1000"

    fs = float(data.get("samplerate") or data.get("sample_rate") or data.get("fs") or 500.0)

    fields: dict = {
        "fs": fs,
        "sig_name": expected_leads,
        "units": ["mV"] * 12,
        "source": "json",
        "layout": "lead-major",
        "channel_count": 12,
        "sample_count": n,
        "scale_mode": scale_mode,
        "json_path": json_path,
        "patient_id": data.get("patient_id"),
    }

    measure_time = data.get("measure_time")
    if measure_time:
        try:
            dt = datetime.fromisoformat(str(measure_time))
            fields["acquisition_date"] = dt.strftime("%m-%d-%Y")
            fields["acquisition_time"] = dt.strftime("%H:%M:%S")
        except ValueError:
            pass

    measurements = data.get("measurements")
    if isinstance(measurements, dict):
        fields["measurements"] = measurements

    return signal, fields
