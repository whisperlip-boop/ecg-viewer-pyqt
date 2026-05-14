import csv
import io
import json

import numpy as np

from ..constants import STANDARD_12_ORDER

_METRICS_PREFIX = "# ecg_metrics:"


def load_csv_waveform(csv_path: str, fs: float = 500.0):
    """Load a 12-lead ECG from CSV format.

    Supports an optional ``# ecg_metrics:{JSON}`` comment line at the top
    (written by the CSV exporter) that carries clinical measurements across
    format conversions.

    Args:
        csv_path: Path to the .csv file.
        fs: Sampling frequency in Hz (default 500).

    Returns:
        Tuple of (signal, fields) where signal has shape (samples, 12) in mV.

    Raises:
        ValueError: If required lead columns are missing or data is unparseable.
    """
    expected_leads = STANDARD_12_ORDER.copy()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        raw_lines = f.readlines()

    measurements = None
    data_lines = []
    for line in raw_lines:
        if line.startswith(_METRICS_PREFIX):
            try:
                measurements = json.loads(line[len(_METRICS_PREFIX):].strip())
            except Exception:
                pass
        else:
            data_lines.append(line)

    reader = csv.DictReader(io.StringIO("".join(data_lines)))
    fieldnames = reader.fieldnames or []
    cols_upper = {str(col).strip().upper(): col for col in fieldnames}

    missing = [lead for lead in expected_leads if lead.upper() not in cols_upper]
    if missing:
        raise ValueError(
            "CSV is missing required 12-lead columns.\n"
            f"Missing: {missing}\n"
            f"Available columns: {fieldnames}"
        )

    ordered_cols = [cols_upper[lead.upper()] for lead in expected_leads]
    rows = []

    for row_idx, row in enumerate(reader, start=2):
        values = []
        for col in ordered_cols:
            raw_value = row.get(col, "")
            text = "" if raw_value is None else str(raw_value).strip()
            if text == "":
                raise ValueError(
                    "CSV row contains an unparseable value.\n"
                    f"Row: {row_idx}, Column: {col}, Value: {raw_value}"
                )
            try:
                values.append(float(text))
            except Exception as exc:
                raise ValueError(
                    "CSV row contains an unparseable value.\n"
                    f"Row: {row_idx}, Column: {col}, Value: {raw_value}"
                ) from exc
        rows.append(values)

    if not rows:
        raise ValueError("CSV file contains no data rows.")

    signal = np.asarray(rows, dtype=np.float64)

    amp99 = float(np.percentile(np.abs(signal), 99))
    scale_mode = "csv_raw"
    if amp99 > 20.0:
        signal = signal / 1000.0
        scale_mode = "csv_div_1000"

    fields: dict = {
        "fs": fs,
        "sig_name": expected_leads,
        "units": ["mV"] * 12,
        "source": "csv",
        "layout": "lead-major",
        "channel_count": 12,
        "sample_count": signal.shape[0],
        "scale_mode": scale_mode,
        "csv_path": csv_path,
    }
    if measurements:
        fields["measurements"] = measurements
    return signal, fields
