import os
from pathlib import Path

import numpy as np

from ..constants import AMP_GAIN_CORRECTION, STANDARD_12_ORDER
from ..signal_processing import derive_12_from_base8, remove_isolated_spikes


def parse_mwf_fixed_family(raw):
    """
    Fixed parser for confirmed Nihon Kohden sample series.
    """
    if len(raw) < 344 + 80000:
        raise ValueError("MWF file is smaller than expected.")

    return {
        "fs": 500.0,
        "count_to_mV": 1.0 / 1000.0,
        "channel_count": 8,
        "sample_count": 5000,
        "wav_offset": 344,
        "wav_length": 80000,
    }


def load_dat_mwf(dat_path, mwf_path):
    if not os.path.exists(mwf_path):
        raise FileNotFoundError(f"MWF file not found: {mwf_path}")

    raw = Path(mwf_path).read_bytes()
    info = parse_mwf_fixed_family(raw)

    payload = raw[info["wav_offset"]:info["wav_offset"] + info["wav_length"]]
    arr = np.frombuffer(payload, dtype="<i2")

    expected = info["channel_count"] * info["sample_count"]
    if len(arr) != expected:
        raise ValueError(
            f"Waveform length mismatch. expected={expected}, actual={len(arr)}"
        )

    base8 = arr.reshape(info["channel_count"], info["sample_count"]).T.astype(np.float64)
    base8 *= info["count_to_mV"] * AMP_GAIN_CORRECTION

    signal = derive_12_from_base8(base8)
    signal = remove_isolated_spikes(signal)

    fields = {
        "fs": info["fs"],
        "sig_name": STANDARD_12_ORDER.copy(),
        "units": ["mV"] * 12,
        "source": "mwf",
        "layout": "lead-major",
        "channel_count": info["channel_count"],
        "sample_count": info["sample_count"],
        "count_to_mV": info["count_to_mV"],
        "wav_offset": info["wav_offset"],
        "wav_length": info["wav_length"],
        "dat_path": dat_path,
        "mwf_path": mwf_path,
    }

    # Attach companion XML path if a same-named .xml exists alongside the MWF
    xml_candidate = Path(mwf_path).with_suffix(".xml")
    if xml_candidate.exists():
        fields["companion_xml_path"] = str(xml_candidate)

    return signal, fields
