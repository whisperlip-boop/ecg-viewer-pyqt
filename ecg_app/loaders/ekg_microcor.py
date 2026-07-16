import numpy as np

from ..constants import STANDARD_12_ORDER
from ..signal_processing import remove_isolated_spikes

_MAGIC = b"\x12\x34\x12\x34"
_HEADER_SIZE = 8
_RECORD_DTYPE = np.dtype(
    [
        ("ts", ">u4"),
        ("ctr", ">i4"),
        ("samples", ">i4", (8,)),
    ]
)
_UV_PER_COUNT = 0.1
_FALLBACK_FS = 550.0

# Raw channel slot -> clinical lead, confirmed against the vendor's microCOR
# PC software (P3.ekg reference render) by matching sign/magnitude pattern
# (positive leads I,II,III,aVF,V3-V6 vs negative aVR,aVL,V1,V2; II largest
# positive; aVR largest negative; V1 smallest overall). Slots 6 and 7 are
# stored with inverted polarity relative to the others.
# raw index: 0    1    2   3    4    5    6         7
# lead:      V1   V2   I   V4   V5   V6   -II       -V3


def load_ekg_waveform(ekg_path: str):
    """Load a 12-lead ECG from a microCOR (Infron Ltd) ``.ekg`` file.

    This is the proprietary recording format of the microCOR portable ECG
    device used in the Gazi University GU-ECG database (PTCA pre/inflation/
    post-inflation recordings). The format has no public specification; the
    layout below was reverse engineered from sample files plus the device
    specs published with the dataset (8800 Hz acquisition, 24-bit, 0.1 uV
    amplitude resolution):

    - 8-byte file header: 4-byte magic ``12 34 12 34`` + 4 unknown bytes.
    - Fixed 40-byte records, big-endian, repeating to EOF:
      4-byte unix timestamp (seconds) + 4-byte running record counter +
      8 signed 32-bit raw ADC samples, in the fixed slot order
      ``[V1, V2, I, V4, V5, V6, -II, -V3]`` (slots 6-7 inverted polarity).

    The per-record timestamp is used to derive the effective sample rate
    directly from the file instead of assuming the device's nominal
    acquisition rate, since internal decimation/oversampling behavior
    between the 8800 Hz ADC and the stored record rate is undocumented.

    Caution:
        The channel slots for I, II, V1, V2, V3 (and the polarity inversion
        on slots 6-7) were confirmed by matching sign/magnitude patterns
        against the vendor's own microCOR PC software output for a sample
        recording. The relative order of V4, V5, V6 among raw slots 3-5 was
        *not* individually confirmed (they are mutually similar positive
        precordial leads) and is assumed to follow the natural slot
        sequence; double-check against the official viewer if the exact
        V4/V5/V6 identity matters clinically.

    Args:
        ekg_path: Path to the .ekg file.

    Returns:
        Tuple of (signal, fields) where signal has shape (samples, 12) in mV.

    Raises:
        ValueError: If the file is too short, the magic number does not
            match, or it contains no complete records.
    """
    with open(ekg_path, "rb") as f:
        data = f.read()

    if len(data) < _HEADER_SIZE + _RECORD_DTYPE.itemsize:
        raise ValueError("EKG file is too short to contain a valid header and records.")

    magic = data[:4]
    if magic != _MAGIC:
        raise ValueError(
            "Unrecognized .ekg file: expected microCOR magic bytes "
            f"{_MAGIC.hex()}, got {magic.hex()}."
        )

    body = data[_HEADER_SIZE:]
    n = len(body) // _RECORD_DTYPE.itemsize
    if n <= 0:
        raise ValueError("EKG file contains no complete records.")

    trailing_bytes = len(body) - n * _RECORD_DTYPE.itemsize
    records = np.frombuffer(body, dtype=_RECORD_DTYPE, count=n)

    timestamps = records["ts"].astype(np.int64)
    duration_s = int(timestamps[-1] - timestamps[0])
    if duration_s > 0:
        fs = n / duration_s
        fs_source = "measured_from_timestamps"
    else:
        fs = _FALLBACK_FS
        fs_source = "fallback_nominal_estimate"

    raw_mv = records["samples"].astype(np.float64) * (_UV_PER_COUNT / 1000.0)

    v1 = raw_mv[:, 0]
    v2 = raw_mv[:, 1]
    i_lead = raw_mv[:, 2]
    v4 = raw_mv[:, 3]
    v5 = raw_mv[:, 4]
    v6 = raw_mv[:, 5]
    ii_lead = -raw_mv[:, 6]
    v3 = -raw_mv[:, 7]

    iii_lead = ii_lead - i_lead
    avr = -(i_lead + ii_lead) / 2.0
    avl = i_lead - ii_lead / 2.0
    avf = ii_lead - i_lead / 2.0

    signal = np.column_stack(
        [i_lead, ii_lead, iii_lead, avr, avl, avf, v1, v2, v3, v4, v5, v6]
    )
    signal = remove_isolated_spikes(signal)

    fields: dict = {
        "fs": fs,
        "sig_name": STANDARD_12_ORDER.copy(),
        "units": ["mV"] * 12,
        "source": "ekg_microcor",
        "layout": "lead-major",
        "channel_count": 12,
        "sample_count": n,
        "scale_mode": "ekg_0.1uv_per_count",
        "ekg_path": ekg_path,
        "fs_source": fs_source,
        "recording_start_unix": int(timestamps[0]),
        "lead_order_confirmed": ["I", "II", "V1", "V2", "V3"],
        "lead_order_assumed": ["V4", "V5", "V6"],
        "dropped_trailing_bytes": trailing_bytes,
    }
    return signal, fields
