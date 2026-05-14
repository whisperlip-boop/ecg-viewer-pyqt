import ctypes
import io
import json
import sys
import os
from pathlib import Path

from PyQt5.QtGui import QIcon

import csv
from array import array

import numpy as np
import wfdb
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

import base64
import xml.etree.ElementTree as ET

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QLabel,
    QStackedLayout,
    QMessageBox,
    QDialog,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QMovie

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


STANDARD_12_ORDER = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

def resource_path(relative_path):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# =========================================================
# Display constants
# =========================================================
MAIN_Y_LIM = 3.0
ZOOM_Y_LIM = 3.0

Y_MAJOR_STEP = 0.5
Y_MINOR_STEP = 0.1

ZOOM_DIALOG_WIDTH = 1900
ZOOM_DIALOG_HEIGHT = 640

# ECG paper grid
SEC_PER_MM = 0.04
MV_PER_MM = 0.1
AMP_GAIN_CORRECTION = 1.37  # 파형 높이 조절 (Nihon Kohden 샘플 기준, 필요에 따라 조정 가능)

GRID_COLOR = "#D98391"
GRID_MAJOR_LINESTYLE = "--"
GRID_MINOR_LINESTYLE = ":"
GRID_MAJOR_WIDTH = 0.8
GRID_MINOR_WIDTH = 0.4
GRID_MINOR_ALPHA = 0.6

PAPER_FACE_COLOR = "#fff7f7"


def safe_bandpass_filter(signal, lowcut, highcut, fs, order=3):
    if signal is None or len(signal) < max(32, order * 8):
        return signal

    nyquist = 0.5 * fs
    if nyquist <= 0:
        return signal

    low = max(lowcut / nyquist, 1e-6)
    high = min(highcut / nyquist, 0.999999)

    if low >= high:
        return signal

    try:
        b, a = butter(order, [low, high], btype="band")
        padlen = 3 * (max(len(a), len(b)) - 1)
        if signal.shape[0] <= padlen:
            return signal
        return filtfilt(b, a, signal, axis=0)
    except Exception:
        return signal


def remove_baseline_wander(signal, fs, cutoff=0.5, order=2):
    if signal is None or len(signal) < 32:
        return signal

    nyquist = 0.5 * fs
    if nyquist <= 0:
        return signal

    wn = cutoff / nyquist
    wn = min(max(wn, 1e-6), 0.999999)

    try:
        b, a = butter(order, wn, btype="high")
        padlen = 3 * (max(len(a), len(b)) - 1)
        if signal.shape[0] <= padlen:
            return signal
        return filtfilt(b, a, signal, axis=0)
    except Exception:
        return signal


def remove_isolated_spikes(signal, z=8.0):
    out = signal.copy()

    for ch in range(out.shape[1]):
        x = out[:, ch]
        if len(x) < 5:
            continue

        local_med = (np.roll(x, 1) + np.roll(x, -1)) / 2.0
        diff = x - local_med

        core = diff[1:-1]
        mad = np.median(np.abs(core - np.median(core)))
        if mad < 1e-12:
            continue

        thresh = z * 1.4826 * mad
        mask = np.abs(diff) > thresh
        mask[0] = False
        mask[-1] = False

        out[mask, ch] = local_med[mask]

    return out


def find_companion_file(base_path, extensions):
    base = Path(base_path)
    folder = base.parent
    stem = base.name

    candidates = []
    for ext in extensions:
        candidates.extend([
            folder / f"{stem}{ext.lower()}",
            folder / f"{stem}{ext.upper()}",
            folder / f"{stem}{ext.capitalize()}",
        ])

    for p in candidates:
        if p.exists():
            return str(p)

    return None


def normalize_input_to_base_path(file_path):
    p = Path(file_path)
    suffix = p.suffix.lower()
    if suffix in [".dat", ".hea", ".mwf", ".mat", ".csv"]:
        return str(p.with_suffix(""))
    if suffix in [".xml", ".ecg"]:
        return str(p)
    return str(p)


def parse_mwf_fixed_family(raw):
    """
    현재 확인된 Nihon Kohden 샘플 계열에 맞춘 고정 파서.
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


def derive_12_from_base8(base8):
    """
    base8 column order:
      0: I
      1: II
      2: V1
      3: V2
      4: V3
      5: V4
      6: V5
      7: V6
    """
    if base8.shape[1] != 8:
        raise ValueError("base8 must have 8 channels")

    I = base8[:, 0]
    II = base8[:, 1]

    V1 = base8[:, 2]
    V2 = base8[:, 3]
    V3 = base8[:, 4]
    V4 = base8[:, 5]
    V5 = base8[:, 6]
    V6 = base8[:, 7]

    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - (II / 2.0)
    aVF = II - (I / 2.0)

    signal12 = np.column_stack([
        I, II, III,
        aVR, aVL, aVF,
        V1, V2, V3, V4, V5, V6
    ])
    return signal12


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

    xml_candidate = Path(mwf_path).with_suffix(".xml")
    if xml_candidate.exists():
        fields["companion_xml_path"] = str(xml_candidate)

    return signal, fields


def _safe_float(text, default=None):
    try:
        return float(text)
    except Exception:
        return default

_METRICS_PREFIX = "# ecg_metrics:"

def _parse_hea_for_mat(hea_path):
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        all_lines = [line.strip() for line in f if line.strip()]

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

        gain_unit = parts[2]      # 예: 306000/mV
        lead_name = parts[-1]     # 예: I, II, V1 ...

        if "/" in gain_unit:
            gain_text, unit = gain_unit.split("/", 1)
        else:
            gain_text, unit = gain_unit, "adu"

        gain = _safe_float(gain_text, 1.0)
        if gain == 0:
            gain = 1.0

        # WFDB 헤더는 baseline/initvalue 등의 위치가 파일마다 조금 다를 수 있어
        # 너무 공격적으로 해석하지 않고 baseline만 안전하게 추출
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

    fields = {
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

def load_csv_waveform(csv_path, fs=500.0):
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
            "CSV file is missing required 12-lead columns.\n"
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
                    "CSV contains an empty cell that cannot be parsed.\n"
                    f"Row: {row_idx}, Column: {col}, Value: {raw_value!r}"
                )
            try:
                values.append(float(text))
            except Exception as exc:
                raise ValueError(
                    "CSV contains a non-numeric value that cannot be parsed.\n"
                    f"Row: {row_idx}, Column: {col}, Value: {raw_value!r}"
                ) from exc
        rows.append(values)

    if not rows:
        raise ValueError("CSV file contains no data rows.")

    signal = np.asarray(rows, dtype=np.float64)

    # Auto-scale: if values appear to be in uV (amp > 20), divide by 1000 to normalize to mV
    amp99 = float(np.percentile(np.abs(signal), 99))
    scale_mode = "csv_raw"
    if amp99 > 20.0:
        signal = signal / 1000.0
        scale_mode = "csv_div_1000"

    fields = {
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


def _hativ_decode_waveform_data(data_text, lead_type, lead_count):
    payload = "".join((data_text or "").split())
    if not payload or lead_count <= 0:
        return None

    raw = base64.b64decode(payload)

    lead_type_lower = (lead_type or "").strip().lower()
    if lead_type_lower == "int32":
        dtype = "<i4"
    elif lead_type_lower == "int16":
        dtype = "<i2"
    else:
        raise ValueError(f"Unsupported HATIV LeadType: {lead_type}")

    arr = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if lead_count > 0 and arr.size >= lead_count:
        arr = arr[:lead_count]
    return arr


def load_hativ_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if _xml_local_name(root.tag) != "HativECG":
        raise ValueError("HATIV XML root is not HativECG.")

    waveform = _xml_find_first(root, "WaveForm")
    if waveform is None:
        raise ValueError("WaveForm not found in HATIV XML.")

    fs = float(_xml_get_child_text(waveform, "SampleRate", "500") or "500")
    ratio_value = float(_xml_get_child_text(waveform, "RatioValue", "1.0") or "1.0")
    measure_mode = _xml_get_child_text(_xml_find_first(root, "ECGInfo"), "MeasureMode", None)

    lead_map = {}
    for lead_data in list(waveform):
        if _xml_local_name(lead_data.tag) != "LeadData":
            continue

        lead_id = (_xml_get_child_text(lead_data, "LeadID", "") or "").strip()
        lead_count = int(_xml_get_child_text(lead_data, "LeadCount", "0") or "0")
        lead_type = _xml_get_child_text(lead_data, "LeadType", "int32")
        waveform_data = _xml_get_child_text(lead_data, "WaveFormData", "")

        if not lead_id or lead_count <= 0 or not waveform_data:
            continue

        arr = _hativ_decode_waveform_data(waveform_data, lead_type, lead_count)
        if arr is None or arr.size == 0:
            continue

        arr = (arr - np.mean(arr)) * ratio_value
        lead_map[lead_id.upper()] = arr

    if not lead_map:
        raise ValueError("No valid waveform data in HATIV XML.")

    if "I" in lead_map and "II" in lead_map:
        i_lead = lead_map["I"]
        ii_lead = lead_map["II"]
        n = min(len(i_lead), len(ii_lead))
        if n <= 0:
            raise ValueError("HATIV 6-lead waveform length is 0.")

        i_lead = i_lead[:n]
        ii_lead = ii_lead[:n]
        iii = ii_lead - i_lead
        avr = -(i_lead + ii_lead) / 2.0
        avl = i_lead - (ii_lead / 2.0)
        avf = ii_lead - (i_lead / 2.0)

        signal = np.column_stack([i_lead, ii_lead, iii, avr, avl, avf])
        sig_name = ["I", "II", "III", "aVR", "aVL", "aVF"]
        layout = "lead-major-6"
    elif "I" in lead_map:
        signal = lead_map["I"][:, np.newaxis]
        sig_name = ["I"]
        layout = "lead-major-1"
    else:
        available = sorted(lead_map.keys())
        raise ValueError(f"Unsupported lead combination in HATIV XML: {available}")

    signal = remove_isolated_spikes(signal)

    fields = {
        "fs": fs,
        "sig_name": sig_name,
        "units": ["mV"] * len(sig_name),
        "source": "hativ_xml",
        "layout": layout,
        "channel_count": len(sig_name),
        "sample_count": signal.shape[0],
        "scale_mode": "hativ_ratio_value",
        "count_to_mV": ratio_value,
        "xml_path": xml_path,
        "measure_mode": measure_mode,
        "device_type": _xml_get_child_text(_xml_find_first(root, "DeviceInfo"), "DeviceType", None),
        "serial_number": _xml_get_child_text(_xml_find_first(root, "DeviceInfo"), "SerialNumber", None),
        "app_name": _xml_get_child_text(_xml_find_first(root, "AppInfo"), "Name", None),
        "user_id": _xml_get_child_text(_xml_find_first(root, "UserInfo"), "UserID", None),
        "measure_start_time": _xml_get_child_text(_xml_find_first(root, "ECGInfo"), "MeasureStartTime", None),
    }
    return signal, fields


class PhilipsLzwDecoder:
    """
    Minimal Philips Sierra XLI/LZW decoder adapted for in-app use.
    """

    def __init__(self, buffer, bits):
        self.buffer = buffer
        self.offset = 0
        self.bits = bits
        self.max_code = (1 << bits) - 2

        self.bit_count = 0
        self.bit_buffer = 0

        self.previous = array("B")
        self.next_code = 256
        self.strings = {code: array("B", [code]) for code in range(256)}

        self.current = None
        self.position = 0

    def read(self):
        if self.current is None or self.position == len(self.current):
            self.current = self._read_next_string()
            self.position = 0

        if len(self.current) > 0:
            byte = self.current[self.position] & 0xFF
            self.position += 1
            return byte

        return -1

    def _read_next_string(self):
        code = self._read_codepoint()
        if 0 <= code <= self.max_code:
            if code not in self.strings:
                data = self.previous[:]
                data.append(self.previous[0])
                self.strings[code] = data
            else:
                data = self.strings[code]

            if len(self.previous) > 0 and self.next_code <= self.max_code:
                next_data = self.previous[:]
                next_data.append(data[0])
                self.strings[self.next_code] = next_data
                self.next_code += 1

            self.previous = data
            return data

        return array("B")

    def _read_codepoint(self):
        while self.bit_count <= 24:
            if self.offset < len(self.buffer):
                next_byte = self.buffer[self.offset]
                self.offset += 1
                self.bit_buffer |= ((next_byte & 0xFF) << (24 - self.bit_count)) & 0xFFFFFFFF
                self.bit_count += 8
            elif self.bit_count < self.bits:
                return -1
            else:
                break

        code = (self.bit_buffer >> (32 - self.bits)) & 0x0000FFFF
        self.bit_buffer = ((self.bit_buffer & 0xFFFFFFFF) << self.bits) & 0xFFFFFFFF
        self.bit_count -= self.bits
        return code


def _xml_local_name(tag):
    if not isinstance(tag, str):
        return ""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _xml_find_first(root, local_name):
    for elem in root.iter():
        if _xml_local_name(elem.tag) == local_name:
            return elem
    return None


def _xml_get_text(parent, local_name, default=None):
    if parent is None:
        return default
    for elem in parent.iter():
        if elem is parent:
            continue
        if _xml_local_name(elem.tag) == local_name:
            text = elem.text.strip() if elem.text else ""
            return text if text != "" else default
    return default


def _xml_get_child_text(parent, local_name, default=None):
    if parent is None:
        return default
    for elem in list(parent):
        if _xml_local_name(elem.tag) == local_name:
            text = elem.text.strip() if elem.text else ""
            return text if text != "" else default
    return default


def _xml_elem_text(elem, default=None):
    if elem is None or elem.text is None:
        return default
    text = elem.text.strip()
    return text if text != "" else default


def _philips_get_attr(elem, name, default=None):
    if elem is None:
        return default
    value = elem.attrib.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value != "" else default


def _philips_get_lead_name(leads_used, index):
    if leads_used in ["STD-12", "10-WIRE"]:
        if index == 1:
            return "I"
        if index == 2:
            return "II"
        if index == 3:
            return "III"
        if index == 4:
            return "aVR"
        if index == 5:
            return "aVL"
        if index == 6:
            return "aVF"
        if 7 <= index <= 12:
            return f"V{index - 6}"
    return f"Channel {index}"


def _philips_get_labels(signal_details, parsed_waveforms):
    lead_labels = _philips_get_attr(parsed_waveforms, "leadlabels", "")
    if lead_labels:
        lead_count = int(_philips_get_attr(parsed_waveforms, "numberofleads", "0") or "0")
        labels = [label for label in lead_labels.split() if label]
        return labels[:lead_count] if lead_count > 0 else labels

    good_channels = int(_xml_get_child_text(signal_details, "numberchannelsallocated", "0") or "0")
    leads_used = _xml_get_child_text(signal_details, "acquisitiontype", "") or ""
    return [_philips_get_lead_name(leads_used, idx + 1) for idx in range(good_channels)]


def _philips_split_leads(waveform_data, lead_count, sample_count):
    all_samples = np.frombuffer(waveform_data, dtype=np.int16)
    leads = []
    offset = 0
    end = lead_count * sample_count
    while offset < end:
        lead = all_samples[offset:offset + sample_count]
        leads.append(lead)
        offset += sample_count
    return leads


def _philips_xli_unpack(buffer):
    unpacked = np.empty(len(buffer) // 2, dtype=np.int16)
    half = len(unpacked)
    for idx in range(half):
        value = (buffer[idx] << 8) | buffer[half + idx]
        if value >= 0x8000:
            value -= 0x10000
        unpacked[idx] = np.int16(value)
    return unpacked


def _philips_xli_decode_deltas(buffer, first):
    deltas = _philips_xli_unpack(buffer)
    if len(deltas) < 3:
        return deltas

    x = int(deltas[0])
    y = int(deltas[1])
    last = first

    for idx in range(2, len(deltas)):
        z = (y + y) - x - last
        last = int(deltas[idx]) - 64
        deltas[idx] = np.int16(z)
        x = y
        y = z

    return deltas


def _philips_xli_decode(data):
    samples = []
    offset = 0
    while offset < len(data):
        header = data[offset:offset + 8]
        if len(header) < 8:
            break
        offset += 8

        size = int.from_bytes(header[0:4], byteorder="little", signed=True)
        start = int.from_bytes(header[6:8], byteorder="little", signed=True)
        if size < 0:
            raise ValueError("Philips XLI chunk size is invalid.")

        chunk = data[offset:offset + size]
        offset += size

        decoder = PhilipsLzwDecoder(chunk, bits=10)
        buffer = []
        while True:
            value = decoder.read()
            if value == -1:
                break
            buffer.append(value & 0xFF)

        if len(buffer) % 2 == 1:
            buffer.append(0)

        samples.append(_philips_xli_decode_deltas(buffer, start))

    return samples


def _philips_fix_derived_leads(lead_map):
    required = ["I", "II", "III", "aVR", "aVL", "aVF"]
    if any(lead not in lead_map for lead in required):
        return lead_map

    lead_i = lead_map["I"].astype(np.int32, copy=True)
    lead_ii = lead_map["II"].astype(np.int32, copy=True)
    lead_iii = lead_map["III"].astype(np.int32, copy=True)
    lead_avr = lead_map["aVR"].astype(np.int32, copy=True)
    lead_avl = lead_map["aVL"].astype(np.int32, copy=True)
    lead_avf = lead_map["aVF"].astype(np.int32, copy=True)

    lead_iii = lead_ii - lead_i - lead_iii
    lead_avr = -lead_avr - np.floor_divide(lead_i + lead_ii, 2)
    lead_avl = np.floor_divide(lead_i - lead_iii, 2) - lead_avl
    lead_avf = np.floor_divide(lead_ii + lead_iii, 2) - lead_avf

    lead_map["III"] = lead_iii.astype(np.int16)
    lead_map["aVR"] = lead_avr.astype(np.int16)
    lead_map["aVL"] = lead_avl.astype(np.int16)
    lead_map["aVF"] = lead_avf.astype(np.int16)
    return lead_map


def _philips_decode_waveforms(root):
    signal_details = _xml_find_first(root, "signalcharacteristics")
    parsed_waveforms = _xml_find_first(root, "parsedwaveforms")

    if signal_details is None or parsed_waveforms is None:
        raise ValueError("signalcharacteristics/parsedwaveforms not found in Philips XML.")

    sampling_freq = float(_xml_get_child_text(signal_details, "samplingrate", "500") or "500")
    duration_ms = int(_philips_get_attr(parsed_waveforms, "durationperchannel", "0") or "0")
    if duration_ms <= 0:
        raise ValueError("Philips XML durationperchannel is invalid.")

    sample_count = int(round(duration_ms * (sampling_freq / 1000.0)))
    labels = _philips_get_labels(signal_details, parsed_waveforms)
    if not labels:
        raise ValueError("Lead labels not found in Philips XML.")

    encoding = _philips_get_attr(parsed_waveforms, "dataencoding", "")
    payload = "".join((parsed_waveforms.text or "").split())
    if encoding != "Base64" or not payload:
        raise ValueError("Philips XML waveform dataencoding is unsupported or empty.")

    waveform_data = base64.b64decode(payload)
    compression = _philips_get_attr(
        parsed_waveforms,
        "compressmethod",
        _philips_get_attr(parsed_waveforms, "compression", "Uncompressed"),
    )

    if compression == "XLI":
        leads = _philips_xli_decode(waveform_data)
    elif compression == "Uncompressed":
        leads = _philips_split_leads(waveform_data, len(labels), sample_count)
    else:
        raise ValueError(f"Unsupported Philips waveform compression: {compression}")

    if len(leads) < len(labels):
        raise ValueError(
            f"Philips XML lead count is insufficient. expected={len(labels)}, actual={len(leads)}"
        )

    lead_map = {}
    for idx, label in enumerate(labels):
        lead_map[label] = np.asarray(leads[idx], dtype=np.int16)

    lead_map = _philips_fix_derived_leads(lead_map)

    resolution_text = _philips_get_attr(parsed_waveforms, "resolution", None)
    if resolution_text is None:
        resolution_text = _xml_get_child_text(signal_details, "resolution", None)
    if resolution_text is None:
        resolution_text = _xml_get_child_text(signal_details, "signalresolution", None)

    resolution_uv = float(resolution_text) if resolution_text not in [None, ""] else 1.0
    count_to_mV = resolution_uv / 1000.0

    return {
        "fs": sampling_freq,
        "duration_ms": duration_ms,
        "sample_count": sample_count,
        "labels": labels,
        "lead_map": lead_map,
        "resolution_uv": resolution_uv,
        "count_to_mV": count_to_mV,
        "compression": compression,
    }


def load_philips_xml(xml_path):
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()

    if _xml_local_name(root.tag) != "restingecgdata":
        raise ValueError("Philips XML root is not restingecgdata.")

    document_info = _xml_find_first(root, "documentinfo")
    doc_type = _xml_get_child_text(document_info, "documenttype", "")
    doc_version = _xml_get_child_text(document_info, "documentversion", "")
    if doc_type not in ["PhilipsECG", "SierraECG"]:
        raise ValueError(f"Unsupported Philips document type: {doc_type}")

    decoded = _philips_decode_waveforms(root)
    lead_map = decoded["lead_map"]
    expected_12 = STANDARD_12_ORDER.copy()
    missing = [lead for lead in expected_12 if lead not in lead_map]
    if missing:
        raise ValueError(f"Philips XML is missing required 12-lead signals: {missing}")

    lengths = [len(lead_map[lead]) for lead in expected_12]
    n = min(lengths)
    if n <= 0:
        raise ValueError("Philips XML waveform length is 0.")

    signal = np.column_stack([
        lead_map[lead][:n].astype(np.float64) * decoded["count_to_mV"]
        for lead in expected_12
    ])
    signal = remove_isolated_spikes(signal)

    report_info = _xml_find_first(root, "reportinfo")
    machine = _xml_find_first(root, "machine")
    patient_id = _xml_elem_text(_xml_find_first(root, "patientid"), None)
    acquisition_date = report_info.attrib.get("date") if report_info is not None else None
    acquisition_time = report_info.attrib.get("time") if report_info is not None else None

    fields = {
        "fs": decoded["fs"],
        "sig_name": expected_12,
        "units": ["mV"] * 12,
        "source": "philips_xml",
        "layout": "lead-major",
        "channel_count": 12,
        "sample_count": n,
        "scale_mode": "philips_resolution_uv",
        "count_to_mV": decoded["count_to_mV"],
        "resolution_uv": decoded["resolution_uv"],
        "xml_path": xml_path,
        "doc_type": doc_type,
        "doc_version": doc_version,
        "compression": decoded["compression"],
        "patient_id": patient_id,
        "device_name": machine.text.strip() if machine is not None and machine.text else None,
        "machine_id": machine.attrib.get("machineid") if machine is not None else None,
        "acquisition_date": acquisition_date,
        "acquisition_time": acquisition_time,
    }
    return signal, fields


def _parse_xml_any_encoding(xml_path):
    import io
    try:
        return ET.parse(xml_path)
    except ET.ParseError:
        pass
    with open(xml_path, 'rb') as f:
        raw = f.read()
    if raw.startswith(b'\xff\xfe'):
        text = raw[2:].decode('utf-16-le')
    elif raw.startswith(b'\xfe\xff'):
        text = raw[2:].decode('utf-16-be')
    elif raw.startswith(b'\xef\xbb\xbf'):
        text = raw[3:].decode('utf-8')
    elif raw.startswith(b'\x3c\x00') or raw.startswith(b'\x00\x00\x3c\x00'):
        text = raw.decode('utf-16-le')
    elif raw.startswith(b'\x00\x3c'):
        text = raw.decode('utf-16-be')
    else:
        text = raw.decode('utf-8', errors='replace')
    stripped = text.lstrip()
    if stripped.startswith('<?xml'):
        end = stripped.index('?>') + 2
        stripped = stripped[end:].lstrip()
    import re
    stripped = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', stripped)
    # Fix unescaped & not part of a valid entity or character reference
    stripped = re.sub(r'&(?!(?:#\d+|#x[\da-fA-F]+|[a-zA-Z]\w*);)', '&amp;', stripped)
    try:
        return ET.parse(io.BytesIO(stripped.encode('utf-8')))
    except ET.ParseError:
        # Last resort: strip declaration and retry with latin-1 fallback
        text2 = raw.decode('latin-1', errors='replace')
        text2 = re.sub(r'<\?xml[^?]*\?>', '', text2)
        text2 = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text2)
        text2 = re.sub(r'&(?!(?:#\d+|#x[\da-fA-F]+|[a-zA-Z]\w*);)', '&amp;', text2)
        return ET.parse(io.BytesIO(text2.encode('utf-8')))


def detect_xml_type(xml_path):
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()
    tag = root.tag

    if _xml_local_name(tag) == "HativECG":
        return "hativ"

    if _xml_local_name(tag) == "restingecgdata":
        document_info = _xml_find_first(root, "documentinfo")
        doc_type = _xml_get_child_text(document_info, "documenttype", "")
        if doc_type in ["PhilipsECG", "SierraECG"]:
            return "philips"

    if _xml_local_name(tag) == "ClinicalDocument":
        code_el = _xml_find_first(root, "code")
        if code_el is not None and code_el.attrib.get("codeSystemName") == "NK_MFER":
            return "nihonkohden"

    # Trismed / HL7 AnnotatedECG
    if "AnnotatedECG" in tag or "hl7-org:v3" in tag:
        return "trismed"

    # GE MAC2000
    if "sapphire" in tag or "urn:ge:sapphire" in tag:
        return "mac2000"

    # MUSE
    if root.find("Waveform") is not None:
        return "muse"

    # Bionet CardioXP
    if _xml_local_name(tag) == "CardioXP":
        return "bionet"

    return "unknown"

def load_muse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Rhythm 우선, 없으면 Median fallback
    waveforms = root.findall("Waveform")
    target_wf = None
    for wf in waveforms:
        if (wf.findtext("WaveformType") or "").strip().lower() == "rhythm":
            target_wf = wf
            break
    if target_wf is None:
        for wf in waveforms:
            if (wf.findtext("WaveformType") or "").strip().lower() == "median":
                target_wf = wf
                break
    if target_wf is None:
        raise ValueError("Rhythm/Median Waveform not found in MUSE XML.")

    fs = float(target_wf.findtext("SampleBase") or 500.0)

    lead_map = {}
    for lead_data in target_wf.findall("LeadData"):
        lead_id = (lead_data.findtext("LeadID") or "").strip()
        b64 = lead_data.findtext("WaveFormData") or ""
        b64 = "".join(b64.split())

        if not lead_id or not b64:
            continue

        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64)

        units_per_bit = float(lead_data.findtext("LeadAmplitudeUnitsPerBit") or 1.0)
        amp_units = (lead_data.findtext("LeadAmplitudeUnits") or "").strip().upper()
        baseline = float(lead_data.findtext("FirstSampleBaseline") or 0.0)

        # digital -> physical
        arr = (arr - baseline)

        if amp_units == "MICROVOLTS":
            arr = arr * units_per_bit / 1000.0   # uV -> mV
        elif amp_units == "MILLIVOLTS":
            arr = arr * units_per_bit
        else:
            # 단위가 애매하면 일단 uV 가정
            arr = arr * units_per_bit / 1000.0

        lead_map[lead_id.upper()] = arr

    base8_order = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    missing = [lead for lead in base8_order if lead not in lead_map]
    if missing:
        raise ValueError(f"MUSE XML is missing required 8 base leads: {missing}")

    lengths = [len(lead_map[lead]) for lead in base8_order]
    n = min(lengths)
    if n <= 0:
        raise ValueError("MUSE XML waveform length is 0.")

    base8 = np.column_stack([lead_map[lead][:n] for lead in base8_order])

    signal = derive_12_from_base8(base8)
    signal = remove_isolated_spikes(signal)

    fields = {
        "fs": fs,
        "sig_name": STANDARD_12_ORDER.copy(),
        "units": ["mV"] * 12,
        "source": "muse_xml",
        "layout": "lead-major",
        "channel_count": 12,
        "sample_count": n,
        "xml_path": xml_path,
        "waveform_type": target_wf.findtext("WaveformType"),
        "patient_id": root.findtext("./PatientDemographics/PatientID"),
        "acquisition_date": root.findtext("./TestDemographics/AcquisitionDate"),
        "acquisition_time": root.findtext("./TestDemographics/AcquisitionTime"),
    }
    return signal, fields

def load_mac2000_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"ge": "urn:ge:sapphire:dcar_1"}

    wfmxg = root.find(".//ge:wav/ge:ecgWaveformMXG", ns)
    if wfmxg is None:
        raise ValueError("ecgWaveformMXG not found in MAC2000 XML.")

    unit = (wfmxg.attrib.get("U") or "").strip().lower()
    scale = float(wfmxg.attrib.get("S", "1.0"))

    sr_elem = wfmxg.find("./ge:sampleRate", ns)
    fs = float(sr_elem.attrib.get("V", "500")) if sr_elem is not None else 500.0

    lead_map = {}
    for elem in wfmxg.findall("./ge:ecgWaveform", ns):
        lead = (elem.attrib.get("lead") or elem.attrib.get("label") or "").strip()
        raw_v = (elem.attrib.get("V") or "").strip()
        if not lead or not raw_v:
            continue

        vals = np.fromstring(raw_v, sep=" ", dtype=np.float64)
        if vals.size == 0:
            continue

        # counts -> mV
        if unit in ["uv", "µv", "microvolt", "microvolts"]:
            vals = vals * scale / 1000.0
            scale_mode = "xml_uv_scale"
        elif unit in ["mv", "millivolt", "millivolts"]:
            vals = vals * scale
            scale_mode = "xml_mv_scale"
        else:
            vals = vals * scale / 1000.0
            scale_mode = "xml_assume_uv_scale"

        lead_map[lead.upper()] = vals

    expected_12 = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    missing = [lead for lead in expected_12 if lead not in lead_map]
    if missing:
        raise ValueError(f"MAC2000 XML is missing required 12-lead signals: {missing}")

    lengths = [len(lead_map[lead]) for lead in expected_12]
    n = min(lengths)
    if n <= 0:
        raise ValueError("MAC2000 XML waveform length is 0.")

    signal = np.column_stack([lead_map[lead][:n] for lead in expected_12])
    signal = remove_isolated_spikes(signal)

    sig_name = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    device_elem = root.find(".//ge:device/ge:deviceName", ns)
    acq_elem = root.find(".//ge:testInfo/ge:acquisitionDateTime", ns)

    fields = {
        "fs": fs,
        "sig_name": sig_name,
        "units": ["mV"] * 12,
        "source": "mac2000_xml",
        "layout": "lead-major",
        "channel_count": 12,
        "sample_count": n,
        "scale_mode": scale_mode,
        "xml_path": xml_path,
        "xml_unit": unit,
        "xml_scale": scale,
        "device_name": device_elem.attrib.get("V") if device_elem is not None else "MAC2000",
        "acquisition_time": acq_elem.attrib.get("V") if acq_elem is not None else None,
    }
    return signal, fields

def load_trismed_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {
        "hl7": "urn:hl7-org:v3",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }

    # RHYTHM series 찾기
    rhythm_series = None
    for series in root.findall("./hl7:component/hl7:series", ns):
        code_elem = series.find("./hl7:code", ns)
        if code_elem is not None and (code_elem.attrib.get("code") or "").strip().upper() == "RHYTHM":
            rhythm_series = series
            break

    if rhythm_series is None:
        raise ValueError("RHYTHM series not found in Trismed XML.")

    # 샘플링 주파수 계산
    inc_elem = None
    seqs_for_time = rhythm_series.findall(
        "./hl7:component/hl7:sequenceSet/hl7:component/hl7:sequence",
        ns,
    )

    for seq in seqs_for_time:
        code_elem = seq.find("./hl7:code", ns)
        if code_elem is None:
            continue
        if (code_elem.attrib.get("code") or "").strip().upper() != "TIME_ABSOLUTE":
            continue

        value_elem = seq.find("./hl7:value", ns)
        if value_elem is None:
            break

        inc_elem = value_elem.find("./hl7:increment", ns)
        break

    if inc_elem is None:
        raise ValueError("TIME_ABSOLUTE/increment not found.")

    dt = float(inc_elem.attrib.get("value", "0.002"))
    if dt <= 0:
        raise ValueError(f"Invalid sample interval: {dt}")

    fs = 1.0 / dt

    # HL7 코드 -> 표시 리드명
    lead_code_map = {
        "MDC_ECG_LEAD_I": "I",
        "MDC_ECG_LEAD_II": "II",
        "MDC_ECG_LEAD_III": "III",
        "MDC_ECG_LEAD_AVR": "aVR",
        "MDC_ECG_LEAD_AVL": "aVL",
        "MDC_ECG_LEAD_AVF": "aVF",
        "MDC_ECG_LEAD_V1": "V1",
        "MDC_ECG_LEAD_V2": "V2",
        "MDC_ECG_LEAD_V3": "V3",
        "MDC_ECG_LEAD_V4": "V4",
        "MDC_ECG_LEAD_V5": "V5",
        "MDC_ECG_LEAD_V6": "V6",
        # 혹시 소문자 표기가 올 수도 있어서 보강
        "MDC_ECG_LEAD_aVR": "aVR",
        "MDC_ECG_LEAD_aVL": "aVL",
        "MDC_ECG_LEAD_aVF": "aVF",
    }

    lead_map = {}

    seqs = rhythm_series.findall("./hl7:component/hl7:sequenceSet/hl7:component/hl7:sequence", ns)
    for seq in seqs:
        code_elem = seq.find("./hl7:code", ns)
        value_elem = seq.find("./hl7:value", ns)

        if code_elem is None or value_elem is None:
            continue

        code = (code_elem.attrib.get("code") or "").strip()
        if code not in lead_code_map:
            continue

        origin_elem = value_elem.find("./hl7:origin", ns)
        scale_elem = value_elem.find("./hl7:scale", ns)
        digits_elem = value_elem.find("./hl7:digits", ns)

        if digits_elem is None or digits_elem.text is None:
            continue

        origin = float(origin_elem.attrib.get("value", "0")) if origin_elem is not None else 0.0
        scale = float(scale_elem.attrib.get("value", "1")) if scale_elem is not None else 1.0
        unit = (scale_elem.attrib.get("unit", "") if scale_elem is not None else "").strip().lower()

        digits = np.fromstring(digits_elem.text.strip(), sep=" ", dtype=np.float64)
        if digits.size == 0:
            continue

        # physical_uV = origin + digits * scale
        values_uv = origin + digits * scale

        if unit in ["uv", "µv", "microvolt", "microvolts", ""]:
            values_mv = values_uv / 1000.0
            scale_mode = "digits_scale_uv"
        elif unit in ["mv", "millivolt", "millivolts"]:
            values_mv = values_uv
            scale_mode = "digits_scale_mv"
        else:
            values_mv = values_uv / 1000.0
            scale_mode = "digits_scale_assume_uv"

        lead_name = lead_code_map[code]
        lead_map[lead_name] = values_mv

    expected_12 = STANDARD_12_ORDER.copy()
    missing = [lead for lead in expected_12 if lead not in lead_map]
    if missing:
        raise ValueError(f"Trismed XML is missing required 12-lead signals: {missing}")

    lengths = [len(lead_map[lead]) for lead in expected_12]
    n = min(lengths)
    if n <= 0:
        raise ValueError("Trismed XML waveform length is 0.")

    signal = np.column_stack([lead_map[lead][:n] for lead in expected_12])
    signal = remove_isolated_spikes(signal)

    acq_low = rhythm_series.find("./hl7:effectiveTime/hl7:low", ns)
    acq_high = rhythm_series.find("./hl7:effectiveTime/hl7:high", ns)
    model_elem = root.find(
        ".//hl7:manufacturerModelName",
        ns,
    )
    maker_elem = root.find(
        ".//hl7:playedManufacturedDevice/hl7:manufacturerOrganization/hl7:name",
        ns,
    )

    fields = {
        "fs": fs,
        "sig_name": expected_12,
        "units": ["mV"] * 12,
        "source": "trismed_xml",
        "layout": "lead-major",
        "channel_count": 12,
        "sample_count": n,
        "scale_mode": scale_mode,
        "xml_path": xml_path,
        "device_name": model_elem.text.strip() if model_elem is not None and model_elem.text else None,
        "manufacturer": maker_elem.text.strip() if maker_elem is not None and maker_elem.text else None,
        "acq_low": acq_low.attrib.get("value") if acq_low is not None else None,
        "acq_high": acq_high.attrib.get("value") if acq_high is not None else None,
    }
    return signal, fields

def load_bionet_xml(xml_path):
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()

    study_info = root.find("StudyInfo")
    wave_info = study_info.find("WaveInfo") if study_info is not None else None

    fs_elem = wave_info.find("SampleRate/Hz") if wave_info is not None else None
    fs = int(fs_elem.text.strip()) if fs_elem is not None and fs_elem.text else 500

    uv_elem = wave_info.find("DataUnit/uV") if wave_info is not None else None
    uv_per_count = float(uv_elem.text.strip()) if uv_elem is not None and uv_elem.text else 10.0

    channel_name_map = {
        "Lead I": "I", "Lead II": "II", "Lead III": "III",
        "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
        "V1": "V1", "V2": "V2", "V3": "V3",
        "V4": "V4", "V5": "V5", "V6": "V6",
    }

    lead_map = {}
    for wave_data in root.findall("StudyInfo/WaveData"):
        ch_elem = wave_data.find("Channel")
        data_elem = wave_data.find("Data")
        if ch_elem is None or data_elem is None or not data_elem.text:
            continue
        raw_name = ch_elem.text.strip() if ch_elem.text else ""
        lead_name = channel_name_map.get(raw_name, raw_name)
        counts = np.array([int(v) for v in data_elem.text.split()], dtype=np.float64)
        lead_map[lead_name] = counts / 1000.0  # Data values are pre-scaled µV; convert to mV

    missing = [lead for lead in STANDARD_12_ORDER if lead not in lead_map]
    if missing:
        raise ValueError(f"Bionet XML is missing required 12-lead signals: {missing}")

    lengths = [len(lead_map[lead]) for lead in STANDARD_12_ORDER]
    n = min(lengths)
    if n <= 0:
        raise ValueError("Bionet XML waveform length is 0.")

    signal = np.column_stack([lead_map[lead][:n] for lead in STANDARD_12_ORDER])
    signal = remove_isolated_spikes(signal)

    record_elem = root.find("StudyInfo/RecordInfo")
    acq_date = None
    acq_time = None
    if record_elem is not None:
        date_elem = record_elem.find("AcqDate")
        time_elem = record_elem.find("AcqTime")
        acq_date = date_elem.text.strip() if date_elem is not None and date_elem.text else None
        acq_time = time_elem.text.strip() if time_elem is not None and time_elem.text else None

    device_elem = root.find("StudyInfo/Device/Model")
    device_name = device_elem.text.strip() if device_elem is not None and device_elem.text else None

    fields = {
        "fs": fs,
        "sig_name": STANDARD_12_ORDER.copy(),
        "units": ["mV"] * 12,
        "source": "bionet_xml",
        "layout": "lead-major",
        "channel_count": 12,
        "sample_count": n,
        "uv_per_count": uv_per_count,
        "xml_path": xml_path,
        "device_name": device_name,
        "acquisition_date": acq_date,
        "acquisition_time": acq_time,
    }
    return signal, fields

# Huffman table recovered from fhdecode.dll .data section (RVA 0x7034).
# State 13 is root; states 0..12 are leaves. Each entry: (child_if_bit0, child_if_bit1).
# Leaf state → delta: states 0..10 → delta = state-5 (-5..+5),
#   state 11 → 8-bit escape, state 12 → 16-bit escape, state -1 → terminator.
_FUKUDA_HUFF_TABLE = {
    13: (15, 14), 14: (5, 4),  15: (17, 16),
    16: (6, 3),   17: (18, 7), 18: (20, 19),
    19: (2, 8),   20: (21, 11),21: (22, 1),
    22: (23, 9),  23: (24, 0), 24: (25, 10),
    25: (-1, 12),
}


def _decode_fukuda_waveform(compressed, lead_size, lead_count=8):
    """Huffman + 2nd-order delta decoder (pure Python port of fhdecode.dll)."""
    n_words = len(compressed) // 2
    words = np.frombuffer(compressed[:n_words * 2], dtype=np.dtype(">u2"))
    total = lead_size * lead_count
    out = np.zeros(total, dtype=np.int16)

    word_idx = 0
    bit_pos = 15
    prev = prev_prev = 0
    n = 0

    def read_bit():
        nonlocal word_idx, bit_pos
        if word_idx >= n_words:
            raise EOFError
        b = (int(words[word_idx]) >> bit_pos) & 1
        bit_pos -= 1
        if bit_pos < 0:
            bit_pos = 15
            word_idx += 1
        return b

    def read_bits(k):
        v = 0
        for _ in range(k):
            v = (v << 1) | read_bit()
        return v

    while n < total:
        state = 13
        try:
            while state >= 13:
                b = read_bit()
                left, right = _FUKUDA_HUFF_TABLE[state]
                state = right if b else left
        except EOFError:
            break
        if state < 0:
            break

        if state <= 10:
            delta = state - 5
        elif state == 11:
            byte = read_bits(8)
            delta = byte - 256 if byte >= 128 else byte
        else:
            word = read_bits(16)
            delta = word - 65536 if word >= 32768 else word

        sample = delta + 2 * prev - prev_prev
        sample = ((sample + 0x8000) & 0xFFFF) - 0x8000  # wrap to int16
        out[n] = sample
        n += 1
        prev_prev = prev
        prev = sample

    return out.reshape(lead_count, lead_size)


def load_fukuda_ecg(ecg_path):
    """Fukuda Denshi proprietary .ecg format (pure Python, no DLL required).

    File layout: 112-byte header (32-byte prefix + 5×16-byte UnitInfo table),
    then 5 sequential unit blobs: Info / Patient / Measurement / Diagno / History.
    The HistoryUnit contains the Huffman+2nd-order-delta compressed waveform.
    """
    import struct

    with open(ecg_path, "rb") as f:
        data = f.read()

    # File header: u32 file_unit_id, u32 file_unit_size, u16 item_count,
    #              20-byte nickname, u16 unit_count  (= 32 bytes)
    # Followed by 5 × UnitInfo (u32 unit_id, u32 size, u32 compress_type, u32 uncompress_size)
    # Total header = 32 + 5×16 = 112 bytes (= file_unit_size).
    file_unit_size = struct.unpack_from(">I", data, 4)[0]

    unit_sizes = [
        struct.unpack_from(">I", data, 32 + i * 16 + 4)[0]
        for i in range(5)
    ]

    # Absolute file offsets for each unit blob
    unit_offsets = [file_unit_size]
    for sz in unit_sizes[:-1]:
        unit_offsets.append(unit_offsets[-1] + sz)

    # HistoryUnit is the 5th unit (index 4)
    hist_off = unit_offsets[4]
    hist_blob = data[hist_off: hist_off + unit_sizes[4]]

    # Parse HistoryUnit fixed prefix (big-endian throughout)
    # Offset 0..9: internal unit header (skipped)
    # Offset 10: sample_rate i16 (stored as 2000; forced to 500 per device spec)
    # Offset 12: lsb_size i16   Offset 14: lead_size i32
    # Offset 18: 2 unknown bytes
    # Offset 20: lead_count i16   Offset 22: lead_id_count i16 (unused)
    # Offset 24: lead_ids (lead_count × i16)
    # Offset 24+2*lead_count: data_size i32, then compressed bytes
    p = 10
    lsb_size  = struct.unpack_from(">h", hist_blob, p + 2)[0]
    lead_size = struct.unpack_from(">i", hist_blob, p + 4)[0]
    lead_count = struct.unpack_from(">h", hist_blob, p + 10)[0]
    p = 24
    p += lead_count * 2  # skip lead_ids
    data_size  = struct.unpack_from(">i", hist_blob, p)[0]
    p += 4
    compressed = bytes(hist_blob[p: p + data_size])

    # Decode compressed waveform → (8, lead_size) int16
    int16_8 = _decode_fukuda_waveform(compressed, lead_size, lead_count)

    # Derive 4 dependent leads using integer arithmetic (matches C# reference)
    i32  = int16_8[0].astype(np.int32)
    ii32 = int16_8[1].astype(np.int32)
    iii = (ii32 - i32).astype(np.int16)
    avr = (-(ii32 + i32) // 2).astype(np.int16)
    avl = ((i32 - (ii32 - i32)) // 2).astype(np.int16)
    avf = ((ii32 + (ii32 - i32)) // 2).astype(np.int16)

    # Assemble (lead_size, 12) float32 mV — standard 12-lead order
    wf12 = np.stack([
        int16_8[0], int16_8[1], iii, avr, avl, avf,
        int16_8[2], int16_8[3], int16_8[4], int16_8[5], int16_8[6], int16_8[7],
    ], axis=-1)
    signal = wf12.astype(np.float32) * float(lsb_size) * 1e-6

    fields = {
        "fs": 500.0,
        "sig_name": list(STANDARD_12_ORDER),
        "units": ["mV"] * 12,
        "n_sig": 12,
        "source": "fukuda_ecg",
        "ecg_path": ecg_path,
    }
    return signal, fields


def load_fukuda_measurements(ecg_path):
    """Read clinical measurement values from MeasurementUnit of a Fukuda .ecg file.

    Offsets verified against 2023051810233700001.ecg. Values stored as BE int16.
    Time fields in ms; voltage fields in 10uV units (abs for SV1/RV6 -- sign
    reflects wave polarity, clinical display uses magnitude).
    """
    with open(ecg_path, "rb") as f:
        data = f.read()

    file_unit_size = struct.unpack_from(">I", data, 4)[0]
    unit_sizes = [struct.unpack_from(">I", data, 32 + i * 16 + 4)[0] for i in range(5)]
    unit_offsets = [file_unit_size]
    for sz in unit_sizes[:-1]:
        unit_offsets.append(unit_offsets[-1] + sz)

    blob = data[unit_offsets[2]: unit_offsets[2] + unit_sizes[2]]

    def rd(off):
        return struct.unpack_from(">h", blob, off)[0]

    sv1_mv = abs(rd(0x0206)) * 10 / 1000.0
    rv6_mv = abs(rd(0x0356)) * 10 / 1000.0

    return {
        "hr":   rd(0x0034),
        "rr":   rd(0x0036) / 1000.0,
        "pr":   rd(0x0038) / 1000.0,
        "qrs":  rd(0x003a) / 1000.0,
        "qt":   rd(0x003c) / 1000.0,
        "qtc":  rd(0x003e) / 1000.0,
        "axis": rd(0x0042),
        "sv1":  sv1_mv,
        "rv6":  rv6_mv,
        "rs":   sv1_mv + rv6_mv,
    }


def load_mac2000_measurements(xml_path):
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()
    INV = -32768

    def _find_text(tag):
        el = _xml_find_first(root, tag)
        return el.text.strip() if el is not None and el.text else None

    def _int(tag):
        v = _find_text(tag)
        try:
            iv = int(v)
            return None if iv == INV else iv
        except (TypeError, ValueError):
            return None

    rr_ms = _int("aveRRInterval")
    hr = round(60000 / rr_ms) if rr_ms else _int("VentricularRate")

    return {
        "hr":      hr,
        "pr":      _int("PRInterval"),
        "qrs":     _int("QRSDuration"),
        "qt":      _int("QTInterval"),
        "qtc":     _int("QTCorrected"),
        "p_axis":  _int("PFrontAxis"),
        "r_axis":  _int("QRSFrontAxis"),
        "t_axis":  _int("TFrontAxis"),
    }


def load_muse_measurements(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _int(tag):
        el = root.find(f".//{tag}")
        if el is None or not (el.text or "").strip():
            return None
        try:
            return int(el.text.strip())
        except ValueError:
            return None

    return {
        "hr":     _int("VentricularRate"),
        "pr":     _int("PRInterval"),
        "qrs":    _int("QRSDuration"),
        "qt":     _int("QTInterval"),
        "qtc":    _int("QTCorrected"),
        "p_axis": _int("PAxis"),
        "r_axis": _int("RAxis"),
        "t_axis": _int("TAxis"),
    }


def load_bionet_measurements(xml_path):
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()

    def _int(tag):
        el = _xml_find_first(root, tag)
        if el is None or not (el.text or "").strip():
            return None
        try:
            return int(el.text.strip())
        except ValueError:
            return None

    return {
        "hr":     _int("HeartRate"),
        "pr":     _int("MeanPRint"),
        "qrs":    _int("MeanQRSdur"),
        "qt":     _int("MeanQTint"),
        "qtc":    _int("MeanQTc"),
        "p_axis": _int("Paxis"),
        "r_axis": None,
        "t_axis": _int("Taxis"),
    }


def load_philips_measurements(xml_path):
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()

    def _int_child(parent, tag):
        el = _xml_find_first(parent, tag) if parent is not None else None
        if el is None or not (el.text or "").strip():
            return None
        try:
            return int(el.text.strip())
        except ValueError:
            return None

    gm = None
    interp = _xml_find_first(root, "interpretation")
    if interp is not None:
        gm = _xml_find_first(interp, "globalmeasurements")
    if gm is not None and _xml_find_first(gm, "heartrate") is not None:
        # TC70 layout
        hr  = _int_child(gm, "heartrate")
        pr  = _int_child(gm, "print")
        qrs = _int_child(gm, "qrsdur")
        qt  = _int_child(gm, "qtint")
        qtc = _int_child(gm, "qtcb")
    else:
        # Trim3 layout
        meas = _xml_find_first(root, "measurements")
        gm2  = _xml_find_first(meas, "globalmeasurements") if meas is not None else None
        hr  = _int_child(gm2, "meanventrate")
        pr  = _int_child(gm2, "meanprint")
        qrs = _int_child(gm2, "meanqrsdur")
        qt  = _int_child(gm2, "meanqtint")
        qtc = _int_child(gm2, "meanqtc")
        if gm is None:
            gm = gm2

    return {
        "hr":     hr,
        "pr":     pr,
        "qrs":    qrs,
        "qt":     qt,
        "qtc":    qtc,
        "p_axis": _int_child(gm, "pfrontaxis"),
        "r_axis": _int_child(gm, "qrsfrontaxis"),
        "t_axis": _int_child(gm, "tfrontaxis"),
    }


def load_trismed_measurements(xml_path):
    import re as _re
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = "urn:hl7-org:v3"

    mdc_map = {}
    for ann in root.iter():
        tag = ann.tag.split("}")[1] if "}" in ann.tag else ann.tag
        if tag != "annotation":
            continue
        code_el = ann.find(f"{{{ns}}}code")
        if code_el is None:
            continue
        code = code_el.attrib.get("code", "")
        val_el = ann.find(f".//{{{ns}}}value")
        if val_el is None:
            continue
        v = val_el.attrib.get("value", "")
        if v:
            try:
                mdc_map[code] = int(v)
            except ValueError:
                pass

    texts = []
    for elem in root.iter():
        tag = elem.tag.split("}")[1] if "}" in elem.tag else elem.tag
        if tag != "value":
            continue
        xsi_type = elem.attrib.get("{http://www.w3.org/2001/XMLSchema-instance}type", "")
        if xsi_type == "ST" and (elem.text or "").strip():
            texts.append(elem.text.strip())

    def parse_axis(pattern, texts, invalid=999.0):
        for line in texts:
            m = _re.search(pattern, line)
            if m:
                val = float(m.group(1))
                return None if val >= invalid else round(val, 2)
        return None

    p_axis     = parse_axis(r'\bP\s+([\d.]+)\s+degree', texts)
    qrs_axis   = parse_axis(r'\bQRS\s+([\d.]+)\s+degree', texts)
    t_axis     = parse_axis(r'\bT\s+([\d.]+)\s+degree', texts)
    qrs_t_axis = parse_axis(r'\bQRS-T\s+([\d.]+)\s+degree', texts)

    qtr = None
    for line in texts:
        m = _re.search(r'\bQTr[:\s]+([\d.]+)', line)
        if m:
            try:
                qtr = float(m.group(1))
            except ValueError:
                pass
            break

    return {
        "hr":     mdc_map.get("MDC_ECG_HEART_RATE"),
        "pr":     mdc_map.get("MDC_ECG_TIME_PD_PR"),
        "qrs":    mdc_map.get("MDC_ECG_TIME_PD_QRS"),
        "qt":     mdc_map.get("MDC_ECG_TIME_PD_QT"),
        "qtc":    mdc_map.get("MDC_ECG_TIME_PD_QTc"),
        "qtr":    qtr,
        "p_axis": p_axis,
        "r_axis": qrs_axis,
        "t_axis": t_axis,
        "qrs_t":  qrs_t_axis,
    }


# LOINC code → measurement key mapping for Nihon Kohden NK_MFER XML
_NK_LOINC_MAP = {
    "9873-1":  ("hr",     None),
    "8625-6":  ("pr",     None),
    "18517-3": ("qrs",    None),
    "8634-8":  ("qt",     None),
    "8636-3":  ("qtc",    None),
    "8626-4":  ("p_axis", None),
    "8632-2":  ("r_axis", None),
    "8638-9":  ("t_axis", None),
    "10040-4": ("sv1",    "mV"),
    "9995-2":  ("rv5",    "mV"),
}


def load_nihonkohden_measurements(xml_path):
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()
    ns = "urn:hl7-org:v3"

    result = {}
    for obs in root.iter():
        tag = obs.tag.split("}")[1] if "}" in obs.tag else obs.tag
        if tag != "observation":
            continue
        code_el = obs.find(f"{{{ns}}}code")
        val_el  = obs.find(f"{{{ns}}}value")
        if code_el is None or val_el is None:
            continue
        loinc = code_el.attrib.get("code", "")
        if loinc not in _NK_LOINC_MAP:
            continue
        key, unit_hint = _NK_LOINC_MAP[loinc]
        raw_v = val_el.attrib.get("value", "")
        if not raw_v:
            continue
        try:
            result[key] = float(raw_v) if unit_hint == "mV" else int(raw_v)
        except ValueError:
            pass

    rv5 = result.get("rv5")
    sv1 = result.get("sv1")
    if rv5 is not None and sv1 is not None:
        result["rs"] = round(rv5 + sv1, 3)

    return result


def _read_hea_measurements(hea_path):
    """Return the ecg_metrics dict embedded in a WFDB .hea comment, or None."""
    try:
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith(_METRICS_PREFIX):
                    return json.loads(line.strip()[len(_METRICS_PREFIX):])
    except Exception:
        pass
    return None


_ECG_WFDB_GAIN = 1000.0   # adu/mV
_MUSE_MEASUREMENT_TAGS = {
    "hr": "VentricularRate", "pr": "PRInterval", "qrs": "QRSDuration",
    "qt": "QTInterval", "qtc": "QTCorrected",
    "p_axis": "PAxis", "r_axis": "RAxis", "t_axis": "TAxis",
}


def export_csv(signal, fields, out_path, measurements=None):
    """Export ECG signal to CSV. Embeds measurements as a # ecg_metrics comment when provided."""
    sig_names = fields["sig_name"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if measurements:
            f.write(f"{_METRICS_PREFIX}{json.dumps(measurements)}\n")
        writer = csv.writer(f)
        writer.writerow(sig_names)
        for row in signal:
            writer.writerow([f"{v:.6f}" for v in row])


def export_wfdb(signal, fields, out_path, measurements=None):
    """Export ECG signal to WFDB (.dat + .hea). Embeds measurements as a .hea comment."""
    base = os.path.splitext(out_path)[0]
    record_name = os.path.basename(base)
    write_dir = os.path.dirname(out_path) or "."
    fs = int(fields["fs"])
    sig_name = [str(n) for n in fields["sig_name"]]
    n_leads = signal.shape[1]
    comments = [f"{_METRICS_PREFIX[2:]}{json.dumps(measurements)}"] if measurements else None
    wfdb.wrsamp(
        record_name=record_name, fs=fs,
        units=["mV"] * n_leads, sig_name=sig_name,
        p_signal=signal.astype(np.float64),
        fmt=["16"] * n_leads,
        adc_gain=[_ECG_WFDB_GAIN] * n_leads,
        baseline=[0] * n_leads,
        write_dir=write_dir,
        comments=comments,
    )


def export_mat_hea(signal, fields, out_path, measurements=None):
    """Export ECG signal to MATLAB (.mat) + WFDB header (.hea). Embeds measurements in .hea."""
    from scipy.io import savemat
    base = os.path.splitext(out_path)[0]
    record_name = os.path.basename(base)
    mat_filename = os.path.basename(out_path)
    fs = fields["fs"]
    sig_name = [str(n) for n in fields["sig_name"]]
    n_sig = len(sig_name)
    n_samples = signal.shape[0]
    val = np.round(signal.T * _ECG_WFDB_GAIN).astype(np.int16)
    savemat(out_path, {"val": val})
    hea_path = base + ".hea"
    with open(hea_path, "w", encoding="utf-8") as f:
        f.write(f"{record_name} {n_sig} {int(fs)} {n_samples}\n")
        if measurements:
            f.write(f"{_METRICS_PREFIX}{json.dumps(measurements)}\n")
        for name in sig_name:
            f.write(f"{mat_filename} 16+24 {int(_ECG_WFDB_GAIN)}/mV 16 0 0 0 0 {name}\n")


def export_muse_xml(signal, fields, out_path, measurements=None):
    """Export ECG signal to GE MUSE XML. Embeds measurements in <RestingECGMeasurements>."""
    from datetime import datetime
    fs = fields["fs"]
    sig_name = [str(n).upper() for n in fields["sig_name"]]
    now = datetime.now()
    root = ET.Element("RestingECG")
    patient_elem = ET.SubElement(root, "PatientDemographics")
    ET.SubElement(patient_elem, "PatientID").text = fields.get("patient_id") or ""
    test_elem = ET.SubElement(root, "TestDemographics")
    ET.SubElement(test_elem, "AcquisitionDate").text = (
        fields.get("acquisition_date") or now.strftime("%m-%d-%Y")
    )
    ET.SubElement(test_elem, "AcquisitionTime").text = (
        fields.get("acquisition_time") or now.strftime("%H:%M:%S")
    )
    waveform = ET.SubElement(root, "Waveform")
    ET.SubElement(waveform, "WaveformType").text = "Rhythm"
    ET.SubElement(waveform, "SampleBase").text = str(int(fs))
    for i, name in enumerate(sig_name):
        if i >= signal.shape[1]:
            break
        lead_data = ET.SubElement(waveform, "LeadData")
        ET.SubElement(lead_data, "LeadID").text = name
        samples = np.round(signal[:, i] * 1000.0).astype(np.int16)
        b64_str = base64.b64encode(samples.astype("<i2").tobytes()).decode("ascii")
        ET.SubElement(lead_data, "WaveFormData").text = b64_str
        ET.SubElement(lead_data, "LeadAmplitudeUnitsPerBit").text = "1"
        ET.SubElement(lead_data, "LeadAmplitudeUnits").text = "MICROVOLTS"
        ET.SubElement(lead_data, "FirstSampleBaseline").text = "0"
    if measurements:
        meas_elem = ET.SubElement(root, "RestingECGMeasurements")
        for key, tag in _MUSE_MEASUREMENT_TAGS.items():
            value = measurements.get(key)
            if value is not None:
                ET.SubElement(meas_elem, tag).text = str(value)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)


def load_waveform(base_path):
    p = Path(base_path)

    if p.suffix.lower() == ".ecg" and p.exists():
        return load_fukuda_ecg(str(p))

    if p.suffix.lower() == ".xml" and p.exists():
        xml_type = detect_xml_type(str(p))

        if xml_type == "hativ":
            return load_hativ_xml(str(p))
        elif xml_type == "philips":
            return load_philips_xml(str(p))
        elif xml_type == "trismed":
            return load_trismed_xml(str(p))
        elif xml_type == "mac2000":
            return load_mac2000_xml(str(p))
        elif xml_type == "muse":
            return load_muse_xml(str(p))
        elif xml_type == "bionet":
            return load_bionet_xml(str(p))
        else:
            raise ValueError("Unsupported XML format.")

    hea_path = find_companion_file(base_path, [".hea"])
    dat_path = find_companion_file(base_path, [".dat"])
    mwf_path = find_companion_file(base_path, [".mwf"])
    mat_path = find_companion_file(base_path, [".mat"])
    csv_path = find_companion_file(base_path, [".csv"])

    if csv_path:
        return load_csv_waveform(csv_path)
    if hea_path and dat_path:
        signal, fields = wfdb.rdsamp(str(Path(hea_path).with_suffix("")))
        measurements = _read_hea_measurements(hea_path)
        if measurements:
            fields["measurements"] = measurements
        return signal, fields
    if hea_path and mat_path:
        return load_mat_hea(base_path)
    if mwf_path:
        return load_dat_mwf(dat_path or "", mwf_path)

    raise FileNotFoundError(
        "No supported file combination found.\n"
        "Supported formats:\n"
        "- .xml (standalone)\n"
        "- .ecg (standalone, Fukuda Denshi)\n"
        "- .dat + .hea\n"
        "- .mat + .hea\n"
        "- .dat + .mwf\n"
        "- .mwf (standalone)\n"
        "- .csv (standalone)"
    )

class LeadZoomDialog(QDialog):
    def __init__(self, signal, fs, lead_name, lead_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Lead Zoom - {lead_name}")
        self.resize(ZOOM_DIALOG_WIDTH, ZOOM_DIALOG_HEIGHT)

        self.signal = signal
        self.fs = fs
        self.lead_name = lead_name
        self.lead_index = lead_index

        self.current_duration_sec = 10.0
        self.y = None
        self.ax = None
        self.total_duration_sec = 0.0

        self.crosshair_v = None
        self.crosshair_h = None
        self.hover_annot = None
        self.measure_annot = None

        self.measure_point1 = None
        self.measure_point2 = None
        self.measure_marker1 = None
        self.measure_marker2 = None
        self.measure_line = None

        self.initial_xlim = None
        self.initial_ylim = None

        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_xlim = None
        self.pan_start_ylim = None

        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Display Duration (s):"))

        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["2.5", "5", "10"])
        self.duration_combo.setCurrentText("10")
        self.duration_combo.currentTextChanged.connect(self.on_duration_changed)
        top_bar.addWidget(self.duration_combo)

        self.save_button = QPushButton("Save PNG")
        self.save_button.clicked.connect(self.save_png)
        top_bar.addWidget(self.save_button)

        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("hh:mm:ss")
        self.time_input.setFixedWidth(120)
        self.time_input.returnPressed.connect(self.move_to_input_time)
        top_bar.addWidget(self.time_input)

        self.move_button = QPushButton("Move")
        self.move_button.clicked.connect(self.move_to_input_time)
        top_bar.addWidget(self.move_button)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.figure = Figure(figsize=(12, 4.8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("axes_leave_event", self.on_axes_leave)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self.on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.redraw()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.full_reset()
            event.accept()
            return
        super().keyPressEvent(event)

    def on_duration_changed(self):
        self.full_reset()
        self.redraw()

    def redraw(self):
        self.current_duration_sec = float(self.duration_combo.currentText())
        self.y = self.signal[:, self.lead_index]
        self.total_duration_sec = (len(self.y) / self.fs) if len(self.y) > 0 else 0.0

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(PAPER_FACE_COLOR)
        if len(self.y) > 0:
            self.ax.plot(np.arange(len(self.y)) / self.fs, self.y, linewidth=1.0)

        self.ax.set_title(self.lead_name, fontsize=16)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude (mV)")

        initial_x1 = min(self.current_duration_sec, self.total_duration_sec)
        if initial_x1 <= 0:
            initial_x1 = self.current_duration_sec
        self.ax.set_xlim(0, initial_x1)
        self.ax.set_ylim(-ZOOM_Y_LIM, ZOOM_Y_LIM)

        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()

        self.ax.xaxis.set_major_locator(MultipleLocator(0.2))
        self.ax.xaxis.set_minor_locator(MultipleLocator(SEC_PER_MM))
        self.ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR_STEP))
        self.ax.yaxis.set_minor_locator(MultipleLocator(Y_MINOR_STEP))

        self.ax.grid(
            True,
            which="major",
            linestyle=GRID_MAJOR_LINESTYLE,
            linewidth=GRID_MAJOR_WIDTH,
            color=GRID_COLOR,
        )

        self.ax.grid(
            True,
            which="minor",
            linestyle=GRID_MINOR_LINESTYLE,
            linewidth=GRID_MINOR_WIDTH,
            color=GRID_COLOR,
            alpha=GRID_MINOR_ALPHA,
        )

        self.crosshair_v = self.ax.axvline(0, linewidth=0.8, alpha=0.5, visible=False)
        self.crosshair_h = self.ax.axhline(0, linewidth=0.8, alpha=0.5, visible=False)

        self.hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.90),
            fontsize=9,
        )
        self.hover_annot.set_visible(False)

        self.measure_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, -40),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#fff4cc", alpha=0.95),
            fontsize=9,
        )
        self.measure_annot.set_visible(False)

        self.measure_marker1 = None
        self.measure_marker2 = None
        self.measure_line = None

        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_xlim = None
        self.pan_start_ylim = None

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_png(self):
        default_name = f"{self.lead_name}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PNG",
            default_name,
            "PNG Files (*.png);;All Files (*)",
        )
        if file_path:
            self.figure.savefig(file_path, dpi=200, bbox_inches="tight")

    def parse_relative_time_input(self):
        text = self.time_input.text().strip()
        if not text:
            return 0

        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError("Invalid time format")

        values = []
        for part in parts:
            part = part.strip()
            if not part:
                values.append(0)
                continue
            if not part.isdigit():
                raise ValueError("Invalid time format")
            values.append(int(part))

        while len(values) < 3:
            values.insert(0, 0)

        hours, minutes, seconds = values
        return hours * 3600 + minutes * 60 + seconds

    def move_to_input_time(self):
        try:
            target_sec = float(self.parse_relative_time_input())
        except ValueError:
            QMessageBox.warning(self, "Invalid time", "Invalid time format")
            return

        self.move_to_time(target_sec)

    def move_to_time(self, target_sec):
        if self.ax is None or len(self.y) == 0:
            QMessageBox.information(self, "No data exists", "No data exists")
            return

        if target_sec > self.total_duration_sec:
            QMessageBox.information(self, "No data exists", "No data exists")
            return

        if target_sec <= 0:
            if self.initial_xlim is not None:
                self.ax.set_xlim(self.initial_xlim)
                self.canvas.draw_idle()
            return

        cur_xlim = self.ax.get_xlim()
        width = cur_xlim[1] - cur_xlim[0]
        half_width = width / 2.0

        new_x0 = target_sec - half_width
        new_x1 = target_sec + half_width
        new_x0, new_x1 = self.clamp_xlim(new_x0, new_x1)

        self.ax.set_xlim(new_x0, new_x1)
        self.canvas.draw_idle()

    def nearest_sample(self, xdata):
        idx = int(np.clip(round(xdata * self.fs), 0, len(self.y) - 1))
        t = idx / self.fs
        amp = float(self.y[idx])
        return idx, t, amp

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.hide_hover_overlay()
            return

        if self.is_panning and self.pan_start_mouse is not None:
            dx = event.xdata - self.pan_start_mouse[0]
            dy = event.ydata - self.pan_start_mouse[1]

            x0, x1 = self.pan_start_xlim
            y0, y1 = self.pan_start_ylim

            new_x0 = x0 - dx
            new_x1 = x1 - dx
            new_x0, new_x1 = self.clamp_xlim(new_x0, new_x1)

            self.ax.set_xlim(new_x0, new_x1)
            self.ax.set_ylim(y0 - dy, y1 - dy)
            self.canvas.draw_idle()
            return

        idx, t, amp = self.nearest_sample(event.xdata)

        self.crosshair_v.set_visible(True)
        self.crosshair_h.set_visible(True)
        self.crosshair_v.set_xdata([t, t])
        self.crosshair_h.set_ydata([amp, amp])

        self.hover_annot.xy = (t, amp)
        self.hover_annot.set_text(
            f"idx: {idx}\n"
            f"t: {t:.3f} s\n"
            f"amp: {amp:.3f} mV"
        )
        self.hover_annot.set_visible(True)

        self.canvas.draw_idle()

    def on_axes_leave(self, event):
        if not self.is_panning:
            self.hide_hover_overlay()

    def hide_hover_overlay(self):
        if self.crosshair_v is not None:
            self.crosshair_v.set_visible(False)
        if self.crosshair_h is not None:
            self.crosshair_h.set_visible(False)
        if self.hover_annot is not None:
            self.hover_annot.set_visible(False)
        self.canvas.draw_idle()

    def on_canvas_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 3:
            self.is_panning = True
            self.pan_start_mouse = (event.xdata, event.ydata)
            self.pan_start_xlim = self.ax.get_xlim()
            self.pan_start_ylim = self.ax.get_ylim()
            return

        if event.dblclick:
            return

        if event.button != 1:
            return

        idx, t, amp = self.nearest_sample(event.xdata)

        if self.measure_point1 is None or self.measure_point2 is not None:
            self.set_measure_point1(idx, t, amp)
        else:
            self.set_measure_point2(idx, t, amp)

    def on_canvas_release(self, event):
        if event.button == 3:
            self.is_panning = False
            self.pan_start_mouse = None
            self.pan_start_xlim = None
            self.pan_start_ylim = None

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        x_center = event.xdata
        x0, x1 = cur_xlim
        x_range = x1 - x0

        if x_range <= 0:
            return

        scale = 0.8 if event.button == "up" else 1.25

        new_range = x_range * scale
        total_min = 0.0
        total_max = self.total_duration_sec

        left_ratio = (x_center - x0) / x_range
        right_ratio = 1.0 - left_ratio

        new_x0 = x_center - new_range * left_ratio
        new_x1 = x_center + new_range * right_ratio

        min_range = max(5.0 / self.fs, SEC_PER_MM)
        max_range = max(total_max - total_min, self.current_duration_sec)

        if new_range < min_range:
            new_range = min_range
            new_x0 = x_center - new_range * left_ratio
            new_x1 = x_center + new_range * right_ratio

        if new_range > max_range:
            new_range = max_range
            new_x0 = total_min
            new_x1 = total_max

        if new_x0 < total_min:
            shift = total_min - new_x0
            new_x0 += shift
            new_x1 += shift
        if new_x1 > total_max:
            shift = new_x1 - total_max
            new_x0 -= shift
            new_x1 -= shift

        new_x0, new_x1 = self.clamp_xlim(new_x0, new_x1)

        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(cur_ylim)
        self.canvas.draw_idle()

    def clamp_xlim(self, x0, x1):
        if len(self.y) == 0:
            return x0, x1

        width = x1 - x0
        total_min = 0.0
        total_max = self.total_duration_sec

        if total_max <= total_min:
            return 0.0, max(self.current_duration_sec, SEC_PER_MM)

        if width >= (total_max - total_min):
            return total_min, total_max

        if x0 < total_min:
            x1 += total_min - x0
            x0 = total_min
        if x1 > total_max:
            x0 -= x1 - total_max
            x1 = total_max

        return x0, x1

    def set_measure_point1(self, idx, t, amp):
        self.clear_measurement_artists_only()

        self.measure_point1 = {"idx": idx, "t": t, "amp": amp}
        self.measure_point2 = None

        self.measure_marker1, = self.ax.plot(
            [t], [amp],
            marker="o",
            markersize=6,
            linestyle="None",
        )

        self.measure_annot.set_visible(False)
        self.canvas.draw_idle()

    def set_measure_point2(self, idx, t, amp):
        self.measure_point2 = {"idx": idx, "t": t, "amp": amp}

        t1 = self.measure_point1["t"]
        a1 = self.measure_point1["amp"]
        t2 = self.measure_point2["t"]
        a2 = self.measure_point2["amp"]

        if self.measure_marker2 is not None:
            self.measure_marker2.remove()
            self.measure_marker2 = None
        if self.measure_line is not None:
            self.measure_line.remove()
            self.measure_line = None

        self.measure_marker2, = self.ax.plot(
            [t2], [a2],
            marker="o",
            markersize=6,
            linestyle="None",
        )

        self.measure_line, = self.ax.plot(
            [t1, t2], [a1, a2],
            linewidth=1.0,
            linestyle="--",
        )

        dt_ms = (t2 - t1) * 1000.0
        da_mv = a2 - a1

        if dt_ms > 0:
            bpm = round(60000.0 / dt_ms)
            bpm_text = f" ({bpm} bpm)"
        else:
            bpm_text = ""

        self.measure_annot.xy = (t2, a2)
        self.measure_annot.set_text(
            f"Δt: {dt_ms:.1f} ms{bpm_text}\n"
            f"Δamp: {da_mv:.3f} mV"
        )
        self.measure_annot.set_visible(True)

        self.canvas.draw_idle()

    def clear_measurement_artists_only(self):
        if self.measure_marker1 is not None:
            self.measure_marker1.remove()
            self.measure_marker1 = None
        if self.measure_marker2 is not None:
            self.measure_marker2.remove()
            self.measure_marker2 = None
        if self.measure_line is not None:
            self.measure_line.remove()
            self.measure_line = None

    def reset_measurement(self):
        self.measure_point1 = None
        self.measure_point2 = None
        self.clear_measurement_artists_only()
        if self.measure_annot is not None:
            self.measure_annot.set_visible(False)
        self.canvas.draw_idle()

    def reset_view(self):
        if self.ax is None:
            return
        if self.initial_xlim is not None:
            self.ax.set_xlim(self.initial_xlim)
        if self.initial_ylim is not None:
            self.ax.set_ylim(self.initial_ylim)

        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_xlim = None
        self.pan_start_ylim = None

    def full_reset(self):
        self.reset_measurement()
        self.reset_view()
        self.hide_hover_overlay()
        self.canvas.draw_idle()

class DropCanvas(FigureCanvas):
    def __init__(self, figure, drop_callback):
        super().__init__(figure)
        self.setAcceptDrops(True)
        self.drop_callback = drop_callback

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if Path(file_path).suffix.lower() in [".dat", ".hea", ".mwf", ".mat", ".csv", ".xml", ".ecg"]:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if Path(file_path).suffix.lower() in [".dat", ".hea", ".mwf", ".mat", ".csv", ".xml", ".ecg"]:
                base_path = normalize_input_to_base_path(file_path)
                self.drop_callback(base_path)
                return


class ECGViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Viewer")
        self.current_path = None
        self.current_signal = None
        self.current_fields = None
        self.ax_lead_map = {}
        self.zoom_dialogs = []

        self.setGeometry(100, 100, 1350, 1380) # 전체 창 크기

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.load_button = QPushButton("Open ECG File (.dat / .hea / .mat / .mwf / .csv / .xml / .ecg)", self)
        self.load_button.clicked.connect(self.load_ecg)
        self.layout.addWidget(self.load_button)

        metrics_row = QWidget()
        metrics_row.setFixedHeight(52)
        metrics_row_layout = QHBoxLayout(metrics_row)
        metrics_row_layout.setContentsMargins(0, 0, 0, 0)
        metrics_row_layout.setSpacing(0)

        self.metrics_label = QLabel("")
        self.metrics_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.metrics_label.setStyleSheet(
            "background-color: #F0F0F0;"
            "color: #222222;"
            "font-family: monospace;"
            "font-size: 14px;"
            "font-weight: bold;"
            "padding: 3px 8px;"
        )
        metrics_row_layout.addWidget(self.metrics_label, stretch=1)

        converter_widget = QWidget()
        converter_widget.setObjectName("converterWidget")
        converter_widget.setStyleSheet(
            "QWidget#converterWidget { background-color: #F0F0F0; }"
        )
        converter_widget.setFixedWidth(170)
        converter_layout = QVBoxLayout(converter_widget)
        converter_layout.setContentsMargins(4, 2, 4, 2)
        converter_layout.setSpacing(2)

        self.convert_format_combo = QComboBox()
        self.convert_format_combo.addItems([
            "CSV (.csv)",
            "WFDB (.dat + .hea)",
            "MAT (.mat + .hea)",
            "MUSE XML (.xml)",
        ])
        self.convert_format_combo.setEnabled(False)
        converter_layout.addWidget(self.convert_format_combo)

        self.convert_button = QPushButton("Convert")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self._on_convert)
        converter_layout.addWidget(self.convert_button)

        metrics_row_layout.addWidget(converter_widget)
        self.layout.addWidget(metrics_row)

        self.stack_layout = QStackedLayout()
        self.layout.addLayout(self.stack_layout)

        self.figure = Figure(figsize=(20, 16)) #각 유도 박스 크기
        self.canvas = DropCanvas(self.figure, self.load_and_plot)
        self.stack_layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        self.spinner_widget = QWidget()
        self.spinner_layout = QVBoxLayout(self.spinner_widget)
        self.spinner_layout.setAlignment(Qt.AlignCenter)

        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_movie = QMovie("spinner.gif")

        if self.spinner_movie.isValid():
            self.spinner_movie.setScaledSize(QSize(64, 64))
            self.spinner_label.setMovie(self.spinner_movie)
        else:
            self.spinner_label.setText("Loading...")
            self.spinner_label.setStyleSheet("font-size: 18px; color: gray;")

        self.spinner_layout.addWidget(self.spinner_label)
        self.stack_layout.addWidget(self.spinner_widget)

        self.canvas_text = self.figure.text(
            0.5, 0.5,
            "Drag and Drop ECG File Here",
            ha="center", va="center",
            fontsize=20, alpha=0.3
        )

    def show_error(self, message):
        QMessageBox.critical(self, "File Load Error", message)

    def _update_metrics(self, fields):
        source = fields.get("source")

        def _fmt_ms(v):
            return f"{v} ms" if v is not None else "--"

        def _fmt_bpm(v):
            return f"{v} bpm" if v is not None else "--"

        def _fmt_deg(v):
            return f"{v}" if v is not None else "--"

        def _fmt_mv(v):
            return f"{v:.3f} mV" if v is not None else "--"

        try:
            if source == "fukuda_ecg":
                m = load_fukuda_measurements(fields["ecg_path"])
                line1 = (
                    f"HR: {m['hr']} bpm    "
                    f"R-R: {m['rr']:.3f} s    "
                    f"P-R: {m['pr']:.3f} s    "
                    f"QRS: {m['qrs']:.3f} s    "
                    f"QT: {m['qt']:.3f} s    "
                    f"QTc: {m['qtc']:.3f}    "
                    f"AXIS: {m['axis']} deg"
                )
                line2 = (
                    f"SV1: {m['sv1']:.2f} mV    "
                    f"RV6: {m['rv6']:.2f} mV    "
                    f"R+S: {m['rs']:.2f} mV"
                )
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source == "mac2000_xml":
                m = load_mac2000_measurements(fields["xml_path"])
                hr  = _fmt_bpm(m.get("hr"))
                pr  = _fmt_ms(m.get("pr"))
                qrs = _fmt_ms(m.get("qrs"))
                qt  = m.get("qt"); qtc = m.get("qtc")
                qt_str = f"{qt}/{qtc} ms" if qt is not None and qtc is not None else "--"
                pa  = _fmt_deg(m.get("p_axis"))
                ra  = _fmt_deg(m.get("r_axis"))
                ta  = _fmt_deg(m.get("t_axis"))
                line1 = f"Vent. rate: {hr}    PR interval: {pr}    QRS duration: {qrs}    QT/QTc: {qt_str}"
                line2 = f"P-R-T axes: {pa}  {ra}  {ta}"
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source == "muse_xml":
                m = load_muse_measurements(fields["xml_path"])
                hr  = _fmt_bpm(m.get("hr"))
                pr  = _fmt_ms(m.get("pr"))
                qrs = _fmt_ms(m.get("qrs"))
                qt  = m.get("qt"); qtc = m.get("qtc")
                qt_str = f"{qt}/{qtc} ms" if qt is not None and qtc is not None else "--"
                pa  = _fmt_deg(m.get("p_axis"))
                ra  = _fmt_deg(m.get("r_axis"))
                ta  = _fmt_deg(m.get("t_axis"))
                line1 = f"Vent. rate: {hr}    PR interval: {pr}    QRS duration: {qrs}    QT/QTc: {qt_str}"
                line2 = f"P-R-T axes: {pa}  {ra}  {ta}"
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source == "bionet_xml":
                m = load_bionet_measurements(fields["xml_path"])
                hr  = _fmt_bpm(m.get("hr"))
                pr  = _fmt_ms(m.get("pr"))
                qrs = _fmt_ms(m.get("qrs"))
                qt  = m.get("qt"); qtc = m.get("qtc")
                qt_str = f"{qt}/{qtc} ms" if qt is not None and qtc is not None else "--"
                pa  = _fmt_deg(m.get("p_axis"))
                ra  = _fmt_deg(m.get("r_axis"))
                ta  = _fmt_deg(m.get("t_axis"))
                line1 = f"Vent. rate: {hr}    PR interval: {pr}    QRS duration: {qrs}    QT/QTc: {qt_str}"
                line2 = f"P-R-T axes: {pa}  {ra}  {ta}"
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source == "philips_xml":
                m = load_philips_measurements(fields["xml_path"])
                hr  = _fmt_bpm(m.get("hr"))
                pr  = _fmt_ms(m.get("pr"))
                qrs = _fmt_ms(m.get("qrs"))
                qt  = m.get("qt"); qtc = m.get("qtc")
                qt_str = f"{qt}/{qtc} ms" if qt is not None and qtc is not None else "--"
                pa  = _fmt_deg(m.get("p_axis"))
                ra  = _fmt_deg(m.get("r_axis"))
                ta  = _fmt_deg(m.get("t_axis"))
                line1 = f"Vent. rate: {hr}    PR interval: {pr}    QRS duration: {qrs}    QT/QTc: {qt_str}"
                line2 = f"P-R-T axes: {pa}  {ra}  {ta}"
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source == "trismed_xml":
                m = load_trismed_measurements(fields["xml_path"])
                hr  = _fmt_bpm(m.get("hr"))
                pr  = _fmt_ms(m.get("pr"))
                qrs = _fmt_ms(m.get("qrs"))
                qt  = m.get("qt"); qtc = m.get("qtc")
                qtr = m.get("qtr")
                qt_str = f"{qt}/{qtc} ms" if qt is not None and qtc is not None else "--"
                qtr_str = f"    QTr: {qtr}" if qtr is not None else ""
                pa  = _fmt_deg(m.get("p_axis"))
                ra  = _fmt_deg(m.get("r_axis"))
                ta  = _fmt_deg(m.get("t_axis"))
                qt_t = _fmt_deg(m.get("qrs_t"))
                line1 = f"Vent. rate: {hr}    PR interval: {pr}    QRS duration: {qrs}    QT/QTc: {qt_str}{qtr_str}"
                line2 = f"P/QRS/T/QRS-T axis: {pa}/{ra}/{ta}/{qt_t} deg"
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source == "mwf":
                companion = fields.get("companion_xml_path")
                if not companion or detect_xml_type(companion) != "nihonkohden":
                    self.metrics_label.setText("")
                    return
                m = load_nihonkohden_measurements(companion)
                hr  = _fmt_bpm(m.get("hr"))
                pr  = _fmt_ms(m.get("pr"))
                qrs = _fmt_ms(m.get("qrs"))
                qt  = m.get("qt"); qtc = m.get("qtc")
                qt_str = f"{qt}/{qtc} ms" if qt is not None and qtc is not None else "--"
                pa  = _fmt_deg(m.get("p_axis"))
                ra  = _fmt_deg(m.get("r_axis"))
                ta  = _fmt_deg(m.get("t_axis"))
                rv5 = m.get("rv5"); sv1 = m.get("sv1"); rs = m.get("rs")
                rv5_str = _fmt_mv(rv5)
                sv1_str = _fmt_mv(sv1)
                rs_str  = _fmt_mv(rs)
                line1 = f"Heart Rate: {hr}    PR Int: {pr}    QRS Int: {qrs}    QT/QTc: {qt_str}"
                line2 = f"P/QRS/T Axis: {pa}/{ra}/{ta} deg    RV5/SV1: {rv5_str}/{sv1_str}    RV5+SV1: {rs_str}"
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif fields.get("measurements"):
                m = fields["measurements"]
                hr  = f"{m['hr']} BPM"     if m.get("hr")     is not None else "?"
                pr  = f"{m['pr']} ms"      if m.get("pr")     is not None else "?"
                qrs = f"{m['qrs']} ms"     if m.get("qrs")    is not None else "?"
                qt  = f"{m['qt']} ms"      if m.get("qt")     is not None else "?"
                qtc = f"{m['qtc']} ms"     if m.get("qtc")    is not None else "?"
                pa  = f"{m['p_axis']} deg" if m.get("p_axis") is not None else "?"
                ra  = f"{m['r_axis']} deg" if m.get("r_axis") is not None else "?"
                ta  = f"{m['t_axis']} deg" if m.get("t_axis") is not None else "?"
                line1 = f"HR: {hr}    PR: {pr}    QRS: {qrs}    QT/QTc: {qt}/{qtc}"
                line2 = f"P-R-T axes: {pa}  {ra}  {ta}"
                self.metrics_label.setText(f"{line1}\n{line2}")

            else:
                self.metrics_label.setText("")

        except Exception:
            self.metrics_label.setText("")

    def load_ecg(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ECG File",
            "",
            "ECG Files (*.dat *.DAT *.hea *.HEA *.mat *.MAT *.mwf *.MWF *.csv *.CSV *.xml *.XML *.ecg *.ECG);;All Files (*)",
            options=options,
        )

        if file_path:
            base_path = normalize_input_to_base_path(file_path)
            self.load_and_plot(base_path)

    def load_and_plot(self, base_path):
        self.spinner_movie.start()
        self.stack_layout.setCurrentWidget(self.spinner_widget)
        QTimer.singleShot(50, lambda: self._load_and_plot(base_path))

    def _load_and_plot(self, base_path):
        self.current_path = base_path
        filename = os.path.basename(base_path)
        self.setWindowTitle(f"ECG Viewer - {filename}")

        try:
            signal, fields = load_waveform(base_path)
            fs = float(fields["fs"])

            signal = remove_baseline_wander(signal, fs, cutoff=0.5, order=2)
            signal = safe_bandpass_filter(signal, 0.5, 150.0, fs)

            self.current_signal = signal
            self.current_fields = fields

            self.plot_ecg(signal, fields)
            self._update_metrics(fields)
            self.convert_format_combo.setEnabled(True)
            self.convert_button.setEnabled(True)

        except Exception as e:
            self.show_error(str(e))
        finally:
            self.spinner_movie.stop()
            self.stack_layout.setCurrentWidget(self.canvas)

    def _collect_measurements(self):
        """Read clinical measurements from the currently loaded source file.

        Returns a normalized dict {hr, pr, qrs, qt, qtc, p_axis, r_axis, t_axis}
        (intervals in ms, axes in degrees) or None when unavailable.
        """
        fields = self.current_fields
        if not fields:
            return None
        source = fields.get("source")
        try:
            if source == "fukuda_ecg":
                m = load_fukuda_measurements(fields["ecg_path"])
                return {
                    "hr":     m.get("hr"),
                    "pr":     int(round(m["pr"]  * 1000)) if m.get("pr")  is not None else None,
                    "qrs":    int(round(m["qrs"] * 1000)) if m.get("qrs") is not None else None,
                    "qt":     int(round(m["qt"]  * 1000)) if m.get("qt")  is not None else None,
                    "qtc":    int(round(m["qtc"] * 1000)) if m.get("qtc") is not None else None,
                    "p_axis": None,
                    "r_axis": m.get("axis"),
                    "t_axis": None,
                }
            if source in ("mac2000_xml", "muse_xml", "bionet_xml", "philips_xml", "trismed_xml"):
                if source == "mac2000_xml":
                    m = load_mac2000_measurements(fields["xml_path"])
                elif source == "muse_xml":
                    m = load_muse_measurements(fields["xml_path"])
                elif source == "bionet_xml":
                    m = load_bionet_measurements(fields["xml_path"])
                elif source == "philips_xml":
                    m = load_philips_measurements(fields["xml_path"])
                else:
                    m = load_trismed_measurements(fields["xml_path"])
                return {
                    "hr": m.get("hr"), "pr": m.get("pr"), "qrs": m.get("qrs"),
                    "qt": m.get("qt"), "qtc": m.get("qtc"),
                    "p_axis": m.get("p_axis"), "r_axis": m.get("r_axis"), "t_axis": m.get("t_axis"),
                }
            if source == "mwf":
                companion = fields.get("companion_xml_path")
                if not companion or detect_xml_type(companion) != "nihonkohden":
                    return None
                m = load_nihonkohden_measurements(companion)
                return {
                    "hr": m.get("hr"), "pr": m.get("pr"), "qrs": m.get("qrs"),
                    "qt": m.get("qt"), "qtc": m.get("qtc"),
                    "p_axis": m.get("p_axis"), "r_axis": m.get("r_axis"), "t_axis": m.get("t_axis"),
                }
        except Exception:
            pass
        return None

    def _on_convert(self):
        """Open a save dialog and export the current ECG to the selected format."""
        if self.current_signal is None or self.current_fields is None:
            return

        fmt = self.convert_format_combo.currentText()

        if "CSV" in fmt:
            file_filter = "CSV Files (*.csv)"
            default_ext = ".csv"
        elif "WFDB" in fmt:
            file_filter = "WFDB Header (*.hea)"
            default_ext = ".hea"
        elif "MAT" in fmt:
            file_filter = "MATLAB File (*.mat)"
            default_ext = ".mat"
        elif "MUSE" in fmt:
            file_filter = "MUSE XML (*.xml)"
            default_ext = ".xml"
        else:
            return

        suggested = ""
        if self.current_path:
            suggested = os.path.splitext(self.current_path)[0] + default_ext

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Converted ECG", suggested, file_filter
        )
        if not out_path:
            return

        measurements = self._collect_measurements()

        try:
            if "CSV" in fmt:
                export_csv(self.current_signal, self.current_fields, out_path, measurements)
            elif "WFDB" in fmt:
                export_wfdb(self.current_signal, self.current_fields, out_path, measurements)
            elif "MAT" in fmt:
                export_mat_hea(self.current_signal, self.current_fields, out_path, measurements)
            elif "MUSE" in fmt:
                export_muse_xml(self.current_signal, self.current_fields, out_path, measurements)
            QMessageBox.information(self, "Convert", f"Saved to:\n{out_path}")
        except Exception as exc:
            self.show_error(f"Convert failed: {exc}")

    def on_plot_click(self, event):
        if event.inaxes is None:
            return

        if not event.dblclick:
            return

        lead = self.ax_lead_map.get(event.inaxes)
        if not lead:
            return

        if self.current_signal is None or self.current_fields is None:
            return

        lead_indices = {
            str(name).upper(): idx
            for idx, name in enumerate(self.current_fields["sig_name"])
        }

        lead_index = lead_indices.get(lead.upper())
        if lead_index is None:
            return

        dialog = LeadZoomDialog(
            self.current_signal,
            float(self.current_fields["fs"]),
            lead,
            lead_index,
            self
        )
        self.zoom_dialogs.append(dialog)
        dialog.finished.connect(lambda _, d=dialog: self._on_zoom_dialog_closed(d))
        dialog.show()

    @staticmethod
    def _apply_ecg_grid(ax, x_max):
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("Amplitude (mV)", fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=6)

        ax.set_ylim([-MAIN_Y_LIM, MAIN_Y_LIM])
        ax.set_xlim([0, x_max])

        ax.set_xticks(np.arange(0, x_max + 1e-9, 0.2))
        ax.set_xticks(np.arange(0, x_max + 1e-9, SEC_PER_MM), minor=True)
        ax.set_yticks(np.arange(-MAIN_Y_LIM, MAIN_Y_LIM + 1e-9, Y_MAJOR_STEP))
        ax.set_yticks(np.arange(-MAIN_Y_LIM, MAIN_Y_LIM + 1e-9, MV_PER_MM), minor=True)

        ax.grid(True, which="major", linestyle=GRID_MAJOR_LINESTYLE,
                linewidth=GRID_MAJOR_WIDTH, color=GRID_COLOR)
        ax.grid(True, which="minor", linestyle=GRID_MINOR_LINESTYLE,
                linewidth=GRID_MINOR_WIDTH, color=GRID_COLOR, alpha=GRID_MINOR_ALPHA)

    def plot_ecg(self, signal, fields):
        self.figure.clear()
        self.ax_lead_map = {}

        fs = float(fields["fs"])
        total_samples = signal.shape[0]

        short_duration = min(int(fs * 2.5), total_samples)
        long_duration = min(int(fs * 10.0), total_samples)

        time_short = np.arange(short_duration) / fs if short_duration > 0 else np.array([])
        time_long = np.arange(long_duration) / fs if long_duration > 0 else np.array([])

        layout_order = [
            "I", "aVR", "V1", "V4",
            "II", "aVL", "V2", "V5",
            "III", "aVF", "V3", "V6",
        ]

        lead_indices = {str(name).upper(): idx for idx, name in enumerate(fields["sig_name"])}

        gs = GridSpec(
            4, 4,
            figure=self.figure,
            height_ratios=[1.0, 1.0, 1.0, 1.0],
            hspace=0.30,
            wspace=0.08
        )

        for i, lead in enumerate(layout_order):
            if lead.upper() not in lead_indices:
                continue

            row = i // 4
            col = i % 4

            idx = lead_indices[lead.upper()]
            ax = self.figure.add_subplot(gs[row, col])
            self.ax_lead_map[ax] = lead
            ax.set_facecolor(PAPER_FACE_COLOR)

            ecg = signal[:short_duration, idx]
            ax.plot(time_short[:len(ecg)], ecg, linewidth=0.8)

            ax.set_title(lead, fontsize=16, pad=2)
            ax.set_box_aspect(0.96)
            self._apply_ecg_grid(ax, 2.5)

        if "II" in lead_indices and long_duration > 0:
            ax = self.figure.add_subplot(gs[3, :])
            self.ax_lead_map[ax] = "II"
            ax.set_facecolor(PAPER_FACE_COLOR)

            idx = lead_indices["II"]
            ecg = signal[:long_duration, idx]
            ax.plot(time_long[:len(ecg)], ecg, linewidth=0.8)

            display_long_sec = 10.0

            ax.set_title("II (repeat)", fontsize=16)
            ax.set_box_aspect(0.24)
            self._apply_ecg_grid(ax, display_long_sec)

        info_text = (
            f"source: {fields.get('source', '?')}\n"
            f"layout: {fields.get('layout', '?')}\n"
            f"fs: {fields.get('fs', '?')}\n"
            f"channels: {fields.get('channel_count', '?')}\n"
            f"samples: {fields.get('sample_count', '?')}\n"
            f"scale: {fields.get('scale_mode', '-')}"
        )
        self.figure.text(
            0.99, 0.99,
            info_text,
            ha="right", va="top",
            fontsize=8, alpha=0.7
        )

        # 바깥 여백
        self.figure.subplots_adjust(
            left=0.025,
            right=0.992,
            top=0.97,
            bottom=0.045,
            wspace=0.08,
            hspace=0.30,
        )
        self.canvas.draw_idle()

    def _on_zoom_dialog_closed(self, dialog):
        try:
            self.zoom_dialogs.remove(dialog)
        except ValueError:
            pass

    def closeEvent(self, event):
        try:
            self.spinner_movie.stop()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "vuno.ecgviewer.desktop.v1"
        )

    app = QApplication(sys.argv)
    icon = QIcon(resource_path("ecg.ico"))
    app.setWindowIcon(icon)
    viewer = ECGViewer()
    viewer.setWindowIcon(icon)
    viewer.show()
    sys.exit(app.exec_())


