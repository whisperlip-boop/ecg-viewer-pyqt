import numpy as np

from ..constants import STANDARD_12_ORDER
from ..signal_processing import remove_isolated_spikes
from .xml_common import _parse_xml_any_encoding


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


def load_bionet_measurements(xml_path):
    """Read clinical measurements from Bionet XML <ShortMeasurementSegment>.
    Note: R axis is not stored in this XML format and will be None.
    """
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()
    seg = root.find("StudyInfo/ShortMeasurementSegment")

    def get(tag):
        if seg is None:
            return None
        el = seg.find(tag)
        if el is None or not (el.text or "").strip():
            return None
        try:
            return int(el.text.strip())
        except ValueError:
            return None

    return {
        "hr":     get("HeartRate"),
        "pr":     get("MeanPRint"),
        "qrs":    get("MeanQRSdur"),
        "qt":     get("MeanQTint"),
        "qtc":    get("MeanQTc"),
        "p_axis": get("Paxis"),
        "r_axis": None,
        "t_axis": get("Taxis"),
    }
