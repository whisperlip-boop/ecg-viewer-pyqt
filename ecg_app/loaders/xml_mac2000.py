import xml.etree.ElementTree as ET

import numpy as np

from ..constants import STANDARD_12_ORDER
from ..signal_processing import remove_isolated_spikes


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
    scale_mode = "xml_assume_uv_scale"
    for elem in wfmxg.findall("./ge:ecgWaveform", ns):
        lead = (elem.attrib.get("lead") or elem.attrib.get("label") or "").strip()
        raw_v = (elem.attrib.get("V") or "").strip()
        if not lead or not raw_v:
            continue

        vals = np.fromstring(raw_v, sep=" ", dtype=np.float64)
        if vals.size == 0:
            continue

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


def load_mac2000_measurements(xml_path):
    """Read clinical measurements from MAC2000 XML <measurements><global> section."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"ge": "urn:ge:sapphire:dcar_1"}

    INV = -32768

    def get(tag):
        el = root.find(f".//ge:measurements/ge:global/ge:{tag}", ns)
        if el is None:
            return None
        v = el.attrib.get("V")
        if v is None:
            return None
        try:
            val = int(v)
            return None if val == INV else val
        except ValueError:
            return None

    rr_ms = get("aveRRInterval")
    hr = round(60000 / rr_ms) if rr_ms else None

    return {
        "hr":   hr,
        "pr":   get("PR_Interval"),
        "qrs":  get("QRS_Duration"),
        "qt":   get("QT_Interval"),
        "qtc":  get("QT_Corrected"),
        "p_axis": get("P_Axis"),
        "r_axis": get("R_Axis"),
        "t_axis": get("T_Axis"),
    }
