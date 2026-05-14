import base64
import xml.etree.ElementTree as ET

import numpy as np

from ..signal_processing import remove_isolated_spikes
from .xml_common import _xml_local_name, _xml_find_first, _xml_get_child_text


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
