import base64
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import STANDARD_12_ORDER
from ..signal_processing import derive_12_from_base8, remove_isolated_spikes


def load_muse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    waveforms = root.findall("Waveform")
    target_wf = None
    for waveform in waveforms:
        if (waveform.findtext("WaveformType") or "").strip().lower() == "rhythm":
            target_wf = waveform
            break
    if target_wf is None:
        for waveform in waveforms:
            if (waveform.findtext("WaveformType") or "").strip().lower() == "median":
                target_wf = waveform
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

        arr = arr - baseline

        if amp_units == "MICROVOLTS":
            arr = arr * units_per_bit / 1000.0
        elif amp_units == "MILLIVOLTS":
            arr = arr * units_per_bit
        else:
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


def load_muse_measurements(xml_path):
    """Read clinical measurements from MUSE XML <RestingECGMeasurements> section."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    m = root.find("RestingECGMeasurements")
    if m is None:
        return {}

    def get(tag):
        el = m.find(tag)
        if el is None or not (el.text or "").strip():
            return None
        try:
            return int(el.text.strip())
        except ValueError:
            return None

    return {
        "hr":     get("VentricularRate"),
        "pr":     get("PRInterval"),
        "qrs":    get("QRSDuration"),
        "qt":     get("QTInterval"),
        "qtc":    get("QTCorrected"),
        "p_axis": get("PAxis"),
        "r_axis": get("RAxis"),
        "t_axis": get("TAxis"),
    }
