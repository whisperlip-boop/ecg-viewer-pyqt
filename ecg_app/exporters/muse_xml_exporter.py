"""GE MUSE XML exporter for ECG signal data."""

import base64
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import numpy as np


_MUSE_MEASUREMENT_TAGS: dict[str, str] = {
    "hr":     "VentricularRate",
    "pr":     "PRInterval",
    "qrs":    "QRSDuration",
    "qt":     "QTInterval",
    "qtc":    "QTCorrected",
    "p_axis": "PAxis",
    "r_axis": "RAxis",
    "t_axis": "TAxis",
}


def export_muse_xml(
    signal: np.ndarray,
    fields: dict[str, Any],
    out_path: str,
    measurements: dict[str, Any] | None = None,
) -> None:
    """Export ECG signal to GE MUSE XML format.

    Encodes each lead as base64 int16 little-endian at 1 µV/bit resolution.
    All available leads are written; the MUSE loader will re-derive
    III/aVR/aVL/aVF from the 8 base leads (I, II, V1–V6) on re-import.
    When *measurements* is provided a ``<RestingECGMeasurements>`` section
    is appended so that clinical metrics survive the round-trip.

    Args:
        signal: Signal array of shape (samples, leads) in mV.
        fields: Metadata dict; must contain 'fs' and 'sig_name'.
                Optional keys used when present: 'patient_id',
                'acquisition_date', 'acquisition_time'.
        out_path: Output .xml file path.
        measurements: Optional dict with keys hr, pr, qrs, qt, qtc,
            p_axis, r_axis, t_axis (int or None, values in BPM/ms/deg).

    Returns:
        None

    Raises:
        OSError: If the file cannot be written.
    """
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

        # mV → µV as int16 little-endian, then base64-encode
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
