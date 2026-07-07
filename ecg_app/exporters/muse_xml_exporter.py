"""GE MUSE XML exporter for ECG signal data."""

import base64
import zlib
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

# Element order copied from a real GE MUSE export (restecg.dtd is positional,
# not name-based, so missing/reordered tags make third-party parsers fail
# outright rather than just losing optional data).
_MEASUREMENT_FIELD_ORDER: list[str] = [
    "VentricularRate", "AtrialRate", "PRInterval", "QRSDuration",
    "QTInterval", "QTCorrected", "PAxis", "RAxis", "TAxis",
    "QRSCount", "QOnset", "QOffset", "POnset", "POffset", "TOffset",
]

_BASE8_LEADS: list[str] = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def _add_measurements_block(
    parent: ET.Element,
    tag: str,
    measurements: dict[str, Any] | None,
    fs: float,
    include_frederica: bool,
) -> None:
    """Append a RestingECGMeasurements-style block.

    Every field GE MUSE normally carries is emitted (empty when unknown) so
    that positional/DTD-order parsers don't choke on a truncated sequence.

    Args:
        parent: Parent XML element to attach the block to.
        tag: Element name, e.g. "RestingECGMeasurements".
        measurements: Optional dict with keys hr, pr, qrs, qt, qtc, p_axis,
            r_axis, t_axis.
        fs: Sampling rate in Hz.
        include_frederica: Whether to append the trailing QTcFrederica tag
            (present in RestingECGMeasurements but not in the Original copy).

    Returns:
        None
    """
    block = ET.SubElement(parent, tag)
    measurements = measurements or {}
    reverse_map = {v: k for k, v in _MUSE_MEASUREMENT_TAGS.items()}
    for field in _MEASUREMENT_FIELD_ORDER:
        key = reverse_map.get(field)
        value = measurements.get(key) if key else None
        ET.SubElement(block, field).text = "" if value is None else str(value)
    ET.SubElement(block, "ECGSampleBase").text = str(int(fs))
    ET.SubElement(block, "ECGSampleExponent").text = "0"
    if include_frederica:
        ET.SubElement(block, "QTcFrederica").text = ""


def export_muse_xml(
    signal: np.ndarray,
    fields: dict[str, Any],
    out_path: str,
    measurements: dict[str, Any] | None = None,
) -> None:
    """Export ECG signal to GE MUSE XML format.

    Mirrors the element order and tag set of a real GE MUSE export
    (DOCTYPE, MuseInfo, PatientDemographics, TestDemographics,
    RestingECGMeasurements pair, single Rhythm Waveform) so that
    downstream MUSE-compatible readers can parse the file. Only the
    8 independently-acquired leads (I, II, V1-V6) are written, matching
    how real MUSE files store data and how this app's own MUSE loader
    re-derives III/aVR/aVL/aVF on import.

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

    lead_indices: list[tuple[str, int]] = [
        (lead, sig_name.index(lead))
        for lead in _BASE8_LEADS
        if lead in sig_name and sig_name.index(lead) < signal.shape[1]
    ]

    root = ET.Element("RestingECG")

    muse_info = ET.SubElement(root, "MuseInfo")
    ET.SubElement(muse_info, "MuseVersion").text = ""

    patient_elem = ET.SubElement(root, "PatientDemographics")
    ET.SubElement(patient_elem, "PatientID").text = fields.get("patient_id") or ""
    ET.SubElement(patient_elem, "Race").text = ""
    ET.SubElement(patient_elem, "PatientLastName").text = ""

    test_elem = ET.SubElement(root, "TestDemographics")
    ET.SubElement(test_elem, "DataType").text = "RESTING"
    ET.SubElement(test_elem, "Site").text = ""
    ET.SubElement(test_elem, "SiteName").text = ""
    ET.SubElement(test_elem, "AcquisitionDevice").text = ""
    ET.SubElement(test_elem, "Status").text = ""
    ET.SubElement(test_elem, "EditListStatus").text = ""
    ET.SubElement(test_elem, "Priority").text = ""
    ET.SubElement(test_elem, "Location").text = ""
    ET.SubElement(test_elem, "LocationName").text = ""
    ET.SubElement(test_elem, "AcquisitionTime").text = (
        fields.get("acquisition_time") or now.strftime("%H:%M:%S")
    )
    ET.SubElement(test_elem, "AcquisitionDate").text = (
        fields.get("acquisition_date") or now.strftime("%m-%d-%Y")
    )
    ET.SubElement(test_elem, "CartNumber").text = ""
    ET.SubElement(test_elem, "AcquisitionSoftwareVersion").text = ""
    ET.SubElement(test_elem, "AnalysisSoftwareVersion").text = ""
    ET.SubElement(test_elem, "HISStatus").text = ""

    _add_measurements_block(
        root, "RestingECGMeasurements", measurements, fs, include_frederica=True
    )
    _add_measurements_block(
        root, "OriginalRestingECGMeasurements", measurements, fs, include_frederica=False
    )

    waveform = ET.SubElement(root, "Waveform")
    ET.SubElement(waveform, "WaveformType").text = "Rhythm"
    ET.SubElement(waveform, "WaveformStartTime").text = "0"
    ET.SubElement(waveform, "NumberofLeads").text = str(len(lead_indices))
    ET.SubElement(waveform, "SampleType").text = "CONTINUOUS_SAMPLES"
    ET.SubElement(waveform, "SampleBase").text = str(int(fs))
    ET.SubElement(waveform, "SampleExponent").text = "0"
    ET.SubElement(waveform, "HighPassFilter").text = ""
    ET.SubElement(waveform, "LowPassFilter").text = ""
    ET.SubElement(waveform, "ACFilter").text = ""

    for name, idx in lead_indices:
        # mV -> uV as int16 little-endian, then base64-encode
        samples = np.round(signal[:, idx] * 1000.0).astype(np.int16)
        raw = samples.astype("<i2").tobytes()
        b64_str = base64.b64encode(raw).decode("ascii")

        lead_data = ET.SubElement(waveform, "LeadData")
        ET.SubElement(lead_data, "LeadByteCountTotal").text = str(len(raw))
        ET.SubElement(lead_data, "LeadTimeOffset").text = "0"
        ET.SubElement(lead_data, "LeadSampleCountTotal").text = str(len(samples))
        ET.SubElement(lead_data, "LeadAmplitudeUnitsPerBit").text = "1"
        ET.SubElement(lead_data, "LeadAmplitudeUnits").text = "MICROVOLTS"
        ET.SubElement(lead_data, "LeadHighLimit").text = "32767"
        ET.SubElement(lead_data, "LeadLowLimit").text = "-32768"
        ET.SubElement(lead_data, "LeadID").text = name
        ET.SubElement(lead_data, "LeadOffsetFirstSample").text = "0"
        ET.SubElement(lead_data, "FirstSampleBaseline").text = "0"
        ET.SubElement(lead_data, "LeadSampleSize").text = "2"
        ET.SubElement(lead_data, "LeadOff").text = "FALSE"
        ET.SubElement(lead_data, "BaselineSway").text = "FALSE"
        ET.SubElement(lead_data, "LeadDataCRC32").text = str(zlib.crc32(raw))
        ET.SubElement(lead_data, "WaveFormData").text = b64_str

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE RestingECG SYSTEM "restecg.dtd">\n')
        tree.write(f, encoding="unicode", xml_declaration=False)
