import xml.etree.ElementTree as ET

import numpy as np

from ..constants import STANDARD_12_ORDER
from ..signal_processing import remove_isolated_spikes


def load_trismed_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {
        "hl7": "urn:hl7-org:v3",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }

    rhythm_series = None
    for series in root.findall("./hl7:component/hl7:series", ns):
        code_elem = series.find("./hl7:code", ns)
        if code_elem is not None and (code_elem.attrib.get("code") or "").strip().upper() == "RHYTHM":
            rhythm_series = series
            break

    if rhythm_series is None:
        raise ValueError("RHYTHM series not found in Trismed XML.")

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
        "MDC_ECG_LEAD_aVR": "aVR",
        "MDC_ECG_LEAD_aVL": "aVL",
        "MDC_ECG_LEAD_aVF": "aVF",
    }

    lead_map = {}
    scale_mode = "digits_scale_assume_uv"

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
    model_elem = root.find(".//hl7:manufacturerModelName", ns)
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


def load_trismed_measurements(xml_path):
    """Read clinical measurements from Trismed HL7/aECG XML.

    Structured values come from MDC observation codes.
    Axis values are embedded in free-text interpretation statements and
    extracted via regex. RV5/SV1 are not stored in the XML.
    """
    import re
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

    # Collect all interpretation text lines
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
            m = re.search(pattern, line)
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
        m = re.search(r'\bQTr[:\s]+([\d.]+)', line)
        if m:
            try:
                qtr = float(m.group(1))
            except ValueError:
                pass
            break

    return {
        "hr":       mdc_map.get("MDC_ECG_HEART_RATE"),
        "pr":       mdc_map.get("MDC_ECG_TIME_PD_PR"),
        "qrs":      mdc_map.get("MDC_ECG_TIME_PD_QRS"),
        "qt":       mdc_map.get("MDC_ECG_TIME_PD_QT"),
        "qtc":      mdc_map.get("MDC_ECG_TIME_PD_QTc"),
        "qtr":      qtr,
        "p_axis":   p_axis,
        "r_axis":   qrs_axis,
        "t_axis":   t_axis,
        "qrs_t":    qrs_t_axis,
    }
