import base64
from array import array

import numpy as np

from ..constants import STANDARD_12_ORDER
from ..signal_processing import remove_isolated_spikes
from .xml_common import (
    _xml_local_name, _xml_find_first, _xml_get_child_text,
    _xml_elem_text, _parse_xml_any_encoding,
)


class PhilipsLzwDecoder:
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
        leads.append(all_samples[offset:offset + sample_count])
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


def load_philips_measurements(xml_path):
    """Read clinical measurements from Philips XML.

    Supports two layouts:
    - TC70 / PageWriter: <interpretations><interpretation><globalmeasurements>
      tags: heartrate, print, qrsdur, qtint, qtcb
    - Trim3 / older:     <measurements><globalmeasurements>
      tags: meanventrate, meanprint, meanqrsdur, meanqtint, meanqtc
    Both share axis tags: pfrontaxis, qrsfrontaxis, tfrontaxis.
    """
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()

    ns_uri = None
    if "}" in root.tag:
        ns_uri = root.tag.split("}")[0].lstrip("{")
    ns = {"p": ns_uri} if ns_uri else {}
    p = "p:" if ns_uri else ""

    def find(path):
        return root.find(path, ns) if ns else root.find(path)

    def get(elem, tag):
        if elem is None:
            return None
        el = find(f".//{p}{tag}") if elem is root else elem.find(f"{p}{tag}", ns) if ns else elem.find(tag)
        if el is None or not (el.text or "").strip():
            return None
        try:
            return int(el.text.strip())
        except ValueError:
            return None

    # TC70 style: interpretation/globalmeasurements with heartrate tag
    gm_interp = find(f".//{p}interpretation/{p}globalmeasurements")
    if gm_interp is not None and (gm_interp.find(f"{p}heartrate", ns) if ns else gm_interp.find("heartrate")) is not None:
        g = gm_interp
        hr_tag, pr_tag, qrs_tag, qt_tag, qtc_tag = "heartrate", "print", "qrsdur", "qtint", "qtcb"
    else:
        # Trim3 style: measurements/globalmeasurements
        g = find(f".//{p}measurements/{p}globalmeasurements") or find(f".//{p}globalmeasurements")
        hr_tag, pr_tag, qrs_tag, qt_tag, qtc_tag = "meanventrate", "meanprint", "meanqrsdur", "meanqtint", "meanqtc"

    def gget(tag):
        if g is None:
            return None
        el = g.find(f"{p}{tag}", ns) if ns else g.find(tag)
        if el is None or not (el.text or "").strip():
            return None
        try:
            return int(el.text.strip())
        except ValueError:
            return None

    return {
        "hr":     gget(hr_tag),
        "pr":     gget(pr_tag),
        "qrs":    gget(qrs_tag),
        "qt":     gget(qt_tag),
        "qtc":    gget(qtc_tag),
        "p_axis": gget("pfrontaxis"),
        "r_axis": gget("qrsfrontaxis"),
        "t_axis": gget("tfrontaxis"),
    }
