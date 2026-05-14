import xml.etree.ElementTree as ET

from .xml_common import _parse_xml_any_encoding


# LOINC code → measurement key mapping
_LOINC_MAP = {
    "9873-1":  ("hr",    None),   # bpm (int)
    "8625-6":  ("pr",    None),   # msec
    "18517-3": ("qrs",   None),   # msec
    "8634-8":  ("qt",    None),   # msec
    "8636-3":  ("qtc",   None),   # msec
    "8626-4":  ("p_axis", None),  # deg
    "8632-2":  ("r_axis", None),  # deg
    "8638-9":  ("t_axis", None),  # deg
    "10040-4": ("sv1",   "mV"),   # mV float
    "9995-2":  ("rv5",   "mV"),   # mV float
}


def load_nihonkohden_measurements(xml_path):
    """Read clinical measurements from Nihon Kohden ClinicalDocument XML (NK_MFER format).

    Values are stored as LOINC-coded observations with <value type='PQ'> attributes.
    RV5+SV1 is computed as the sum of RV5 and SV1.
    """
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
        if loinc not in _LOINC_MAP:
            continue
        key, unit_hint = _LOINC_MAP[loinc]
        raw = val_el.attrib.get("value", "")
        if not raw:
            continue
        try:
            result[key] = float(raw) if unit_hint == "mV" else int(raw)
        except ValueError:
            pass

    rv5 = result.get("rv5")
    sv1 = result.get("sv1")
    if rv5 is not None and sv1 is not None:
        result["rs"] = round(rv5 + sv1, 3)

    return result
