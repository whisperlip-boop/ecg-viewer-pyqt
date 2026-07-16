import json
from pathlib import Path

import wfdb

from ._utils import find_companion_file

_METRICS_PREFIX = "# ecg_metrics:"


def _read_hea_measurements(hea_path: str) -> dict | None:
    """Scan a .hea file for an embedded ``# ecg_metrics:{JSON}`` comment.

    Args:
        hea_path: Path to the .hea header file.

    Returns:
        Parsed measurements dict, or None if not found.
    """
    try:
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith(_METRICS_PREFIX):
                    return json.loads(line.strip()[len(_METRICS_PREFIX):])
    except Exception:
        pass
    return None
from .mwf import load_dat_mwf
from .mat_hea import load_mat_hea
from .csv_loader import load_csv_waveform
from .json_loader import load_json_waveform
from .ekg_microcor import load_ekg_waveform
from .fukuda import load_fukuda_ecg
from .xml_common import detect_xml_type
from .xml_hativ import load_hativ_xml
from .xml_philips import load_philips_xml
from .xml_muse import load_muse_xml
from .xml_mac2000 import load_mac2000_xml
from .xml_trismed import load_trismed_xml
from .xml_bionet import load_bionet_xml


def normalize_input_to_base_path(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in [".dat", ".hea", ".mwf", ".mat", ".csv"]:
        return str(path.with_suffix(""))
    if suffix in [".xml", ".ecg", ".json", ".ekg"]:
        return str(path)
    return str(path)


def load_waveform(base_path):
    path = Path(base_path)

    if path.suffix.lower() == ".ecg" and path.exists():
        return load_fukuda_ecg(str(path))

    if path.suffix.lower() == ".json" and path.exists():
        return load_json_waveform(str(path))

    if path.suffix.lower() == ".ekg" and path.exists():
        return load_ekg_waveform(str(path))

    if path.suffix.lower() == ".xml" and path.exists():
        xml_type = detect_xml_type(str(path))

        if xml_type == "hativ":
            return load_hativ_xml(str(path))
        if xml_type == "philips":
            return load_philips_xml(str(path))
        if xml_type == "trismed":
            return load_trismed_xml(str(path))
        if xml_type == "mac2000":
            return load_mac2000_xml(str(path))
        if xml_type == "muse":
            return load_muse_xml(str(path))
        if xml_type == "bionet":
            return load_bionet_xml(str(path))
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
        "- .csv (standalone)\n"
        "- .json (standalone)\n"
        "- .ekg (standalone, microCOR)"
    )
