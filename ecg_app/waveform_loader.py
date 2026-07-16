# Backwards-compatibility shim — all logic has moved to ecg_app/loaders/.
from .loaders import (
    load_waveform,
    normalize_input_to_base_path,
    find_companion_file,
)
from .loaders.mwf import parse_mwf_fixed_family, load_dat_mwf
from .loaders.mat_hea import load_mat_hea
from .loaders.csv_loader import load_csv_waveform
from .loaders.json_loader import load_json_waveform
from .loaders.ekg_microcor import load_ekg_waveform
from .loaders.fukuda import load_fukuda_ecg
from .loaders.xml_common import detect_xml_type
from .loaders.xml_hativ import load_hativ_xml
from .loaders.xml_philips import load_philips_xml
from .loaders.xml_muse import load_muse_xml
from .loaders.xml_mac2000 import load_mac2000_xml
from .loaders.xml_trismed import load_trismed_xml
from .loaders.xml_bionet import load_bionet_xml

__all__ = [
    "load_waveform",
    "normalize_input_to_base_path",
    "find_companion_file",
    "parse_mwf_fixed_family",
    "load_dat_mwf",
    "load_mat_hea",
    "load_csv_waveform",
    "load_json_waveform",
    "load_ekg_waveform",
    "load_fukuda_ecg",
    "detect_xml_type",
    "load_hativ_xml",
    "load_philips_xml",
    "load_muse_xml",
    "load_mac2000_xml",
    "load_trismed_xml",
    "load_bionet_xml",
]
