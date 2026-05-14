import io
import re
import xml.etree.ElementTree as ET


def _xml_local_name(tag):
    if not isinstance(tag, str):
        return ""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _xml_find_first(root, local_name):
    for elem in root.iter():
        if _xml_local_name(elem.tag) == local_name:
            return elem
    return None


def _xml_get_child_text(parent, local_name, default=None):
    if parent is None:
        return default
    for elem in list(parent):
        if _xml_local_name(elem.tag) == local_name:
            text = elem.text.strip() if elem.text else ""
            return text if text != "" else default
    return default


def _xml_elem_text(elem, default=None):
    if elem is None or elem.text is None:
        return default
    text = elem.text.strip()
    return text if text != "" else default


def _parse_xml_any_encoding(xml_path):
    try:
        return ET.parse(xml_path)
    except ET.ParseError:
        pass
    with open(xml_path, 'rb') as f:
        raw = f.read()
    if raw.startswith(b'\xff\xfe'):
        text = raw[2:].decode('utf-16-le')
    elif raw.startswith(b'\xfe\xff'):
        text = raw[2:].decode('utf-16-be')
    elif raw.startswith(b'\xef\xbb\xbf'):
        text = raw[3:].decode('utf-8')
    elif raw.startswith(b'\x3c\x00') or raw.startswith(b'\x00\x00\x3c\x00'):
        text = raw.decode('utf-16-le')
    elif raw.startswith(b'\x00\x3c'):
        text = raw.decode('utf-16-be')
    else:
        text = raw.decode('utf-8', errors='replace')
    stripped = text.lstrip()
    if stripped.startswith('<?xml'):
        end = stripped.index('?>') + 2
        stripped = stripped[end:].lstrip()
    stripped = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', stripped)
    # Fix unescaped & not part of a valid entity or character reference
    stripped = re.sub(r'&(?!(?:#\d+|#x[\da-fA-F]+|[a-zA-Z]\w*);)', '&amp;', stripped)
    try:
        return ET.parse(io.BytesIO(stripped.encode('utf-8')))
    except ET.ParseError:
        # Last resort: strip declaration and retry with latin-1 fallback
        text2 = raw.decode('latin-1', errors='replace')
        text2 = re.sub(r'<\?xml[^?]*\?>', '', text2)
        text2 = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text2)
        text2 = re.sub(r'&(?!(?:#\d+|#x[\da-fA-F]+|[a-zA-Z]\w*);)', '&amp;', text2)
        return ET.parse(io.BytesIO(text2.encode('utf-8')))


def detect_xml_type(xml_path):
    tree = _parse_xml_any_encoding(xml_path)
    root = tree.getroot()
    tag = root.tag

    if _xml_local_name(tag) == "HativECG":
        return "hativ"

    if _xml_local_name(tag) == "restingecgdata":
        document_info = _xml_find_first(root, "documentinfo")
        doc_type = _xml_get_child_text(document_info, "documenttype", "")
        if doc_type in ["PhilipsECG", "SierraECG"]:
            return "philips"

    if _xml_local_name(tag) == "ClinicalDocument":
        code_el = _xml_find_first(root, "code")
        if code_el is not None and code_el.attrib.get("codeSystemName") == "NK_MFER":
            return "nihonkohden"

    if "AnnotatedECG" in tag or "hl7-org:v3" in tag:
        return "trismed"
    if "sapphire" in tag or "urn:ge:sapphire" in tag:
        return "mac2000"
    if root.find("Waveform") is not None:
        return "muse"

    # Bionet CardioXP
    if _xml_local_name(tag) == "CardioXP":
        return "bionet"

    return "unknown"
