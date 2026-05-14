import numpy as np

from ..constants import STANDARD_12_ORDER

# Huffman table recovered from fhdecode.dll .data section (RVA 0x7034).
# State 13 is root; states 0..12 are leaves. Each entry: (child_if_bit0, child_if_bit1).
# Leaf state → delta: states 0..10 → delta = state-5 (-5..+5),
#   state 11 → 8-bit escape, state 12 → 16-bit escape, state -1 → terminator.
_FUKUDA_HUFF_TABLE = {
    13: (15, 14), 14: (5, 4),  15: (17, 16),
    16: (6, 3),   17: (18, 7), 18: (20, 19),
    19: (2, 8),   20: (21, 11), 21: (22, 1),
    22: (23, 9),  23: (24, 0), 24: (25, 10),
    25: (-1, 12),
}


def _decode_fukuda_waveform(compressed, lead_size, lead_count=8):
    """Huffman + 2nd-order delta decoder (pure Python port of fhdecode.dll)."""
    n_words = len(compressed) // 2
    words = np.frombuffer(compressed[:n_words * 2], dtype=np.dtype(">u2"))
    total = lead_size * lead_count
    out = np.zeros(total, dtype=np.int16)

    word_idx = 0
    bit_pos = 15
    prev = prev_prev = 0
    n = 0

    def read_bit():
        nonlocal word_idx, bit_pos
        if word_idx >= n_words:
            raise EOFError
        b = (int(words[word_idx]) >> bit_pos) & 1
        bit_pos -= 1
        if bit_pos < 0:
            bit_pos = 15
            word_idx += 1
        return b

    def read_bits(k):
        v = 0
        for _ in range(k):
            v = (v << 1) | read_bit()
        return v

    while n < total:
        state = 13
        try:
            while state >= 13:
                b = read_bit()
                left, right = _FUKUDA_HUFF_TABLE[state]
                state = right if b else left
        except EOFError:
            break
        if state < 0:
            break

        if state <= 10:
            delta = state - 5
        elif state == 11:
            byte = read_bits(8)
            delta = byte - 256 if byte >= 128 else byte
        else:
            word = read_bits(16)
            delta = word - 65536 if word >= 32768 else word

        sample = delta + 2 * prev - prev_prev
        sample = ((sample + 0x8000) & 0xFFFF) - 0x8000  # wrap to int16
        out[n] = sample
        n += 1
        prev_prev = prev
        prev = sample

    return out.reshape(lead_count, lead_size)


def load_fukuda_ecg(ecg_path):
    """Fukuda Denshi proprietary .ecg format (pure Python, no DLL required).

    File layout: 112-byte header (32-byte prefix + 5×16-byte UnitInfo table),
    then 5 sequential unit blobs: Info / Patient / Measurement / Diagno / History.
    The HistoryUnit contains the Huffman+2nd-order-delta compressed waveform.
    """
    import struct

    with open(ecg_path, "rb") as f:
        data = f.read()

    file_unit_size = struct.unpack_from(">I", data, 4)[0]

    unit_sizes = [
        struct.unpack_from(">I", data, 32 + i * 16 + 4)[0]
        for i in range(5)
    ]

    unit_offsets = [file_unit_size]
    for sz in unit_sizes[:-1]:
        unit_offsets.append(unit_offsets[-1] + sz)

    hist_off = unit_offsets[4]
    hist_blob = data[hist_off: hist_off + unit_sizes[4]]

    p = 10
    lsb_size   = struct.unpack_from(">h", hist_blob, p + 2)[0]
    lead_size  = struct.unpack_from(">i", hist_blob, p + 4)[0]
    lead_count = struct.unpack_from(">h", hist_blob, p + 10)[0]
    p = 24
    p += lead_count * 2
    data_size  = struct.unpack_from(">i", hist_blob, p)[0]
    p += 4
    compressed = bytes(hist_blob[p: p + data_size])

    int16_8 = _decode_fukuda_waveform(compressed, lead_size, lead_count)

    i32  = int16_8[0].astype(np.int32)
    ii32 = int16_8[1].astype(np.int32)
    iii = (ii32 - i32).astype(np.int16)
    avr = (-(ii32 + i32) // 2).astype(np.int16)
    avl = ((i32 - (ii32 - i32)) // 2).astype(np.int16)
    avf = ((ii32 + (ii32 - i32)) // 2).astype(np.int16)

    wf12 = np.stack([
        int16_8[0], int16_8[1], iii, avr, avl, avf,
        int16_8[2], int16_8[3], int16_8[4], int16_8[5], int16_8[6], int16_8[7],
    ], axis=-1)
    signal = wf12.astype(np.float32) * float(lsb_size) * 1e-6

    fields = {
        "fs": 500.0,
        "sig_name": list(STANDARD_12_ORDER),
        "units": ["mV"] * 12,
        "n_sig": 12,
        "source": "fukuda_ecg",
        "ecg_path": ecg_path,
    }
    return signal, fields


def load_fukuda_measurements(ecg_path):
    """Read clinical measurement values from MeasurementUnit of a Fukuda .ecg file.

    Offsets verified against 2023051810233700001.ecg. Values stored as BE int16.
    Time fields in ms; voltage fields in 10µV units (abs for SV1/RV6 — sign
    reflects wave polarity, clinical display uses magnitude).
    """
    import struct

    with open(ecg_path, "rb") as f:
        data = f.read()

    file_unit_size = struct.unpack_from(">I", data, 4)[0]
    unit_sizes = [struct.unpack_from(">I", data, 32 + i * 16 + 4)[0] for i in range(5)]
    unit_offsets = [file_unit_size]
    for sz in unit_sizes[:-1]:
        unit_offsets.append(unit_offsets[-1] + sz)

    blob = data[unit_offsets[2]: unit_offsets[2] + unit_sizes[2]]

    def rd(off):
        return struct.unpack_from(">h", blob, off)[0]

    sv1_mv = abs(rd(0x0206)) * 10 / 1000.0   # 10µV units → mV
    rv6_mv = abs(rd(0x0356)) * 10 / 1000.0

    return {
        "hr":   rd(0x0034),                    # bpm
        "rr":   rd(0x0036) / 1000.0,           # ms → s
        "pr":   rd(0x0038) / 1000.0,           # ms → s
        "qrs":  rd(0x003a) / 1000.0,           # ms → s
        "qt":   rd(0x003c) / 1000.0,           # ms → s
        "qtc":  rd(0x003e) / 1000.0,           # ms → s (or dimensionless)
        "axis": rd(0x0042),                    # degrees
        "sv1":  sv1_mv,                        # mV
        "rv6":  rv6_mv,                        # mV
        "rs":   sv1_mv + rv6_mv,               # mV (Sokolow-Lyon)
    }
