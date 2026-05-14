import numpy as np
from scipy.signal import butter, filtfilt


def safe_bandpass_filter(signal, lowcut, highcut, fs, order=3):
    if signal is None or len(signal) < max(32, order * 8):
        return signal

    nyquist = 0.5 * fs
    if nyquist <= 0:
        return signal

    low = max(lowcut / nyquist, 1e-6)
    high = min(highcut / nyquist, 0.999999)

    if low >= high:
        return signal

    try:
        b, a = butter(order, [low, high], btype="band")
        padlen = 3 * (max(len(a), len(b)) - 1)
        if signal.shape[0] <= padlen:
            return signal
        return filtfilt(b, a, signal, axis=0)
    except Exception:
        return signal


def remove_baseline_wander(signal, fs, cutoff=0.5, order=2):
    if signal is None or len(signal) < 32:
        return signal

    nyquist = 0.5 * fs
    if nyquist <= 0:
        return signal

    wn = cutoff / nyquist
    wn = min(max(wn, 1e-6), 0.999999)

    try:
        b, a = butter(order, wn, btype="high")
        padlen = 3 * (max(len(a), len(b)) - 1)
        if signal.shape[0] <= padlen:
            return signal
        return filtfilt(b, a, signal, axis=0)
    except Exception:
        return signal


def remove_isolated_spikes(signal, z=8.0):
    out = signal.copy()

    for ch in range(out.shape[1]):
        x = out[:, ch]
        if len(x) < 5:
            continue

        local_med = (np.roll(x, 1) + np.roll(x, -1)) / 2.0
        diff = x - local_med

        core = diff[1:-1]
        mad = np.median(np.abs(core - np.median(core)))
        if mad < 1e-12:
            continue

        thresh = z * 1.4826 * mad
        mask = np.abs(diff) > thresh
        mask[0] = False
        mask[-1] = False

        out[mask, ch] = local_med[mask]

    return out


def derive_12_from_base8(base8):
    if base8.shape[1] != 8:
        raise ValueError("base8 must have 8 channels")

    i_lead = base8[:, 0]
    ii_lead = base8[:, 1]

    v1 = base8[:, 2]
    v2 = base8[:, 3]
    v3 = base8[:, 4]
    v4 = base8[:, 5]
    v5 = base8[:, 6]
    v6 = base8[:, 7]

    iii = ii_lead - i_lead
    avr = -(i_lead + ii_lead) / 2.0
    avl = i_lead - (ii_lead / 2.0)
    avf = ii_lead - (i_lead / 2.0)

    return np.column_stack([
        i_lead,
        ii_lead,
        iii,
        avr,
        avl,
        avf,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
    ])
