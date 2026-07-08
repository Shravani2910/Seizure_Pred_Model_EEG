"""
Preprocess CHB-MIT EDF files into labeled, filtered, windowed EEG segments.

Usage:
    python preprocess.py --patient chb01 --data_dir data/chb01 --out_dir processed/chb01

Expects the standard CHB-MIT layout:
    data/chb01/chb01_03.edf
    data/chb01/chb01-summary.txt   <- contains seizure start/end times

Output:
    processed/chb01/X.npy   shape (n_windows, n_channels, n_samples)
    processed/chb01/y.npy   shape (n_windows,)  1 = preictal, 0 = interictal
"""
import argparse
import os
import re
import numpy as np
import mne

# ---- Config ----
LOWCUT, HIGHCUT = 0.5, 40.0     # band-pass filter range (Hz)
NOTCH_FREQ = 60.0                # use 50.0 if your recordings are from a 50Hz-mains region
WINDOW_SEC = 10                  # window length in seconds
OVERLAP = 0.5                    # 50% overlap between windows
PREICTAL_MIN = 60                # minutes before seizure onset considered "preictal"
HORIZON_MIN = 5                  # exclude the last N minutes before onset (too close to ictal)
INTERICTAL_BUFFER_HR = 4         # hours away from any seizure to count as interictal


def parse_summary(summary_path):
    """Parse chbXX-summary.txt for per-file seizure start/end times (in seconds)."""
    seizures = {}  # filename -> list of (start_sec, end_sec)
    current_file = None
    with open(summary_path, "r", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            current_file = line.split(":", 1)[1].strip()
            seizures.setdefault(current_file, [])
        elif "Seizure" in line and "Start Time" in line:
            sec = int(re.findall(r"\d+", line)[-1])
            seizures[current_file].append([sec, None])
        elif "Seizure" in line and "End Time" in line:
            sec = int(re.findall(r"\d+", line)[-1])
            # attach to the last seizure entry missing an end time
            for entry in reversed(seizures[current_file]):
                if entry[1] is None:
                    entry[1] = sec
                    break
    return seizures


def load_edf(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.filter(LOWCUT, HIGHCUT, fir_design="firwin", verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    return raw


def window_signal(data, sfreq, window_sec=WINDOW_SEC, overlap=OVERLAP):
    """data: (n_channels, n_samples) -> list of (n_channels, window_samples) + start times (sec)."""
    win_len = int(window_sec * sfreq)
    step = int(win_len * (1 - overlap))
    windows, starts = [], []
    for start in range(0, data.shape[1] - win_len, step):
        windows.append(data[:, start:start + win_len])
        starts.append(start / sfreq)
    return windows, starts


def label_windows(starts, duration_sec, seizure_times):
    """
    Returns label per window start time:
      1 = preictal (within PREICTAL_MIN before onset, excluding HORIZON_MIN buffer)
      0 = interictal (more than INTERICTAL_BUFFER_HR from any seizure)
     -1 = discard (ambiguous: ictal, postictal, or in the excluded gap)
    """
    labels = []
    for t in starts:
        label = 0
        for (onset, offset) in seizure_times:
            if onset is None or offset is None:
                continue
            preictal_start = onset - PREICTAL_MIN * 60
            horizon_end = onset - HORIZON_MIN * 60
            if preictal_start <= t < horizon_end:
                label = 1
                break
            # too close to / during / after seizure -> ambiguous, discard
            if (onset - INTERICTAL_BUFFER_HR * 3600) <= t <= (offset + INTERICTAL_BUFFER_HR * 3600):
                label = -1
        labels.append(label)
    return labels


def main(args):
    summary_path = os.path.join(args.data_dir, f"{args.patient}-summary.txt")
    seizures_by_file = parse_summary(summary_path)

    all_X, all_y = [], []
    for fname, seizure_times in seizures_by_file.items():
        edf_path = os.path.join(args.data_dir, fname)
        if not os.path.exists(edf_path):
            continue
        print(f"Processing {fname} ({len(seizure_times)} seizures)")

        raw = load_edf(edf_path)
        data = raw.get_data()  # (n_channels, n_samples)
        sfreq = raw.info["sfreq"]

        windows, starts = window_signal(data, sfreq)
        labels = label_windows(starts, data.shape[1] / sfreq, seizure_times)

        for w, l in zip(windows, labels):
            if l == -1:
                continue
            all_X.append(w)
            all_y.append(l)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    print(f"Total windows: {len(y)} | preictal: {(y==1).sum()} | interictal: {(y==0).sum()}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True, help="e.g. chb01")
    parser.add_argument("--data_dir", required=True, help="folder with .edf + summary.txt")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)
