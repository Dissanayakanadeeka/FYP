import os
import pickle
import numpy as np
from utils import remove_baseline, segment_epochs, normalize_epoch, compute_psd

RAW_PATH = "data/raw_deap"
SAVE_PATH = "data/processed"

os.makedirs(SAVE_PATH, exist_ok=True)

for subject_file in sorted(os.listdir(RAW_PATH)):
    if not subject_file.endswith(".dat"):
        continue

    print(f"Processing {subject_file}")

    with open(os.path.join(RAW_PATH, subject_file), "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    data = subject["data"]        # (40, 32, 8064)
    labels = subject["labels"]    # (40, 4)

    X_raw, X_norm, X_psd, X_psdnorm = [], [], [], []
    y_valence, y_arousal = [], []

    for trial in range(40):
        eeg = data[trial]
        valence = labels[trial][0]
        arousal = labels[trial][1]

        eeg = remove_baseline(eeg)
        epochs = segment_epochs(eeg)

        for epoch in epochs:
            raw = epoch
            norm = normalize_epoch(epoch)
            psd = compute_psd(epoch)
            psdnorm = np.concatenate([norm.flatten(), psd.flatten()])

            X_raw.append(raw)
            X_norm.append(norm)
            X_psd.append(psd)
            X_psdnorm.append(psdnorm)

            y_valence.append(valence)
            y_arousal.append(arousal)

    save_file = subject_file.replace(".dat", ".npz")
    np.savez(
        os.path.join(SAVE_PATH, save_file),
        X_RAW=np.array(X_raw),
        X_NORM=np.array(X_norm),
        X_PSD=np.array(X_psd),
        X_PSDNORM=np.array(X_psdnorm),
        y_valence=np.array(y_valence),
        y_arousal=np.array(y_arousal),
    )

    print(f"Saved {save_file}")
