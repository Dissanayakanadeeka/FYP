import numpy as np
from scipy.signal import welch

def remove_baseline(eeg, fs=128, baseline_sec=3):
    baseline_samples = baseline_sec * fs
    return eeg[:, baseline_samples:]

def segment_epochs(eeg, fs=128, epoch_sec=1):
    samples_per_epoch = fs * epoch_sec
    num_epochs = eeg.shape[1] // samples_per_epoch

    epochs = []
    for i in range(num_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epochs.append(eeg[:, start:end])

    return np.array(epochs)

def normalize_epoch(epoch):
    mean = np.mean(epoch, axis=1, keepdims=True)
    std = np.std(epoch, axis=1, keepdims=True) + 1e-8
    return (epoch - mean) / std

def compute_psd(epoch, fs=128, fmin=4, fmax=45):
    psd_features = []

    for ch in range(epoch.shape[0]):
        freqs, psd = welch(epoch[ch], fs=fs, nperseg=fs)
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        psd_features.append(psd[idx])

    return np.array(psd_features)
