import librosa
import numpy as np
import torch
from transformers import pipeline

from src.config import (
    CLASS_NAMES, MAX_WINDOWS, MIN_SECONDS, PRETRAINED_MODEL_ID,
    PRETRAINED_SR, WINDOW_SECONDS,
)


def load_classifier():
    """Ładuje gotowy model (na GPU, jeśli jest dostępne, w przeciwnym razie na CPU)."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("audio-classification", model=PRETRAINED_MODEL_ID,
                    device=device, top_k=len(CLASS_NAMES))

def load_windows(path):
    """Wczytuje do MAX_WINDOWS okien po WINDOW_SECONDS, równomiernie z całego nagrania.
    Zwraca (lista okien audio, całkowita długość w sekundach)."""
    total = librosa.get_duration(path=path)
    if total < MIN_SECONDS:
        raise ValueError(f"Nagranie jest zbyt krótkie ({total:.1f} s). Minimum to {MIN_SECONDS:.0f} s.")

    if total <= WINDOW_SECONDS:
        starts, length = [0.0], total
    else:
        n = max(1, min(MAX_WINDOWS, int(total // WINDOW_SECONDS)))
        starts, length = np.linspace(0, total - WINDOW_SECONDS, n), WINDOW_SECONDS

    windows = []
    for s in starts:
        y, _ = librosa.load(path, sr=PRETRAINED_SR, mono=True, offset=float(s), duration=length)
        windows.append(y)
    return windows, total


def predict_genre(clf, windows):
    """Zwraca wektor 10 prawdopodobieństw (kolejność CLASS_NAMES), uśredniony po oknach."""
    all_probs = []
    for y in windows:
        scores = {p["label"]: p["score"] for p in clf(y)}
        all_probs.append([scores[g] for g in CLASS_NAMES])
    return np.mean(all_probs, axis=0)