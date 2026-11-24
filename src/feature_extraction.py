import librosa
import numpy as np

from src.config import (
    HOP_LENGTH, N_FFT, N_MELS, SEGMENT_SAMPLES, SR, TOP_DB,
)


def load_audio(path):
    """Wczytuje plik audio (WAV/MP3) jako mono, z częstotliwością SR."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y

def split_into_segments(y):
    """Dzieli sygnał na kolejne fragmenty po SEGMENT_SAMPLES próbek (bez nakładania).
    Niepełny ostatni fragment jest odrzucany. Zbyt krótki sygnał (< 1 s) zgłasza błąd,
    a krótszy niż jeden fragment jest dopełniany zerami."""
    if len(y) < SR:
        raise ValueError("Nagranie jest zbyt krótkie (minimum 1 sekunda).")
    if len(y) < SEGMENT_SAMPLES:
        y = np.pad(y, (0, SEGMENT_SAMPLES - len(y)))

    n_segments = len(y) // SEGMENT_SAMPLES
    segments = y[: n_segments * SEGMENT_SAMPLES].reshape(n_segments, SEGMENT_SAMPLES)
    return segments

def segment_to_mel(segment):
    """Zamienia jeden fragment audio na znormalizowany spektrogram Mel (N_MELS, czas)."""
    mel = librosa.feature.melspectrogram(
        y=segment, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)   # skala dB: od -80 do 0
    mel_norm = (mel_db + TOP_DB) / TOP_DB                          # skala 0..1
    return mel_norm.astype(np.float32)

def audio_to_features(y):
    """Sygnał audio -> tablica (n_fragmentów, N_MELS, czas, 1), gotowa dla CNN."""
    segments = split_into_segments(y)
    mels = np.stack([segment_to_mel(seg) for seg in segments])
    return mels[..., np.newaxis]          # dodajemy oś "kanał", jak w obrazie


def file_to_features(path):
    """Ścieżka do pliku -> tablica cech dla CNN."""
    return audio_to_features(load_audio(path))