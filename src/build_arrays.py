import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, OUTPUTS_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.feature_extraction import file_to_features


def build_split(df_split):
    """Dla każdego utworu ze zbioru: pocięcie na fragmenty -> spektrogramy.
    Zwraca X (fragmenty), y (numer klasy) i ids (numer utworu dla każdego fragmentu)."""
    X_list, y_list, id_list = [], [], []

    for i, row in enumerate(df_split.itertuples(), start=1):
        feats = file_to_features(PROJECT_ROOT / row.filepath)     # (n, 128, 216, 1)
        label = CLASS_NAMES.index(row.genre)                      # numer klasy z config.py

        X_list.append(feats)
        y_list.append(np.full(len(feats), label, dtype=np.int64))
        id_list.append(np.full(len(feats), row.track_id, dtype=np.int64))

        if i % 50 == 0:
            print(f"   {i}/{len(df_split)} utworów...")

    return np.concatenate(X_list), np.concatenate(y_list), np.concatenate(id_list)

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(OUTPUTS_DIR / "metadata_split.csv")

    track_ids_per_split = {}
    for split in ["train", "val", "test"]:
        print(f"\nPrzetwarzanie zbioru: {split}")
        X, y, ids = build_split(meta[meta["split"] == split])

        np.save(PROCESSED_DIR / f"X_{split}.npy", X)
        np.save(PROCESSED_DIR / f"y_{split}.npy", y)
        np.save(PROCESSED_DIR / f"ids_{split}.npy", ids)
        track_ids_per_split[split] = set(ids.tolist())

        print(f"   X_{split}: {X.shape}, {X.dtype}")
        print(f"   y_{split}: {y.shape}")
        print(f"   zakres wartości: {X.min():.3f} .. {X.max():.3f}")
        print(f"   liczba fragmentów na klasę: {np.bincount(y, minlength=len(CLASS_NAMES)).tolist()}")

    # Kontrola data leakage na poziomie fragmentów
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (track_ids_per_split[a] & track_ids_per_split[b]), f"LEAKAGE: {a}/{b}"
    print("\nKontrola: żaden utwór nie ma fragmentów w więcej niż jednym zbiorze. OK")


if __name__ == "__main__":
    main()