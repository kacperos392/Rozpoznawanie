import hashlib
import os

import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf

from src.config import CLASS_NAMES, DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT


def build_file_table():
    """Przechodzi po folderach gatunków i buduje tabelę: ścieżka, gatunek, nazwa pliku."""
    rows = []
    for genre in sorted(os.listdir(DATA_DIR)):
        genre_dir = DATA_DIR / genre
        if not genre_dir.is_dir():
            continue
        for fname in sorted(os.listdir(genre_dir)):
            if fname.lower().endswith(".wav"):
                path = (genre_dir / fname).relative_to(PROJECT_ROOT).as_posix()
                rows.append({"filepath": path, "genre": genre, "filename": fname})
    return pd.DataFrame(rows)

def inspect_audio(df):
    """Dla każdego pliku: czy da się wczytać, jaka długość, częstotliwość, kanały, hash zawartości."""
    sample_rates, durations, channels, ok_flags, errors, hashes = [], [], [], [], [], []

    for i, rel_path in enumerate(df["filepath"], start=1):
        full_path = PROJECT_ROOT / rel_path
        try:
            data, sr = sf.read(full_path, dtype="float32", always_2d=True)
            n_frames, n_channels = data.shape
            sample_rates.append(sr)
            durations.append(n_frames / sr)
            channels.append(n_channels)
            hashes.append(hashlib.md5(data.tobytes()).hexdigest())
            ok_flags.append(True)
            errors.append("")
        except Exception as e:  # uszkodzony lub nieczytelny plik
            sample_rates.append(None)
            durations.append(None)
            channels.append(None)
            hashes.append(None)
            ok_flags.append(False)
            errors.append(str(e))

        if i % 100 == 0:
            print(f"Sprawdzono {i}/{len(df)} plików...")

    df = df.copy()
    df["sr"] = sample_rates
    df["duration_s"] = durations
    df["channels"] = channels
    df["md5"] = hashes
    df["ok"] = ok_flags
    df["error"] = errors
    return df


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df = build_file_table()
    print("Liczba znalezionych plików:", len(df))

    # Czy foldery zgadzają się z naszą listą klas?
    found = sorted(df["genre"].unique())
    print("Znalezione gatunki:", found)
    print("Zgodne z CLASS_NAMES:", found == sorted(CLASS_NAMES))

    df = inspect_audio(df)

    # Liczba nagrań w klasach
    print("\nLiczba nagrań w klasach:")
    print(df["genre"].value_counts().sort_index())

    # Uszkodzone pliki
    bad = df[~df["ok"]]
    print(f"\nUszkodzone/nieczytelne pliki: {len(bad)}")
    if len(bad) > 0:
        print(bad[["filepath", "error"]].to_string(index=False))

    good = df[df["ok"]]

    # Parametry techniczne
    print("\nCzęstotliwości próbkowania (sr):")
    print(good["sr"].value_counts())
    print("\nLiczba kanałów:")
    print(good["channels"].value_counts())
    print("\nDługość nagrań (sekundy):")
    print(good["duration_s"].describe())

    # Dokładne duplikaty (identyczna zawartość audio)
    dup_mask = good.duplicated(subset="md5", keep=False)
    print(f"\nPliki o identycznej zawartości audio: {dup_mask.sum()}")
    if dup_mask.sum() > 0:
        print(good[dup_mask].sort_values("md5")[["filepath", "md5"]].to_string(index=False))

    # Wykres rozkładu klas
    counts = df["genre"].value_counts().reindex(CLASS_NAMES)
    plt.figure(figsize=(9, 4))
    plt.bar(counts.index, counts.values)
    plt.title("Liczba nagrań w poszczególnych gatunkach (GTZAN)")
    plt.ylabel("Liczba nagrań")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=150)
    plt.show()

    # Zapis tabeli do pliku (przekazanie wyników innym osobom)
    df.to_csv(OUTPUTS_DIR / "metadata_raw.csv", index=False)
    print("\nZapisano: outputs/metadata_raw.csv i outputs/class_distribution.png")


if __name__ == "__main__":
    main()