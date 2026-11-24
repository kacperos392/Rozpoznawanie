import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE


def clean_metadata(df):
    """Usuwa uszkodzone pliki i duplikaty (na podstawie hasha zawartości audio)."""
    n_start = len(df)

    # 1) uszkodzone pliki (kolumna 'ok' z Etapu 2)
    df = df[df["ok"]].copy()
    print(f"Usunięto uszkodzonych: {n_start - len(df)}")

    # 2) ten sam dźwięk z RÓŻNYMI etykietami: niejednoznaczne, usuwamy wszystkie kopie
    n_genres = df.groupby("md5")["genre"].transform("nunique")
    conflict = df[n_genres > 1]
    print(f"Usunięto z konfliktem etykiet: {len(conflict)}")
    print(conflict[["filepath", "genre"]].to_string(index=False))
    df = df[n_genres == 1]

    # 3) identyczne kopie w tej samej klasie: zostawiamy jedną
    n_before = len(df)
    df = df.drop_duplicates(subset="md5", keep="first")
    print(f"Usunięto zdublowanych kopii: {n_before - len(df)}")

    print(f"Utworów po czyszczeniu: {len(df)}")
    return df.reset_index(drop=True)

def split_tracks(df):
    """Dzieli UTWORY (nie fragmenty) na train/val/test z zachowaniem proporcji klas."""
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["genre"], random_state=RANDOM_SEED
    )
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)   # udział walidacji w pozostałych danych
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val["genre"],
        random_state=RANDOM_SEED,
    )

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")

    out = pd.concat([train, val, test]).sort_values("filepath").reset_index(drop=True)
    out["track_id"] = out.index          # unikalny numer utworu (przyda się w Etapie 7)
    return out


def main():
    df = pd.read_csv(OUTPUTS_DIR / "metadata_raw.csv")
    df = clean_metadata(df)
    df = split_tracks(df)

    # --- Kontrole jakości podziału ---
    print("\nLiczba utworów w zbiorach i klasach:")
    print(pd.crosstab(df["genre"], df["split"], margins=True))

    sets = {name: set(g["filepath"]) for name, g in df.groupby("split")}
    hashes = {name: set(g["md5"]) for name, g in df.groupby("split")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (sets[a] & sets[b]), f"Wspólne pliki: {a} i {b}"
        assert not (hashes[a] & hashes[b]), f"Identyczne nagrania w {a} i {b}"
    print("\nKontrola: żaden utwór ani identyczne audio nie występuje w dwóch zbiorach. OK")

    cols = ["track_id", "filepath", "genre", "filename", "md5", "split"]
    df[cols].to_csv(OUTPUTS_DIR / "metadata_split.csv", index=False)
    print("Zapisano: outputs/metadata_split.csv")


if __name__ == "__main__":
    main()