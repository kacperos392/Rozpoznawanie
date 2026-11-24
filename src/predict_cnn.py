import argparse

import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR


def main(model_path):
    X = np.load(PROCESSED_DIR / "X_test.npy")
    ids = np.load(PROCESSED_DIR / "ids_test.npy")

    model = keras.models.load_model(model_path)
    probs = model.predict(X, batch_size=64, verbose=0)          # (fragmenty, 10)

    # Uśrednienie przewidywań fragmentów w obrębie każdego utworu
    df = pd.DataFrame(probs, columns=[f"p_{g}" for g in CLASS_NAMES])
    df["track_id"] = ids
    per_track = df.groupby("track_id").mean().reset_index()

    meta = pd.read_csv(OUTPUTS_DIR / "metadata_split.csv")
    meta = meta[meta["split"] == "test"][["track_id", "filepath", "genre"]]
    meta = meta.rename(columns={"genre": "true"})
    out = meta.merge(per_track, on="track_id")

    out.to_csv(OUTPUTS_DIR / "preds_cnn_test.csv", index=False)
    print(f"Utworów: {len(out)}, fragmentów: {len(X)}")
    print("Zapisano: outputs/preds_cnn_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODELS_DIR / "music_model.keras"))
    args = parser.parse_args()
    main(args.model)