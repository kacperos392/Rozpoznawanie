import librosa
import pandas as pd
import torch
from transformers import pipeline

from src.config import CLASS_NAMES, OUTPUTS_DIR, PROJECT_ROOT

MODEL_ID = "WasuratS/distilhubert-finetuned-gtzan"


def main():
    meta = pd.read_csv(OUTPUTS_DIR / "metadata_split.csv")
    test = meta[meta["split"] == "test"].reset_index(drop=True)

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("audio-classification", model=MODEL_ID, device=device,
                   top_k=len(CLASS_NAMES))

    rows = []
    for i, row in enumerate(test.itertuples(), start=1):
        y, _ = librosa.load(PROJECT_ROOT / row.filepath, sr=16000, mono=True)
        scores = {p["label"]: p["score"] for p in clf(y)}

        rec = {"track_id": row.track_id, "filepath": row.filepath, "true": row.genre}
        rec.update({f"p_{g}": scores[g] for g in CLASS_NAMES})   # stała kolejność klas
        rows.append(rec)

        if i % 25 == 0:
            print(f"{i}/{len(test)} utworów...")

    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "preds_pretrained_test.csv", index=False)
    print("Zapisano: outputs/preds_pretrained_test.csv")


if __name__ == "__main__":
    main()