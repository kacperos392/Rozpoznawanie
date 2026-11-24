import librosa
import torch
from transformers import pipeline

from src.config import DATA_DIR

MODEL_ID = "WasuratS/distilhubert-finetuned-gtzan"


def main():
    device = 0 if torch.cuda.is_available() else -1      # 0 = pierwsze GPU, -1 = CPU
    print("Urządzenie:", "GPU" if device == 0 else "CPU")

    clf = pipeline("audio-classification", model=MODEL_ID, device=device, top_k=10)
    print("Etykiety modelu:", clf.model.config.id2label)

    # Model oczekuje audio 16 kHz mono (nasze pliki mają 22,05 kHz: Librosa przeliczy)
    for name in ["blues.00000.wav", "classical.00001.wav", "metal.00002.wav"]:
        genre = name.split(".")[0]
        y, sr = librosa.load(DATA_DIR / genre / name, sr=16000, mono=True)
        preds = clf(y)
        top3 = [(p["label"], round(p["score"], 3)) for p in preds[:3]]
        print(f"{name}: {top3}")


if __name__ == "__main__":
    main()