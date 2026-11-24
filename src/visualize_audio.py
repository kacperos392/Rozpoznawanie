import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.config import DATA_DIR, HOP_LENGTH, OUTPUTS_DIR, SR
from src.feature_extraction import audio_to_features, load_audio, split_into_segments


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # 1) Jeden utwór: przebieg, zbliżenie na próbki i spektrogram Mel
    path = DATA_DIR / "blues" / "blues.00000.wav"
    y = load_audio(path)
    print("Liczba próbek:", len(y), "| długość [s]:", len(y) / SR)

    segments = split_into_segments(y)
    features = audio_to_features(y)
    print("Liczba fragmentów:", len(segments))
    print("Kształt cech:", features.shape, "| typ:", features.dtype)
    print("Min/max wartości:", features.min(), features.max())

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    librosa.display.waveshow(y, sr=SR, ax=axes[0])
    axes[0].set_title("Przebieg audio całego utworu (blues.00000)")

    start = SR * 10                      # 10. sekunda
    axes[1].plot(np.arange(200) / SR * 1000, y[start:start + 200])
    axes[1].set_title("Zbliżenie: 200 kolejnych próbek (ok. 9 ms)")
    axes[1].set_xlabel("czas [ms]")

    img = librosa.display.specshow(
        features[0, :, :, 0], sr=SR, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=axes[2],
    )
    axes[2].set_title("Spektrogram Mel: pierwszy 5-sekundowy fragment")
    fig.colorbar(img, ax=axes[2])

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "audio_waveform_and_mel.png", dpi=150)

    # 2) Porównanie gatunków
    genres = ["classical", "metal", "hiphop", "jazz"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, genre in zip(axes.ravel(), genres):
        y_g = load_audio(DATA_DIR / genre / f"{genre}.00001.wav")
        mel = audio_to_features(y_g)[0, :, :, 0]
        librosa.display.specshow(mel, sr=SR, hop_length=HOP_LENGTH,
                                 x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(genre)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "mel_genres_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()