import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, RANDOM_SEED
from src.model import build_model, compile_model


def load_split(split):
    """Wczytuje tablice cech i etykiet zapisane w Etapie 4."""
    X = np.load(PROCESSED_DIR / f"X_{split}.npy")
    y = np.load(PROCESSED_DIR / f"y_{split}.npy")
    return X, y



def train(epochs, batch_size, patience):
    keras.utils.set_random_seed(RANDOM_SEED)       # powtarzalność wyników
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    print(f"Trening: {X_train.shape}, walidacja: {X_val.shape}")

    model = compile_model(build_model())

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            MODELS_DIR / "music_model.keras",
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=2,
    )
    print(f"\nCzas treningu: {(time.time() - start) / 60:.1f} min, "
          f"wykonane epoki: {len(history.history['loss'])}")

    # Zapis historii (żeby wykresy dało się odtworzyć bez ponownego treningu)
    pd.DataFrame(history.history).to_csv(OUTPUTS_DIR / "training_history.csv", index_label="epoch")

    # Kolejność klas obok modelu: kontrakt dla aplikacji (Osoba 4)
    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f)

    return history


def plot_history(history):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    best = int(np.argmin(h["val_loss"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in [(axes[0], "loss", "Strata (loss)"),
                              (axes[1], "accuracy", "Dokładność (accuracy)")]:
        ax.plot(epochs, h[metric], label="trening")
        ax.plot(epochs, h[f"val_{metric}"], label="walidacja")
        ax.axvline(best, color="gray", linestyle="--", label=f"najlepsza epoka ({best})")
        ax.set_title(title)
        ax.set_xlabel("epoka")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "training_curves.png", dpi=150)

    print(f"\nNajlepsza epoka: {best}")
    print(f"  trening:    loss={h['loss'][best-1]:.3f}  acc={h['accuracy'][best-1]:.3f}")
    print(f"  walidacja:  loss={h['val_loss'][best-1]:.3f}  acc={h['val_accuracy'][best-1]:.3f}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    hist = train(args.epochs, args.batch_size, args.patience)
    plot_history(hist)