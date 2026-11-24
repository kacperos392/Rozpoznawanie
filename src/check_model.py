import numpy as np

from src.config import CLASS_NAMES, PROCESSED_DIR
from src.model import build_model, compile_model


def main():
    model = compile_model(build_model())
    model.summary()

    # Test: kilkanaście prawdziwych fragmentów walidacyjnych przez NIEWYTRENOWANY model
    X = np.load(PROCESSED_DIR / "X_val.npy", mmap_mode="r")[:16]
    y = np.load(PROCESSED_DIR / "y_val.npy")[:16]
    X = np.array(X)                                   # kopia do pamięci

    probs = model.predict(X, verbose=0)
    print("\nKształt wyjścia:", probs.shape)           # oczekujemy (16, 10)
    print("Suma prawdopodobieństw (pierwsze 3):", probs.sum(axis=1)[:3])

    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Strata niewytrenowanego modelu: {loss:.3f} (losowe zgadywanie: {np.log(len(CLASS_NAMES)):.3f})")
    print(f"Accuracy niewytrenowanego modelu: {acc:.3f}")


if __name__ == "__main__":
    main()