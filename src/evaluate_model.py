import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from src.config import CLASS_NAMES, OUTPUTS_DIR


def evaluate(name, csv_name, png_name):
    path = OUTPUTS_DIR / csv_name
    if not path.exists():
        print(f"Brak pliku {csv_name}, pomijam.")
        return None

    df = pd.read_csv(path)
    P = df[[f"p_{g}" for g in CLASS_NAMES]].values
    y_pred = np.array(CLASS_NAMES)[P.argmax(axis=1)]
    y_true = df["true"].values

    acc = accuracy_score(y_true, y_pred)
    margin = 1.96 * np.sqrt(acc * (1 - acc) / len(df))     # przybliżony przedział 95%
    print(f"\n===== {name} =====")
    print(f"Accuracy (na poziomie utworów): {acc:.3f} ± {margin:.3f}  (n = {len(df)})")
    print(classification_report(y_true, y_pred, labels=CLASS_NAMES, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    pairs = [(cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
             for i in range(len(CLASS_NAMES)) for j in range(len(CLASS_NAMES))
             if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    print("Najczęstsze pomyłki (prawdziwy -> przewidziany):")
    for count, true_g, pred_g in pairs[:6]:
        print(f"   {true_g} -> {pred_g}: {count}")

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Macierz pomyłek: {name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / png_name, dpi=150)
    return acc


def main():
    evaluate("Własny CNN", "preds_cnn_test.csv", "confusion_cnn.png")
    evaluate("Gotowy model (DistilHuBERT)", "preds_pretrained_test.csv", "confusion_pretrained.png")
    plt.show()


if __name__ == "__main__":
    main()