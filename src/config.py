from pathlib import Path

# Główny folder projektu (rodzic folderu src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data" / "genres_original"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Kolejność klas jest STAŁA w całym projekcie
# (indeks w tej liście = numer klasy w modelu)
CLASS_NAMES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# --- Parametry przetwarzania audio (wspólne dla treningu i aplikacji) ---
SR = 22050              # częstotliwość próbkowania [Hz]
N_MELS = 128            # liczba pasm Mel
N_FFT = 2048            # długość okna FFT [próbki]
HOP_LENGTH = 512        # przesunięcie okna [próbki]
SEGMENT_SECONDS = 5.0   # długość jednego fragmentu [s]
TOP_DB = 80.0           # zakres dynamiki w dB (do normalizacji)

SEGMENT_SAMPLES = int(SR * SEGMENT_SECONDS)   # 110250 próbek


# --- Podział danych ---
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Przygotowane tablice (duże pliki, poza repozytorium Git)
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

# --- Parametry modelu ---
N_FRAMES = 1 + SEGMENT_SAMPLES // HOP_LENGTH     # liczba kolumn czasu w spektrogramie (216)
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)              # (128, 216, 1)
NUM_CLASSES = len(CLASS_NAMES)                   # 10

# --- Gotowy model (aplikacja) ---
PRETRAINED_MODEL_ID = "WasuratS/distilhubert-finetuned-gtzan"
PRETRAINED_SR = 16000          # częstotliwość próbkowania oczekiwana przez model
WINDOW_SECONDS = 30.0          # długość okna analizy
MAX_WINDOWS = 3                # maks. liczba okien dla długich nagrań
MIN_SECONDS = 5.0              # minimalna długość nagrania


# --- YouTube (moduł eksperymentalny) ---
YT_CLIP_SECONDS = 90           # tyle sekund audio pobieramy (ze środka filmu)
YT_MAX_VIDEO_SECONDS = 3600    # dłuższych filmów nie analizujemy