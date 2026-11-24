import os
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, PRETRAINED_SR
from src.inference import load_classifier, load_windows, predict_genre
from src.youtube_audio import YouTubeError, download_audio_clip

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")


@st.cache_resource
def get_classifier():
    return load_classifier()


def analyze_upload(uploaded_file):
    """Zapisuje wgrany plik do pliku tymczasowego, analizuje go i ZAWSZE go usuwa."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded_file.getvalue())
        tmp.close()                                   # Windows: zamknąć przed odczytem
        windows, total = load_windows(tmp.name)
        probs = predict_genre(get_classifier(), windows)
        return windows, total, probs
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)



st.title("🎵 Music Genre Classifier")
def show_results(windows, total, probs, note=""):
    best = int(probs.argmax())
    st.subheader(f"Przewidywany gatunek: {CLASS_NAMES[best]}")
    st.caption(f"Analizowano {len(windows)} okien z fragmentu o długości {total:.0f} s. {note}")
    if probs[best] < 0.6:
        st.warning("Model jest niepewny (najwyższe prawdopodobieństwo poniżej 60%). "
                   "Próg jest orientacyjny.")

    st.write("Wyniki dla wszystkich klas:")
    st.bar_chart(pd.DataFrame({"prawdopodobieństwo": probs}, index=CLASS_NAMES))

    y0 = windows[0][: PRETRAINED_SR * 10]
    mel = librosa.feature.melspectrogram(y=y0, sr=PRETRAINED_SR, n_mels=128)
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(librosa.power_to_db(mel, ref=mel.max()),
                             sr=PRETRAINED_SR, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Spektrogram Mel (pierwsze 10 s)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Spektrogram służy tylko do podglądu: ten model analizuje surowy sygnał audio.")

    st.info(
        "To przewidywanie modelu wytrenowanego na zbiorze GTZAN (10 gatunków, "
        "nagrania z lat 90.), a nie pewne rozpoznanie. Wartości prawdopodobieństw "
        "nie są miarą pewności: model bywa bardzo „pewny siebie” także wtedy, gdy się myli. "
        "Przy współczesnej muzyce, nagraniach koncertowych i remiksach "
        "skuteczność może być niższa."
    )


st.title("🎵 Music Genre Classifier")
tab_file, tab_yt = st.tabs(["Wgraj plik audio", "Wklej link YouTube"])

with tab_file:
    st.write("Wgraj nagranie (WAV lub MP3), a model oszacuje jego gatunek muzyczny.")
    uploaded = st.file_uploader("Wybierz plik audio", type=["wav", "mp3"])

    if uploaded is not None:
        st.audio(uploaded)
        if st.button("Rozpoznaj gatunek", key="btn_file"):
            try:
                with st.spinner("Analizuję nagranie..."):
                    windows, total, probs = analyze_upload(uploaded)
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error("Nie udało się wczytać lub przeanalizować pliku. "
                         "Sprawdź, czy to poprawny plik WAV/MP3.")
                st.caption(f"Szczegóły techniczne: {e}")
            else:
                show_results(windows, total, probs)

with tab_yt:
    st.write("Wklej link do publicznego filmu muzycznego na YouTube.")
    st.warning(
        "Funkcja eksperymentalna, do użytku edukacyjnego i lokalnego. Pobieranie treści z YouTube "
        "może być ograniczone regulaminem serwisu i nie zawsze się powiedzie. Aplikacja pobiera "
        "tylko krótki fragment audio, nie zapisuje go trwale i usuwa po analizie. "
        "Jeśli to się nie uda, wgraj plik audio w pierwszej zakładce."
    )
    url = st.text_input("Link do filmu", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Analizuj", key="btn_yt"):
        if not url.strip():
            st.error("Wklej link do filmu.")
        else:
            try:
                with st.spinner("Pobieram fragment audio i analizuję..."):
                    with download_audio_clip(url) as (path, meta):
                        windows, total = load_windows(path)
                        probs = predict_genre(get_classifier(), windows)
            except (YouTubeError, ValueError) as e:
                st.error(str(e))
                st.caption("Możesz wgrać legalnie pozyskany plik audio w pierwszej zakładce.")
            except Exception as e:
                st.error("Wystąpił nieoczekiwany błąd podczas analizy filmu.")
                st.caption(f"Szczegóły techniczne: {e}")
            else:
                st.write(f"**Film:** {meta['title']}")
                show_results(windows, total, probs,
                             note=f"Fragment od {meta['start']:.0f}. s filmu.")