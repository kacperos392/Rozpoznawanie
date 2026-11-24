import os
import tempfile
from html import escape

import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from src.config import CLASS_NAMES, PRETRAINED_SR
from src.inference import load_classifier, load_windows, predict_genre
from src.youtube_audio import YouTubeError, download_audio_clip, extract_video_id

st.set_page_config(page_title="Muzyczne porównanie gatunków", page_icon="🎵",
                   layout="wide")

st.markdown(
    """
    <style>
    html, body { background: #10121f !important; }
    .stApp {
        background: transparent;
        color: #f4f3fa;
    }
    [data-testid="stHeader"] {
        background: rgba(18, 19, 34, 0.55);
        z-index: 2;
    }
    [data-testid="stAppDeployButton"] { display: none !important; }
    [data-testid="stMain"] { position: relative; z-index: 1; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        position: relative;
        z-index: 1;
        border-radius: 22px;
        background: linear-gradient(90deg, rgba(16, 18, 31, .76), rgba(16, 18, 31, .62));
        backdrop-filter: blur(1.5px);
    }
    .artist-backdrop {
        position: fixed;
        z-index: 0;
        inset: 0;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        overflow: hidden;
        pointer-events: none;
    }
    .artist-slide {
        position: relative;
        min-width: 0;
        height: 100vh;
        overflow: hidden;
    }
    .artist-track {
        display: flex;
        height: 500vh;
        flex-direction: column;
        animation: vertical-slideshow 28s ease-in-out infinite;
    }
    .artist-slide:nth-child(2) .artist-track { animation-delay: -7s; }
    .artist-slide:nth-child(3) .artist-track { animation-delay: -14s; }
    .artist-slide:nth-child(4) .artist-track { animation-delay: -21s; }
    .artist-frame {
        position: relative;
        flex: 0 0 100vh;
        height: 100vh;
        overflow: hidden;
    }
    .artist-frame img {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: grayscale(30%) brightness(.58);
        transform: scale(1.025);
    }
    .artist-frame::after {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(17, 18, 31, .25), rgba(17, 18, 31, .82));
        content: "";
    }
    @keyframes vertical-slideshow {
        0%, 15% { transform: translateY(0); }
        20%, 35% { transform: translateY(-20%); }
        40%, 55% { transform: translateY(-40%); }
        60%, 75% { transform: translateY(-60%); }
        80%, 100% { transform: translateY(-80%); }
    }
    h1, h2, h3 { color: #f8f6ff; }
    p, label, [data-testid="stCaptionContainer"] { color: #c4c3d5; }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(30, 31, 54, 0.78);
        border-color: #3a3b59;
        border-radius: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #3a3b59;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #66e0cc;
        border-bottom-color: #66e0cc;
    }
    .stTextInput input {
        background: #1e2037;
        color: #fff;
        border-color: #484966;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(110deg, #ff876f, #f16d85);
        border-color: transparent;
        border-radius: 10px;
        color: #171323;
        font-weight: 700;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(110deg, #ffa18b, #ff8296);
        border-color: transparent;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding-right: 1rem;
            padding-left: 1rem;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        .artist-track { animation: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CLASSIC_TRACKS = {
    "blues": {"title": "B.B. King — The Thrill Is Gone", "video_id": "4fk2prKnYnI"},
    "classical": {"title": "Beethoven — Moonlight Sonata", "video_id": "4Tr0otuiQuU"},
    "country": {"title": "Johnny Cash — Ring of Fire", "video_id": "1WaV2x8GXj0"},
    "disco": {"title": "Bee Gees — Stayin' Alive", "video_id": "I_izvAbhExY"},
    "hiphop": {"title": "Grandmaster Flash — The Message", "video_id": "PobrSpMwKk4"},
    "jazz": {"title": "Dave Brubeck — Take Five", "video_id": "vmDDOFXSgAs"},
    "metal": {"title": "Black Sabbath — Paranoid", "video_id": "0qanF-91aJo"},
    "pop": {"title": "Whitney Houston — I Will Always Love You", "video_id": "3JWTaaS7LdU"},
    "reggae": {"title": "Bob Marley — Three Little Birds", "video_id": "zaGUr6wzyT8"},
    "rock": {"title": "Queen — Bohemian Rhapsody", "video_id": "fJ9rUzIMcZQ"},
}

ARTIST_PANELS = [
    {
        "name": "B.B. King",
        "image": "https://upload.wikimedia.org/wikipedia/commons/5/52/B.B._King%2C_2009.jpg",
        "credit": "Werner100359 · CC BY-SA 3.0",
        "source": "https://commons.wikimedia.org/wiki/File:B.B._King,_2009.jpg",
    },
    {
        "name": "Beethoven",
        "image": "https://upload.wikimedia.org/wikipedia/commons/5/57/Anonymous_-_Gem%C3%A4lde%2C_Portrait_Ludwig_van_Beethoven_-_SAM_1000_-_Kunsthistorisches_Museum.jpg",
        "credit": "F.G. Waldmüller · domena publiczna",
        "source": "https://commons.wikimedia.org/wiki/File:Anonymous_-_Gem%C3%A4lde,_Portrait_Ludwig_van_Beethoven_-_SAM_1000_-_Kunsthistorisches_Museum.jpg",
    },
    {
        "name": "Johnny Cash",
        "image": "https://upload.wikimedia.org/wikipedia/commons/c/c5/1971_-_Johnny_Cash_-_Allentown_fair_-_Allentown_PA.jpg",
        "credit": "Lehigh County Agricultural Society · domena publiczna",
        "source": "https://commons.wikimedia.org/wiki/File:1971_-_Johnny_Cash_-_Allentown_fair_-_Allentown_PA.jpg",
    },
    {
        "name": "Bob Marley",
        "image": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Bob-Marley.jpg",
        "credit": "Eddie Mallin · CC BY 2.0",
        "source": "https://commons.wikimedia.org/wiki/File:Bob-Marley.jpg",
    },
]


artist_sequence = ARTIST_PANELS + [ARTIST_PANELS[0]]
artist_frames = "".join(
    '<div class="artist-frame">'
    f'<img src="{escape(artist["image"], quote=True)}" '
    'alt="">'
    '</div>'
    for artist in artist_sequence
)
artist_slides = "".join(
    '<section class="artist-slide">'
    f'<div class="artist-track">{artist_frames}</div></section>'
    for _ in ARTIST_PANELS
)
st.markdown(
    f'<div class="artist-backdrop" aria-hidden="true">{artist_slides}</div>',
    unsafe_allow_html=True,
)


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



def show_comparison(video_id, song_title, genre):
    classic = CLASSIC_TRACKS[genre]
    st.subheader("Porównaj brzmienie")
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Twój utwór")
            st.caption(song_title)
            st.video(f"https://www.youtube.com/watch?v={video_id}")

    with right:
        with st.container(border=True):
            st.markdown(f"#### Klasyk: {genre.capitalize()}")
            st.caption(classic["title"])
            st.video(f"https://www.youtube.com/watch?v={classic['video_id']}")


def show_animated_probabilities(probs):
    rows = []
    best = int(probs.argmax())
    for index, (genre, probability) in enumerate(zip(CLASS_NAMES, probs)):
        percent = min(100.0, max(0.0, float(probability) * 100))
        fill_class = "fill-best" if index == best else "fill"
        rows.append(
            f'<div class="column">'
            f'<span class="percent">0%</span>'
            f'<div class="track"><div class="{fill_class}" '
            f'data-value="{percent:.2f}"></div></div>'
            f'<span class="genre">{escape(genre.capitalize())}</span>'
            f'</div>'
        )

    chart = """
    <!doctype html>
    <html>
    <head>
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0;
          padding: 18px 20px;
          color: #f4f3fa;
          background: #191a2d;
          font: 14px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .heading { margin: 0 0 14px; color: #c4c3d5; font-size: 13px; }
        .chart {
          display: grid;
          grid-template-columns: repeat(10, minmax(0, 1fr));
          align-items: end;
          gap: clamp(4px, 1.2vw, 14px);
          height: 360px;
        }
        .column {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 9px;
          height: 100%;
          min-width: 0;
        }
        .genre {
          min-height: 32px;
          color: #deddec;
          font-size: clamp(9px, 1.1vw, 12px);
          text-align: center;
          overflow-wrap: anywhere;
        }
        .percent {
          height: 18px;
          color: #f4f3fa;
          font-size: 12px;
          font-weight: 700;
          text-align: center;
          font-variant-numeric: tabular-nums;
        }
        .track {
          position: relative;
          flex: 1;
          width: min(100%, 68px);
          overflow: hidden;
          border: 1px solid #444563;
          border-radius: 14px;
          background: linear-gradient(180deg, #24263e, #30314b);
          box-shadow: inset 0 2px 8px rgba(0, 0, 0, .24);
        }
        .fill, .fill-best {
          position: absolute;
          right: 0;
          bottom: 0;
          left: 0;
          height: 0;
          border-radius: inherit;
          transition: height 750ms cubic-bezier(.18,.75,.22,1);
        }
        .fill { background: linear-gradient(0deg, #786df5, #66e0cc); }
        .fill-best { background: linear-gradient(0deg, #ff876f, #ffc477); }
      </style>
    </head>
    <body>
      <p class="heading">Prawdopodobieństwo dla każdego gatunku</p>
      <div class="chart">__ROWS__</div>
      <script>
        document.querySelectorAll(".column").forEach((column) => {
          const bar = column.querySelector(".fill, .fill-best");
          const counter = column.querySelector(".percent");
          const target = Number(bar.dataset.value);
          const reducedMotion = window.matchMedia(
            "(prefers-reduced-motion: reduce)"
          ).matches;
          if (reducedMotion) {
            bar.style.height = target + "%";
            counter.textContent = Math.round(target) + "%";
            return;
          }

          const randomStart = 12 + Math.random() * 76;
          const revealDuration = 1700;
          const settleDelay = 2800;
          const settleDuration = 5000;

          function countBetween(from, to, duration) {
            const start = performance.now();
            function animate(now) {
              const progress = Math.min((now - start) / duration, 1);
              const eased = 1 - Math.pow(1 - progress, 3);
              counter.textContent =
                Math.round(from + (to - from) * eased) + "%";
              if (progress < 1) requestAnimationFrame(animate);
            }
            requestAnimationFrame(animate);
          }

          requestAnimationFrame(() => {
            bar.style.height = randomStart + "%";
            countBetween(0, randomStart, revealDuration);
          });
          setTimeout(() => {
            bar.style.transition =
              "height " + settleDuration + "ms cubic-bezier(.18,.75,.22,1)";
            bar.style.height = target + "%";
            countBetween(randomStart, target, settleDuration);
          }, settleDelay);
        });
      </script>
    </body>
    </html>
    """
    components.html(chart.replace("__ROWS__", "".join(rows)), height=455, scrolling=False)


def show_results(windows, total, probs, note=""):
    best = int(probs.argmax())
    st.subheader(f"Przewidywany gatunek: {CLASS_NAMES[best]}")
    st.caption(f"Analizowano {len(windows)} okien z fragmentu o długości {total:.0f} s. {note}")
    if probs[best] < 0.6:
        st.warning("Model jest niepewny (najwyższe prawdopodobieństwo poniżej 60%). "
                   "Próg jest orientacyjny.")

    show_animated_probabilities(probs)

    y0 = windows[0][: PRETRAINED_SR * 10]
    mel = librosa.feature.melspectrogram(y=y0, sr=PRETRAINED_SR, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=mel.max())
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#191a2d")
    ax.set_facecolor("#191a2d")
    image = librosa.display.specshow(
        mel_db, sr=PRETRAINED_SR, x_axis="time", y_axis="mel",
        cmap="magma", vmin=-65, vmax=0, ax=ax,
    )
    ax.set_title("Jak wygląda brzmienie?", color="#f4f3fa", fontsize=15, pad=14)
    ax.set_xlabel("Czas", color="#c4c3d5")
    ax.set_ylabel("Częstotliwość (skala Mel)", color="#c4c3d5")
    ax.tick_params(colors="#c4c3d5")
    for spine in ax.spines.values():
        spine.set_color("#41425f")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, format="%+2.0f dB")
    colorbar.set_label("Poziom względem maksimum", color="#c4c3d5")
    colorbar.ax.tick_params(colors="#c4c3d5")
    colorbar.outline.set_edgecolor("#41425f")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "Jaśniejsze pasma oznaczają mocniejszy dźwięk. Spektrogram jest podglądem — "
        "model analizuje surowy sygnał audio."
    )

    st.info(
        "To przewidywanie modelu wytrenowanego na zbiorze GTZAN (10 gatunków, "
        "nagrania z lat 90.), a nie pewne rozpoznanie. Wartości prawdopodobieństw "
        "nie są miarą pewności: model bywa bardzo „pewny siebie” także wtedy, gdy się myli. "
        "Przy współczesnej muzyce, nagraniach koncertowych i remiksach "
        "skuteczność może być niższa."
    )


st.markdown(
    "<h1 style='margin:.2rem 0'>Poznaj gatunek utworu </h1>",
    unsafe_allow_html=True,
)
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
    st.caption(
        "Odtwarzacz działa bezpośrednio w aplikacji. Do analizy pobierany jest tylko krótki "
        "fragment; jest usuwany po zakończeniu."
    )
    url = st.text_input("Link do filmu", placeholder="https://www.youtube.com/watch?v=...")
    video_id = extract_video_id(url) if url.strip() else None

    analyze_clicked = st.button("Analizuj utwór", key="btn_yt", type="primary")
    if analyze_clicked:
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
                show_results(windows, total, probs,
                             note=f"Fragment od {meta['start']:.0f}. s filmu.")
                if video_id is not None:
                    show_comparison(video_id, meta["title"], CLASS_NAMES[int(probs.argmax())])
    elif video_id is not None:
        st.markdown("### Podgląd utworu")
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    elif url.strip():
        st.caption("Podaj poprawny, publiczny link do filmu YouTube, aby wyświetlić podgląd.")

with st.expander("Źródła i licencje grafik w tle"):
    for artist in ARTIST_PANELS:
        st.markdown(
            f"[{artist['name']}]({artist['source']}) — {artist['credit']}"
        )