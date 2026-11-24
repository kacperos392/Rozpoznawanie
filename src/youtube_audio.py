import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

import yt_dlp
from yt_dlp.utils import download_range_func

from src.config import MIN_SECONDS, YT_CLIP_SECONDS, YT_MAX_VIDEO_SECONDS

ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
                 "music.youtube.com", "youtu.be"}
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeError(Exception):
    """Błąd z komunikatem czytelnym dla użytkownika."""


def extract_video_id(url):
    """Zwraca ID filmu albo None, jeśli link nie jest poprawnym linkiem YouTube."""
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https") or p.hostname not in ALLOWED_HOSTS:
        return None

    parts = [s for s in p.path.split("/") if s]
    if p.hostname == "youtu.be":
        vid = parts[0] if parts else ""
    elif p.path == "/watch":
        vid = parse_qs(p.query).get("v", [""])[0]
    elif len(parts) >= 2 and parts[0] in ("shorts", "embed"):
        vid = parts[1]
    else:
        return None
    return vid if VIDEO_ID.match(vid) else None


def explain_error(message):
    """Zamienia techniczny komunikat yt-dlp na zrozumiały dla użytkownika."""
    m = message.lower()
    if "private video" in m:
        return "Ten film jest prywatny."
    if "unavailable" in m or "removed" in m or "does not exist" in m:
        return "Film jest niedostępny (usunięty lub zablokowany w Twoim regionie)."
    if "confirm your age" in m or "age-restricted" in m or "age restricted" in m:
        return "Film ma ograniczenie wieku, więc nie można go pobrać."
    if "not a bot" in m or "403" in m:
        return ("YouTube odmówił dostępu do tego filmu (błąd ochrony przed automatami lub 403). "
                "Spróbuj później, zaktualizuj yt-dlp albo wgraj plik audio.")
    if "drm" in m or "premium" in m:
        return "Ten film jest chroniony i nie może zostać pobrany."
    if "javascript runtime" in m or "deno" in m:
        return "Brakuje środowiska JavaScript (Deno) wymaganego przez yt-dlp."
    return "Nie udało się pobrać audio z tego filmu."

@contextmanager
def download_audio_clip(url):
    """Pobiera krótki fragment audio do katalogu tymczasowego.
    Zwraca (ścieżka do WAV, metadane). Katalog jest usuwany po wyjściu z bloku `with`."""
    vid = extract_video_id(url)
    if vid is None:
        raise YouTubeError("To nie wygląda na poprawny link do filmu na YouTube. "
                           "Przykład: https://www.youtube.com/watch?v=...")
    if shutil.which("ffmpeg") is None:
        raise YouTubeError("Brak programu ffmpeg w systemie. Zainstaluj go albo wgraj plik audio.")

    canonical = f"https://www.youtube.com/watch?v={vid}"     # bez parametrów playlisty
    base = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "socket_timeout": 20, "retries": 2}

    # 1) Metadane (bez pobierania)
    try:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(canonical, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise YouTubeError(explain_error(str(e))) from e

    if info.get("is_live"):
        raise YouTubeError("To transmisja na żywo. Nie można jej analizować.")
    duration = info.get("duration") or 0
    if duration < MIN_SECONDS:
        raise YouTubeError("Film jest zbyt krótki do analizy.")
    if duration > YT_MAX_VIDEO_SECONDS:
        raise YouTubeError(f"Film jest zbyt długi (limit {YT_MAX_VIDEO_SECONDS // 60} min).")

    clip = min(YT_CLIP_SECONDS, duration)
    start = max(0.0, (duration - clip) / 2)                  # fragment ze środka

    # 2) Pobranie tylko wybranego fragmentu do katalogu tymczasowego
    with tempfile.TemporaryDirectory(prefix="mgc_") as tmpdir:
        opts = {
            **base,
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "clip.%(ext)s"),
            "download_ranges": download_range_func(None, [(start, start + clip)]),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([canonical])
        except yt_dlp.utils.DownloadError as e:
            raise YouTubeError(explain_error(str(e))) from e

        path = os.path.join(tmpdir, "clip.wav")
        if not os.path.exists(path):
            raise YouTubeError("Nie udało się przygotować pliku audio.")

        yield path, {"title": info.get("title", ""), "duration": duration,
                     "start": start, "clip": clip}