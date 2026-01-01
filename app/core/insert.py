import os
import librosa
from mutagen import File
from app.core.analysis import songAnalysis
from app.core.database import insert_song

SUPPORTED = (".mp3", ".flac", ".wav", ".ogg")

def insert_folder(folder):
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith(SUPPORTED):
                continue

            path = os.path.join(root, fname)

            try:
                y, sr = librosa.load(path)
                features = songAnalysis(y, sr)

                audio = File(path, easy=True)
                title = audio.get("title", [os.path.basename(path)])[0] if audio else os.path.basename(path)
                artist = audio.get("artist", ["Unknown"])[0] if audio else "Unknown"

                insert_song(path, title, artist, features)
            except Exception as e:
                print(f"Failed: {path} — {e}")