import librosa
import numpy as np
import sqlite3
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from insertion import songAnalysis

def load_database():
    conn = sqlite3.connect("songs.db")
    cur = conn.cursor()
    cur.execute("SELECT name, artist, features FROM songs")
    rows = cur.fetchall()
    conn.close()

    names = []
    artists=[]
    vectors = []
    for name, artist, blob in rows:
        names.append(name)
        artists.append(artist)
        vectors.append(pickle.loads(blob))

    X = np.vstack(vectors)
    return names, artists, X


names, artists, X = load_database()

mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma==0] = 1.0
X_norm = (X - mu) / sigma

song_input = input("Enter song name: ")
conn = sqlite3.connect("songs.db")
cur = conn.cursor()
cur.execute("SELECT path, artist FROM songs WHERE name = ?", (song_input,))
rows = cur.fetchall()
conn.close()

query_path = rows[0][0]
artist_name = rows[0][1]

y, sr = librosa.load(query_path)
q = songAnalysis(y, sr)
q_norm = (q - mu) / sigma


sims = cosine_similarity([q_norm], X_norm)[0]

idx = np.argsort(sims)[::-1]

print("Query:", song_input, "by", artist_name)
print("\nMost Similar Songs:")
for i in idx[:5]:
    print(f"{artists[i]} - {names[i]} : similarity = {sims[i]:.3f}")