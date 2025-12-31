import librosa
import numpy as np
import sqlite3
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from insertion import songAnalysis

def load_database():
    conn = sqlite3.connect("songs.db")
    cur = conn.cursor()
    cur.execute("SELECT path, features FROM songs")
    rows = cur.fetchall()
    conn.close()

    paths = []
    vectors = []
    for path, blob in rows:
        paths.append(path)
        vectors.append(pickle.loads(blob))

    X = np.vstack(vectors)
    return paths, X


paths, X = load_database()

mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma==0] = 1.0
X_norm = (X - mu) / sigma


query_path = r"SomeSongs\06 - Love Is Lonely.flac"
y, sr = librosa.load(query_path)
q = songAnalysis(y, sr)
q_norm = (q - mu) / sigma


sims = cosine_similarity([q_norm], X_norm)[0]

idx = np.argsort(sims)[::-1]

print("Query:", query_path)
print("\nMost Similar Songs:")
for i in idx[:5]:
    print(f"{paths[i]} - similarity = {sims[i]:.3f}")