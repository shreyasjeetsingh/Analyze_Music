import sqlite3
import pickle
import numpy as np

DB_PATH = "data/songs.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        curr = conn.cursor()
        curr.execute("""
            CREATE TABLE IF NOT EXISTS songs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            name TEXT,
            artist TEXT,
            features BLOB
        )
        """)
        conn.commit()

def insert_song(path, name, artist, features):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO songs(path, name, artist, features) VALUES(?, ?, ?, ?)",
            (path, name, artist, pickle.dumps(features))   
        )

def load_songs():
    with get_connection() as conn:
        rows = conn.execute("SELECT path, name, artist, features FROM songs").fetchall()
    
    paths, names, artists, vectors = [], [], [], []
    for p, n, a, b in rows:
        paths.append(p)
        names.append(n)
        artists.append(a)
        vectors.append(pickle.loads(b))
    
    if vectors:
        X = np.vstack(vectors)
    else:
        X = np.empty((0,))

    
    return paths, names, artists, X