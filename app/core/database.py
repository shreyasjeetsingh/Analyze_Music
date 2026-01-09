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

def get_existing_paths():
    with get_connection() as conn:
        try:
            rows = conn.execute("SELECT path FROM songs").fetchall()
            return set(row[0] for row in rows)
        except sqlite3.OperationalError:
            return set()

def insert_batch(songs_data):
    #songs_data: list of tuples (path, name, artist, features)
    
    if not songs_data:
        return

    prepared_data = [
        (path, name, artist, pickle.dumps(features)) 
        for path, name, artist, features in songs_data
    ]

    with get_connection() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO songs(path, name, artist, features) VALUES(?, ?, ?, ?)",
            prepared_data
        )
        conn.commit()

def load_songs():
    with get_connection() as conn:
        try:
            rows = conn.execute("SELECT path, name, artist, features FROM songs").fetchall()
        except sqlite3.OperationalError:
            return [], [], [], np.zeros((0, 125))
    
    paths, names, artists, vectors = [], [], [], []
    for p, n, a, b in rows:
        paths.append(p)
        names.append(n)
        artists.append(a)
        vectors.append(pickle.loads(b))
    
    if vectors:
        X = np.vstack(vectors)
    else:
        X = np.zeros((0, 125)) 
    
    return paths, names, artists, X