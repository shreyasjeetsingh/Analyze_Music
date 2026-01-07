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

# --- REQUIRED FOR SCANNER ---
def get_existing_paths():
    """Returns a set of all file paths currently in the database."""
    with get_connection() as conn:
        # We verify if the table exists first to avoid crashing on a fresh DB
        try:
            rows = conn.execute("SELECT path FROM songs").fetchall()
            return set(row[0] for row in rows)
        except sqlite3.OperationalError:
            return set()

# --- REQUIRED FOR SCANNER ---
def insert_batch(songs_data):
    """
    songs_data: list of tuples (path, name, artist, features)
    """
    if not songs_data:
        return

    # Serialize features for all songs in the batch
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
            # Table doesn't exist yet
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
        # Ensure this matches your feature vector size (125 based on your previous code)
        X = np.zeros((0, 125)) 
    
    return paths, names, artists, X