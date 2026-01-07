import os
import librosa
from mutagen import File
from concurrent.futures import ProcessPoolExecutor

# --- CRITICAL: Make sure these imports match what is in database.py ---
from app.core.database import insert_batch, get_existing_paths
from app.core.analysis import songAnalysis

SUPPORTED = (".mp3", ".flac", ".wav", ".ogg", ".m4a")

def process_one_song(path):
    """
    Worker function that runs in a separate process.
    Returns a tuple: (path, title, artist, features) or None if failed.
    """
    try:
        # Optimization: sr=22050 standardizes sample rate for faster loading
        y, sr = librosa.load(path, sr=22050)
        
        features = songAnalysis(y, sr)

        audio = File(path, easy=True)
        title = "Unknown"
        artist = "Unknown"
        
        if audio:
            # Metadata fallback logic
            title = audio.get("title", [os.path.splitext(os.path.basename(path))[0]])[0]
            artist = audio.get("artist", ["Unknown"])[0]
        else:
            title = os.path.splitext(os.path.basename(path))[0]

        return (path, title, artist, features)

    except Exception as e:
        # print(f"Failed: {path} — {e}") 
        return None

def scan_library_parallel(folder_path, progress_callback=None):
    print("Fetching existing songs from DB...")
    
    # This calls the function you defined in database.py
    existing_paths = get_existing_paths()
    
    files_to_process = []
    print("Scanning folder structure...")
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(SUPPORTED):
                full_path = os.path.join(root, fname)
                # Skip if already in database
                if full_path not in existing_paths:
                    files_to_process.append(full_path)
    
    total = len(files_to_process)
    
    if total == 0:
        print("No new songs to add.")
        if progress_callback:
            progress_callback(0, 0)
        return

    print(f"Found {total} new songs. Starting parallel processing...")

    batch_size = 50
    current_batch = []
    
    with ProcessPoolExecutor() as executor:
        # Map returns results in the order they were submitted
        for i, result in enumerate(executor.map(process_one_song, files_to_process), 1):
            if result:
                current_batch.append(result)

            if len(current_batch) >= batch_size:
                insert_batch(current_batch)
                current_batch = []
            
            # Send progress back to GUI
            if progress_callback:
                progress_callback(i, total)

    # Insert leftovers
    if current_batch:
        insert_batch(current_batch)
        print("Final batch saved.")