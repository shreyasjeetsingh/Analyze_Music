import tkinter as tk
from tkinter import filedialog
import librosa
from app.core.analysis import songAnalysis
from app.core.database import load_songs
from app.core.insert import insert_folder
from app.core.similarity import compute_similarity

class MainView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.paths, self.names, self.artists, self.X = load_songs()

        tk.Label(self, text="Song name: ").pack()
        self.entry = tk.Entry(self, width=40)
        self.entry.pack(pady=5)
        self.entry.bind("<Return>", lambda event: self.search())

        tk.Button(self, text="Find Similar", command=self.search).pack(pady=10)
        self.output = tk.Text(self, height=12)
        self.output.pack(fill="both", expand=True)

        tk.Button(self, text="Add Music Folder", command=self.add_folder).pack(pady=5)

    def search(self):
        song = self.entry.get()

        self.output.delete("1.0", "end")
        self.output.insert("end", "Finding similar songs...\n")
        self.update_idletasks()

        if song not in self.names:
            self.output.delete("1.0", "end")
            self.output.insert("end", "Song not found\n")
            return
        
        idx = self.names.index(song)
        path = self.paths[idx]

        y, sr = librosa.load(path)
        q = songAnalysis(y, sr)

        sims = compute_similarity(q, self.X)
        order = sims.argsort()[::-1]

        self.output.delete("1.0", "end")
        self.output.insert("end", f"Query: {song}\n\n")
        for i in order[:5]:
            self.output.insert("end", f"{self.artists[i]} - {self.names[i]} ({sims[i]:.3f})\n")
    
    def add_folder(self):
        folder = filedialog.askdirectory(title="Select Music Folder")
        if not folder:
            return

        self.output.delete("1.0", "end")
        self.output.insert("end", f"Scanning folder:\n{folder}\n\nPlease wait...")
        self.update_idletasks()

        insert_folder(folder)

        self.names, self.artists, self.paths, self.X = load_songs()

        self.output.delete("1.0", "end")
        self.output.insert("end", "Done adding songs.\n")