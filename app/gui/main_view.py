import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from app.core.database import init_db, load_songs
from app.core.similarity import compute_similarity
from app.core.insert import scan_library_parallel

class MainView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.paths = []
        self.names = []
        self.artists = []
        self.X = None

        init_db()
        self.refresh_data()
        self.setup_ui()

    def setup_ui(self):
        top_frame = tk.Frame(self)
        top_frame.pack(fill="x", pady=10, padx=10)

        self.btn_scan = tk.Button(top_frame, text="📂 Scan Music Folder", command=self.start_scan_thread)
        self.btn_scan.pack(side="left")

        self.lbl_progress = tk.Label(top_frame, text="Ready", fg="blue", font=("Consolas", 10))
        self.lbl_progress.pack(side="left", padx=15)

        self.lbl_total = tk.Label(top_frame, text=f"Total: {len(self.names)} songs", fg="gray")
        self.lbl_total.pack(side="right")

        tk.Frame(self, height=2, bd=1, relief="sunken").pack(fill="x", padx=5, pady=5)

        search_frame = tk.Frame(self)
        search_frame.pack(pady=10)

        tk.Label(search_frame, text="Song name: ").pack(side="left")
        self.entry = tk.Entry(search_frame, width=40)
        self.entry.pack(side="left", padx=5)
        self.entry.bind("<Return>", lambda event: self.search())

        self.btn_search = tk.Button(search_frame, text="Find Similar", command=self.search)
        self.btn_search.pack(side="left")

        self.output = tk.Text(self, height=15)
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh_data(self):
        self.paths, self.names, self.artists, self.X = load_songs()
        if hasattr(self, 'lbl_total'):
            self.lbl_total.config(text=f"Total: {len(self.names)} songs")

    def start_scan_thread(self):
        folder_path = filedialog.askdirectory(title="Select Music Folder")
        if not folder_path: return

        self.btn_scan.config(state="disabled")
        self.btn_search.config(state="disabled")
        self.lbl_progress.config(text="Initializing scan...", fg="orange")
        
        thread = threading.Thread(target=self.run_scan, args=(folder_path,))
        thread.daemon = True
        thread.start()

    def run_scan(self, folder_path):
        def on_progress(current, total):
            self.parent.after(0, lambda: self.update_progress_text(current, total))
        
        try:
            scan_library_parallel(folder_path, progress_callback=on_progress)
            
            self.parent.after(0, self.on_scan_complete)
            
        except Exception as e:
            print(f"Error: {e}")
            self.parent.after(0, self.reset_ui_state)

    def update_progress_text(self, current, total):
        current_batch = (current // 50) + 1
        
        self.lbl_progress.config(
            text=f"Processing: {current} / {total} songs (Batch {current_batch})",
            fg="blue"
        )

    def on_scan_complete(self):
        self.refresh_data()
        self.lbl_progress.config(text="Scan Complete ✔", fg="green")
        messagebox.showinfo("Success", f"Scan complete! Library now has {len(self.names)} songs.")
        self.reset_ui_state()

    def reset_ui_state(self):
        self.btn_scan.config(state="normal")
        self.btn_search.config(state="normal")
        if self.lbl_progress.cget("text") != "Scan Complete ✔":
            self.lbl_progress.config(text="Ready", fg="black")

    def search(self):
        song_query = self.entry.get().strip()
        if not song_query: return

        matches = [i for i, name in enumerate(self.names) if song_query.lower() == name.lower()]

        if not matches:
            matches = [i for i, name in enumerate(self.names) if song_query in name.lower()]

        if not matches:
            self.output.delete("1.0", "end")
            self.output.insert("end", f"Song '{song_query}' not found.\n")
            return
        
        idx = matches[0] 
        actual_name = self.names[idx]
        actual_artist = self.artists[idx]
        
        q = self.X[idx]
        sims = compute_similarity(q, self.X)
        order = sims.argsort()[::-1]

        self.output.delete("1.0", "end")
        self.output.insert("end", f"Query: {actual_artist} - {actual_name}\n")
        self.output.insert("end", "="*50 + "\n")

        count = 0
        for i in order:
            if i == idx: continue

            score = sims[i]
            label = ""
            if score > 0.90: label = " (Twin)"
            elif score > 0.80: label = " (Great)"
            elif score > 0.65: label = " (Good)"

            self.output.insert("end", f"{self.artists[i]} - {self.names[i]} [{score:.3f}]{label}\n")
            count += 1
            if count >= 10: break