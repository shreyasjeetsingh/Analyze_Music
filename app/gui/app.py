import tkinter as tk
from app.gui.main_view import MainView

def start():
    root = tk.Tk()
    root.title("Music Similarity")
    root.geometry("600x400")
    MainView(root).pack(fill="both", expand=True)
    root.mainloop()