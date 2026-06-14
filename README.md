# Analyze Music
 
A Python tool that builds playlists based on the "vibe" of a single song you pick. Give it one track, and it analyzes audio features to find and recommend similar songs from your local music library.
 
## How It Works
 
1. Scans your local music library and reads track metadata.
2. Extracts audio feature vectors for each track (e.g. tempo, timbre, energy).
3. Stores track info and features in a local database.
4. When you pick a song, computes cosine similarity between its feature vector and all other tracks.
5. Returns the top 10 most similar songs — instantly giving you a playlist that matches the vibe of your chosen track.
## Features
 
- Pick any song from your library as a "seed" track
- Audio feature extraction for similarity comparison
- Local database for storing track metadata and features
- Cosine similarity-based recommendation engine
- Simple GUI for interacting with the app
## Project Structure
 
```
Analyze_Music/
├── app/                # Core application code (GUI, analysis logic)
├── main.py             # Entry point — launches the app
├── requirements.txt    # Python dependencies
└── .gitignore
```
 
## Getting Started
 
### Prerequisites
 
- Python 3.x
- pip
### Installation
 
```bash
git clone https://github.com/shreyasjeetsingh/Analyze_Music.git
cd Analyze_Music
pip install -r requirements.txt
```
 
### Usage
 
```bash
python main.py
```
 
This launches the GUI, where you can select a song and generate a list of similar tracks.
