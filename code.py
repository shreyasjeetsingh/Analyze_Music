import librosa
import numpy as np

def songAnalysis(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo.item()

    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    mean_tempogram = np.mean(tempogram, axis=1)
    std_tempogram = np.std(tempogram, axis=1)
    
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    
    M = librosa.feature.mfcc(y=y, sr=sr)
    mean_M = np.mean(M, axis=1)
    std_M = np.std(M, axis=1)
    
    delta_M = librosa.feature.delta(M)
    mean_delta_M = np.mean(delta_M, axis=1)
    std_delta_M = np.std(delta_M, axis=1)
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    std_centroid = np.std(centroid)
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = np.mean(flatness)
    std_flatness = np.std(flatness)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mean_contrast = np.mean(contrast, axis=1)
    std_contrast = np.std(contrast, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)    
    mean_chroma = np.mean(chroma, axis=1)
    std_chroma = np.std(chroma, axis=1)

    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    mean_tonnetz = np.mean(tonnetz, axis=1)
    std_tonnetz = np.std(tonnetz, axis=1)

    feature_vector = np.concatenate([
        [tempo],
        [mean_rms, std_rms],
        mean_M, std_M,
        std_delta_M,
        [mean_centroid, std_centroid],
        [mean_flatness, std_flatness],
        mean_contrast, std_contrast,
        mean_chroma, std_chroma,
        mean_tonnetz, std_tonnetz,
        mean_tempogram, std_tempogram
    ])

    return feature_vector


paths = [
    r"Some Songs\Imaginary Friend - ITZY.flac",
    r"Some Songs\04 - 1-800-hot-n-fun.flac",
    r"Some Songs\06 - Love Is Lonely.flac"
]

vectors = []

for p in paths:
    y, sr = librosa.load(p)
    v = songAnalysis(y, sr)
    vectors.append(v)

X = np.vstack(vectors)

mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma == 0] = 1.0   # avoid division by zero

X_norm = (X - mu) / sigma

from sklearn.metrics.pairwise import cosine_similarity
S = cosine_similarity(X_norm)

pairs = []
n = len(paths)
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((i, j, S[i, j]))

# sort by similarity descending
pairs.sort(key=lambda x: x[2], reverse=True)

for i, j, sim in pairs:
    print(f"{paths[i]} and {paths[j]} are similar with value {sim:.3f}")
