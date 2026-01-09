import librosa
import numpy as np

def songAnalysis(y, sr):
    # ====== Rhythm ======

    # tempo: estimated speed of the song in BPM 
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo.item()

   # tempogram: measures the strength of different tempos over time (rhythmic pattern)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    mean_tempogram = np.mean(tempogram, axis=1)  
    std_tempogram = np.std(tempogram, axis=1)    
    

    # ====== Energy / Loudness ======

    # rms: short-term signal energy (perceived loudness / intensity)
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)  
    std_rms = np.std(rms)    
    

    # ====== Timbre (sound texture) ======

    # mfcc: perceptual spectral envelope (tone color / instrumentation)
    M = librosa.feature.mfcc(y=y, sr=sr)
    mean_M = np.mean(M, axis=1)  
    std_M = np.std(M, axis=1)    
    
    # delta mfcc: temporal change of timbre (articulation, motion)
    delta_M = librosa.feature.delta(M)
    std_delta_M = np.std(delta_M, axis=1)  
    

    # ====== Spectral Shape / Brightness ======

    # centroid: center of mass of the spectrum (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)  
    std_centroid = np.std(centroid)    

    # flatness: how noise-like vs tonal the sound is
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = np.mean(flatness)  
    std_flatness = np.std(flatness)    

    # contrast: difference between spectral peaks and valleys (punchiness / clarity)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mean_contrast = np.mean(contrast, axis=1)  
    std_contrast = np.std(contrast, axis=1)    

    # rolloff: frequency below which most spectral energy lies (high-frequency content)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    mean_rolloff = np.mean(rolloff, axis=1)
    std_rolloff = np.std(rolloff, axis=1)  

    # spectral bandwidth: spread of frequencies around the centroid (richness / complexity)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    mean_spec_bw = np.mean(spec_bw, axis=1)  
    std_spec_bw = np.std(spec_bw, axis=1)   


    # ====== Harmony / Pitch Content ======

    # chroma: energy in the 12 pitch classes (notes independent of octave)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)  
    std_chroma = np.std(chroma, axis=1)   

    # tonnetz: harmonic relations between pitches (chord structure / harmonic flow)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    mean_tonnetz = np.mean(tonnetz, axis=1) 
    std_tonnetz = np.std(tonnetz, axis=1)    


    # ====== Noise / Percussiveness ======

    # zcr: rate of waveform sign changes (noisiness / percussiveness)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    mean_zcr = np.mean(zcr, axis=1) 
    std_zcr = np.std(zcr, axis=1)    


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
        mean_zcr.flatten(), std_zcr.flatten(),
        mean_rolloff.flatten(), std_rolloff.flatten(),
        mean_spec_bw.flatten(), std_spec_bw.flatten(),
        [np.mean(mean_tempogram), np.mean(std_tempogram)] 
    ])
    
    return feature_vector
