import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

GROUP_INFLUENCE = {
    "timbre": 0.40, # Instrument and tone character (e.g., piano vs guitar, clean vs distorted)
    "harmony": 0.25, # Pitch, chords, and harmonic movement (key, progression, melodic content)
    "rhythm": 0.20, # Tempo, beat, and rhythmic complexity (speed and groove)
    "texture": 0.15, # Brightness, noisiness, and spectral balance (dark vs bright, smooth vs harsh)
}

# Total size = 125
FEATURE_SIZES = {
    "tempo": 1,
    "rms": 2,
    "mfcc": 40,
    "delta_mfcc": 20,
    "centroid": 2,
    "flatness": 2,
    "contrast": 14,
    "chroma": 24,
    "tonnetz": 12,
    "zcr": 2,
    "rolloff": 2,
    "bandwidth": 2,
    "tempogram": 2,
}

FEATURE_GROUP = {
    "mfcc": "timbre",
    "delta_mfcc": "timbre",
    "rms": "timbre",

    "chroma": "harmony",
    "tonnetz": "harmony",

    "tempo": "rhythm",
    "tempogram": "rhythm",

    "centroid": "texture",
    "flatness": "texture",
    "contrast": "texture",
    "rolloff": "texture",
    "bandwidth": "texture",
    "zcr": "texture",
}

def build_weight_vector(feature_dim):
    w = []
    
    # Compute total dimensionality of each group (e.g., how many dims are "timbre" overall)
    group_dims = {}
    for feat, grp in FEATURE_GROUP.items():
        group_dims[grp] = group_dims.get(grp, 0)  + FEATURE_SIZES[feat]

    feature_order = [
        "tempo", "rms", "mfcc", "delta_mfcc",
        "centroid", "flatness", "contrast",
        "chroma", "tonnetz",
        "zcr", "rolloff", "bandwidth", "tempogram"
    ]

    for feat in feature_order:
        grp = FEATURE_GROUP.get(feat)
        if grp is None:
            raise ValueError(f"Feature {feat} not assigned to any group")
        
        # Base weight per dimension = group influence divided across all its dimensions
        base_w = GROUP_INFLUENCE[grp]/group_dims[grp]

         # Apply exponential decay across MFCC coefficients
        if feat == "mfcc":
            decay = np.exp(-0.05 * np.arange(FEATURE_SIZES[feat]))
            group_vals = base_w * decay * (FEATURE_SIZES[feat] / np.sum(decay))
            w.extend(group_vals.tolist())
        else:
            w.extend([base_w] * FEATURE_SIZES[feat])

    W = np.array(w)

    # Sanity check: ensure the weight vector matches the feature vector dimensionality
    assert W.shape[0] == feature_dim, (
        f"Weight vector length {W.shape[0]} != feature length {feature_dim}"
    )

    return W


def compute_similarity(query_vec, X):

    # normalize features 
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma==0] = 1.0

    Xn = (X - mu) / sigma
    qn = (query_vec - mu) / sigma

    # build and apply feature weights
    W = build_weight_vector(X.shape[1])
    Xw = Xn * W
    qw = qn * W

    sims = cosine_similarity([qw], Xw)[0]
    return sims