import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

GROUP_WEIGHTS = {
    "tempo": 1.3,
    "rms": 0.8,
    "mfcc": 1.8,       
    "delta_mfcc": 1.2,
    "centroid": 0.7,
    "flatness": 0.6,
    "contrast": 1.0,
    "chroma": 1.5,     
    "tonnetz": 1.2,
    "zcr": 0.6,
    "rolloff": 0.7,
    "bandwidth": 0.7,
    "tempogram": 1.0
}

def build_weight_vector(feature_dim):
    w = []
    w += [GROUP_WEIGHTS["tempo"]] * 1
    w += [GROUP_WEIGHTS["rms"]] * 2
    w += [GROUP_WEIGHTS["mfcc"]] * 40
    w += [GROUP_WEIGHTS["delta_mfcc"]] * 20
    w += [GROUP_WEIGHTS["centroid"]] * 2
    w += [GROUP_WEIGHTS["flatness"]] * 2
    w += [GROUP_WEIGHTS["contrast"]] * 14
    w += [GROUP_WEIGHTS["chroma"]] * 24
    w += [GROUP_WEIGHTS["tonnetz"]] * 12
    w += [GROUP_WEIGHTS["zcr"]] * 2
    w += [GROUP_WEIGHTS["rolloff"]] * 2
    w += [GROUP_WEIGHTS["bandwidth"]] * 2
    w += [GROUP_WEIGHTS["tempogram"]] * 2

    W = np.array(w)
    assert W.shape[0] == feature_dim, f"Weight vector length {W.shape[0]} != feature length {feature_dim}"    
    return W


def compute_similarity(query_vec, X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma==0] = 1.0

    Xn = (X - mu) / sigma
    qn = (query_vec - mu) / sigma

    W = build_weight_vector(X.shape[1])

    Xw = Xn * W
    qw = qn * W

    sims = cosine_similarity([qw], Xw)[0]
    return sims