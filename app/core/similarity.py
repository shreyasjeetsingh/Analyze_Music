import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(query_vec, X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma==0] = 1.0

    Xn = (X - mu) / sigma
    qn = (query_vec - mu) / sigma

    sims = cosine_similarity([qn], Xn)[0]
    return sims