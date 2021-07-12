import numpy as np

# euclidean distance: dist1
def euclidean_dist(a,b):
    return np.linalg.norm(a-b)

# cosine distance: dist2
def cosine_dist(a,b):
    return 1-a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))