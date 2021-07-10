import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

glove_path = '../large-files/glove6B/glove.6B.50d.txt'


# euclidean distance: dist1
def euclidean_dist(a,b):
    return np.linalg.norm(a-b)

# cosine distance: dist2
def cosine_dist(a,b):
    return 1-a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))

# TODO use lfs for large files
def load_word2vec(path = '../large-files/glove6B/glove.6B.50d.txt'):
    print(f'loading word embeddings word to vec from path {path}')
    
    word2vec = {}
    embedding = []
    index_to_word = []
    
    with open(path) as file:
        num = 0
        for line in file:
            values = line.split()
            
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            
            word2vec[word] = vec
            
            embedding.append(vec)
            index_to_word.append(word)
            
            num +=1
            if num % 200000 == 0:
                print(f'line number {num}')
                print(line)
    
    embedding = pd.DataFrame(embedding, index=word2vec.keys())
    num_words, num_dims = embedding.shape
    
    print(f'total number of entries found:  {num_words}. Dimension: {num_dims}')
    return(word2vec, embedding, index_to_word)

word2vec, embedding, index_to_word = load_word2vec(glove_path)
dist, metric = cosine_dist, 'cosine'


def find_analogies(w1, w2, w3, word2vec=word2vec, embedding=embedding, index_to_word=index_to_word, dist=dist, metric=metric):
    V, D = embedding.shape
    
    for w in (w1, w2, w3):
        if w not in word2vec:
            print("%s not in dictionary" % w)
            return

    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king - man + woman

    distances = pairwise_distances(v0.reshape(1, D), embedding, metric=metric).reshape(V)
    idxs = distances.argsort()[:4]

    for idx in idxs:
        word = index_to_word[idx]
        if word not in (w1, w2, w3): 
            best_word = word
            break

    print(w1, "-", w2, "=", best_word, "-", w3)
    return best_word


def nearest_neighbors(w, index_to_word, word2vec, n=5):
    D = len(list(word2vec.values())[0])
    V = len(word2vec)
    
    if w not in word2vec:
        print("%s not in dictionary:" % w)
        return

    v = word2vec[w]
    distances = pairwise_distances(v.reshape(1, D), embedding, metric=metric).reshape(V)
    idxs = distances.argsort()[1:n+1]
    print("neighbors of: %s" % w)
    for idx in idxs:
        print("\t%s" % index_to_word[idx])
    
    return idxs