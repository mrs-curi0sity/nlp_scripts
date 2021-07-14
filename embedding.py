import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from embedding_utilities import euclidean_dist, cosine_dist

#glove_path = '../large-files/glove6B/glove.6B.50d.txt'

class Embedding():
    
    def __init__(self, language, path, dist=cosine_dist, metric='cosine'):
        self.language=language
        self.path = path
        self.dist = dist
        self.metric = metric
        self.word2vec, self.embedding, self.index_to_word = self.load_word2vec()
        self.V, self.D = self.embedding.shape
        
    
    # TODO use lfs for large files
    def load_word2vec(self):
        print(f'loading word embeddings word to vec from path {self.path}')

        word2vec = {}
        embedding = []
        index_to_word = []

        with open(self.path) as file:
            num = 0
            for line in file:
                values = line.split()

                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')

                word2vec[word] = vec

                embedding.append(vec)
                index_to_word.append(word)

                num +=1
                if num % 160000 == 0:
                    print(f'line number {num}')
                    print(line[:16])

        embedding = pd.DataFrame(embedding, index=word2vec.keys())
        num_words, num_dims = embedding.shape

        print(f'total number of entries found:  {num_words}. Dimension: {num_dims}')
        return(word2vec, embedding, index_to_word)
    
    def find_analogies(self, w1, w2, w3):

        for w in (w1, w2, w3):
            if w not in self.word2vec:
                print("%s not in dictionary" % w)
                return

        king = self.word2vec[w1]
        man = self.word2vec[w2]
        woman = self.word2vec[w3]
        # king - queen = man - woman
        # king - queen - man = - woman
        # - king + queen + man = woman
        v0 = king - man + woman

        distances = pairwise_distances(v0.reshape(1, self.D), self.embedding, self.metric).reshape(self.V)
        idxs = distances.argsort()[:4]

        for idx in idxs:
            word = self.index_to_word[idx]
            if word not in (w1, w2, w3): 
                best_word = word
                break

        print(w1, "-", w2, "=", best_word, "-", w3)
        return best_word


    def nearest_neighbors(self, w, n=5):

        if w not in self.word2vec:
            print("%s not in dictionary:" % w)
            return

        v = self.word2vec[w]
        distances = pairwise_distances(v.reshape(1, self.D), self.embedding, self.metric).reshape(self.V)
        idxs = distances.argsort()[1:n+1]
        print("neighbors of: %s" % w)
        for idx in idxs:
            print("\t%s" % self.index_to_word[idx])

        return idxs