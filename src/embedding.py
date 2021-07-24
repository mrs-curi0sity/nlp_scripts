import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from src.embedding_utilities import euclidean_dist, cosine_dist
import smart_open
from tqdm import tqdm

#glove_path = '../large-files/glove6B/glove.6B.50d.txt'

class Embedding():
    
    def __init__(self, language, path, dist=cosine_dist, metric='cosine'):
        self.language=language
        self.path = path
        self.dist = dist
        self.metric = metric
        self.word2vec, self.embedding, self.index_to_word = {}, [], []
        self.load_word2vec()
        self.V, self.D = self.embedding.shape
        
        
    def read_embedding_from_line(self, line):
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        self.word2vec[word] = vec

        self.embedding.append(vec)
        self.index_to_word.append(word)
        

    # TODO use lfs for large files
    def load_word2vec(self):
        num_lines = 0
        print(f'loading word embeddings word to vec from path {self.path}, num lines: {num_lines}')
        
        if self.path.startswith('s3:'): # aws bucket
            for i, line in enumerate(tqdm(smart_open.smart_open(self.path))):
                self.read_embedding_from_line(line)

        else:
            with open(self.path) as file:
                for i, line in enumerate(tqdm(file)):
                    self.read_embedding_from_line(line)

        self.embedding = pd.DataFrame(self.embedding, index=self.word2vec.keys())
        
        num_words, num_dims = self.embedding.shape
        print(f'total number of entries found:  {num_words}. Dimension: {num_dims}')

        
    
    def get_embedding(self, word):
        try:
            return self.word2vec[word]
        except:
            print(f'sorry, word {word} not in index')
            return 

    def find_analogies(self, w1, w2, w3):

        for w in (w1, w2, w3):
            if w not in self.word2vec:
                print("%s not in dictionary" % w)
                return

        king = self.word2vec[w1]
        queen = self.word2vec[w2]
        man = self.word2vec[w3]
        # man = self.word2vec[w2]
        # woman = self.word2vec[w3]
        # king - queen = man - woman
        # king - queen - man = - woman
        # - king + queen + man = woman
        v0 = - king + queen + man

        distances = pairwise_distances(v0.reshape(1, self.D), self.embedding, self.metric).reshape(self.V)
        idxs = distances.argsort()[:4]

        for idx in idxs:
            word = self.index_to_word[idx]
            if word not in (w1, w2, w3): 
                best_word = word
                break

        print(w1, "-", w2, "=", w3, "-", best_word)
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