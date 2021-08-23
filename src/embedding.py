import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from src.embedding_utilities import euclidean_dist, cosine_dist
import smart_open
from tqdm import tqdm
import logging
import csv
import sys

class Embedding():
    
    def __init__(self, language, path_list, dist=cosine_dist, metric='cosine'):
        self.language=language
        self.path_list = path_list 
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


    def load_word2vec(self):
        logging.info(f'loading word embeddings word to vec from path {self.path_list}')
        
        """ since using s3fs working with s3 files is just as easy as working locally
        if self.path_list[0].startswith('s3:'): # aws bucket
            for idx, path in enumerate(self.path_list):
                logging.info(f"-- reading in part {idx} of {len(self.path_list)}")
                for i, line in enumerate(tqdm(smart_open.smart_open(path))):
                    self.read_embedding_from_line(line)
                    

        else:
        """
        df_embeddings = pd.concat([pd.read_csv(path, sep=' ', index_col=0, header=None, engine='python', quoting=csv.QUOTE_NONE) for path in self.path_list])
        df_embeddings = df_embeddings[df_embeddings.index.notnull()] # es gibt im 50k datensatz eine NaN Zeile, die gedroppt werden sollte
        logging.info(f'read in embeddings of shape {df_embeddings.shape}')
        self.word2vec = df_embeddings.to_dict(orient='index')
        logging.info(f'reading embedding')
        self.embedding = df_embeddings #df_embeddings.values.tolist()
        logging.info(f'reading index_to_word')
        self.index_to_word = list(df_embeddings.index)
            
            #for idx, path in enumerate(self.path_list):
            #    logging.info(f"-- reading in part {idx} of {len(self.path_list)}")
                
                #with open(path) as file:
                #    for i, line in enumerate(tqdm(file)):
                #        self.read_embedding_from_line(line)

        #self.embedding = pd.DataFrame(self.embedding, index=self.word2vec.keys())
        
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

        king = self.embedding.loc[w1]
        queen = self.embedding.loc[w2]
        man = self.embedding.loc[w3]
        # king - queen = man - woman
        # - king + queen + man = woman
        v0 = - king + queen + man

        distances = pairwise_distances(v0.values.reshape(1, self.D), self.embedding, self.metric).reshape(self.V)
        idxs = distances.argsort()[:4]

        for idx in idxs:
            word = self.index_to_word[idx]
            if word not in (w1, w2, w3): 
                best_word = word
                break

        logging.info(w1, "-", w2, "=", w3, "-", best_word)
        logging.info(f'Distances {sys.getsizeof(distances)}')# word2vec {sys.getsizeof(self.word2vec)} embedding {sys.getsizeof(self.embedding)}  index_to_word {sys.getsizeof{self.index_to_word}}')

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