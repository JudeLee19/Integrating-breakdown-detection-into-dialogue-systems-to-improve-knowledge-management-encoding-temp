from gensim.models import word2vec
import numpy as np

class UtteranceEmbed():
    
    def __init__(self, file_name, dim=300):
        self.dim = dim
        try:
            print('Loading english word2vec model')
            self.word2vec_model = word2vec.Word2Vec.load(file_name)
        except:
            print('Error while loading word2vec model')
        
    def embed_utterance(self, utterance):
        word_embeddings = [self.word2vec_model[word] for word in utterance.split(' ') if word and word in self.word2vec_model]
        if len(word_embeddings):
            return np.mean(word_embeddings, axis=0)
        else:
            return np.zeros([self.dim], np.float32)