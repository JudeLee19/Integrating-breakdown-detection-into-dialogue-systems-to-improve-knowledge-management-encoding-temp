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
        utterance = utterance.lower()
        word_embeddings = []
        for word in utterance.split(' '):
            if len(word):
                if "'" in word:
                    pre_word = word.split("'")[0]
                    after_word = "'" + word.split("'")[1]
                    if pre_word in self.word2vec_model:
                        word_embeddings.append(self.word2vec_model[pre_word])
                    if after_word in self.word2vec_model:
                        word_embeddings.append(self.word2vec_model[after_word])
                else:
                    if word in self.word2vec_model:
                        word_embeddings.append(self.word2vec_model[word])
        if len(word_embeddings):
            return np.mean(word_embeddings, axis=0)
        else:
            return np.zeros([self.dim], np.float32)