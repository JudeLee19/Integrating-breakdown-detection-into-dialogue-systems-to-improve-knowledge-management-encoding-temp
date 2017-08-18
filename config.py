import os


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
    
    output_path = 'results/word2vec_lstm/'
    model_output = output_path + 'model.weights/'
    