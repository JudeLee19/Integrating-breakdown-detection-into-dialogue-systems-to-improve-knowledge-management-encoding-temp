import os


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
    
    output_path = 'results/word2vec_lstm/'
    model_output = output_path + 'model.weights/'
    
    lr = 0.001
    lr_decay = 0.9
    clip = -1
    nepoch_no_imprv = 3
    reload = False
    
    num_epochs = 15
    batch_size = 10