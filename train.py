from config import Config
from lstm_model import LstmModel
from data_process import Data
from utterance_embed import UtteranceEmbed
from cnn_embed import CnnLstmModel


def main(config):
    
    data = Data(config.train_filename, config.dev_filename, config.test_filename)
    train_data = data.train_set
    dev_data = data.dev_set
    test_data = data.test_set

    # load word2vec
    utter_embed = UtteranceEmbed('dbdc3/data/word2vec/wiki_en_model')
    
    input_size = (utter_embed.get_vector_size() * 2)
    num_classes = 3
    
    if config.embed_method == 'word2vec':
        num_hideen = 128
        model = LstmModel(input_size, num_hideen, num_classes, utter_embed, config)
        model.build()
        model.train(train_data, dev_data, test_data)
        
    elif config.embed_method == 'cnn':
        num_rnn_hiden = 128
        sequence_length = 50
        filter_sizes_str = '3,4,5'
        filter_sizes = list(map(int, filter_sizes_str.split(",")))
        num_filters = 128
        l2_reg_lambda = 0.0
        
        model = CnnLstmModel(num_rnn_hiden, num_classes, utter_embed, sequence_length, filter_sizes, num_filters,
                            config, l2_reg_lambda)
        model.build()
        model.train(train_data, dev_data, test_data)

if __name__ == "__main__":
    config = Config()
    
    main(config)
    