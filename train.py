from config import Config
from lstm_model import LstmModel
from data_process import Data
from utterance_embed import UtteranceEmbed


def main(config):
    
    data = Data(config.train_filename, config.dev_filename, config.test_filename)
    train_data = data.train_set
    dev_data = data.dev_set

    # load word2vec
    utter_embed = UtteranceEmbed('dbdc3/data/word2vec/wiki_en_model')
    
    input_size = (utter_embed.get_vector_size() * 2)
    num_hideen = 128
    
    model = LstmModel(input_size, num_hideen, 3, utter_embed, config)
    model.build()
    model.train(train_data, dev_data)

if __name__ == "__main__":
    config = Config()
    
    main(config)
    