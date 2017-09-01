from config import Config
from lstm_model import LstmModel
from data_process import Data
from utterance_embed import UtteranceEmbed
from cnn_embed import CnnLstmModel
import tensorflow as tf

tf.flags.DEFINE_boolean('train', True, 'if true train, if not evaluate')
tf.flags.DEFINE_boolean('reload', False, 'reload')
tf.flags.DEFINE_string('embed_method', 'word2vec', 'embedding method')
tf.flags.DEFINE_integer('num_hidden', 128, 'number of rnn hidden size')
tf.flags.DEFINE_integer('num_filters', 128, 'number of cnn filter sizes')
tf.flags.DEFINE_string('output_path', 'results/model/', 'output path')
tf.flags.DEFINE_string('model_output', 'model.weights/', 'model_path')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'l2 reg  value')
FLAGS = tf.flags.FLAGS


def main(config):
    
    data = Data(config.train_filename, config.dev_filename, config.test_filename)
    train_data = data.train_set
    dev_data = data.dev_set
    test_data = data.test_set

    # load word2vec
    utter_embed = UtteranceEmbed('dbdc3/data/word2vec/wiki_en_model')
    
    input_size = (utter_embed.get_vector_size() * 2)
    num_classes = 3
    
    config.reload = FLAGS.reload
    config.lr = FLAGS.lr
    config.output_path = FLAGS.output_path
    config.model_output = FLAGS.output_path + FLAGS.model_output
    config.train = FLAGS.train
    config.embed_method = FLAGS.embed_method
    
    if config.embed_method == 'word2vec':
        num_hideen = FLAGS.num_hidden
        model = LstmModel(input_size, num_hideen, num_classes, utter_embed, config)
        model.build()

    elif config.embed_method == 'cnn':
        num_rnn_hiden = FLAGS.num_hidden
        sequence_length = 50
        filter_sizes_str = '2,3,4'
        filter_sizes = list(map(int, filter_sizes_str.split(",")))
        num_filters = FLAGS.num_filters
        l2_reg_lambda = 0.0
    
        model = CnnLstmModel(num_rnn_hiden, num_classes, utter_embed, sequence_length, filter_sizes, num_filters,
                             config, l2_reg_lambda)
        model.build()
    
    if config.train == True:
        model.train(train_data, dev_data, test_data)
    else:
        model.evaluate(test_data)
    
if __name__ == "__main__":
    config = Config()
    main(config)
    