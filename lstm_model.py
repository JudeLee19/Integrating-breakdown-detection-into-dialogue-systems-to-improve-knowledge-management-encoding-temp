import tensorflow as tf
import numpy as np
from general_utils import Progbar, print_sentence
from tensorflow.contrib.layers import xavier_initializer as xav
from data_process import minibatches
from utterance_embed import UtteranceEmbed


class LstmModel():
    
    def __init__(self, input_size, num_hidden, num_classes, config):
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.logger = config.logger
    
    def add_placeholders(self):
        self.input_features = tf.placeholder(tf.float32, [1, self.input_size], name='input_features')
        self.init_state_c, self.init_state_h = (tf.placeholder(tf.float32, [1, self.num_hidden]) for _ in range(2))
        self.ground_label = tf.placeholder(tf.int32, name='ground_label')
    
    def add_logits_op(self):
        with tf.variable_scope('lstm'):
            W_i = tf.get_varaiable('W_i', [self.input_size, self.num_hidden], initializer=xav())
            b_i = tf.get_varaiable('b_i', [self.num_hidden], initializer=tf.constant_initializer(0.))

            proj_input_features = tf.matmul(self.input_features, W_i) + b_i
            lstm_fw = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)
            
            lstm_op, state = lstm_fw(inputs=proj_input_features, state=(self.init_state_c, self.init_state_h))
            
            state_reshaped = tf.concat(axis=1, values=(state.c, state.h))
            
        with tf.variable_scope('output_projection'):
            W_o = tf.get_variable('Wo', [2 * self.num_hidden, self.num_classes],
                                 initializer=xav())
            b_o = tf.get_variable('bo', [self.num_classes],
                                 initializer=tf.constant_initializer(0.))
            self.logits = tf.matmul(state_reshaped, W_o) + b_o
        
    def add_pred_op(self):
        self.label_probs = tf.nn.softmax(self.logits)
        self.prediction = tf.arg_max(self.label_probs, dimension=0)
    
    def add_loss_op(self):
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.ground_label)
        tf.summaray.scalar('loss', self.loss)
    
    def add_train_op(self):
        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)
            
            if self.config.clip > 0:
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, global_norm = tf.clip_by_global_norm(gradients, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(self.loss)
    
    def add_init_op(self):
        self.init = tf.global_variables_initializer()
        # set init state to zeros
        self.init_state_c = np.zeros([1, self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1, self.nb_hidden], dtype=np.float32)
    
    def add_summary(self, sess):
        self.merged = tf.summaray.merge_all()
        self.file_write = tf.summary.FileWriter(self.config.output_path, sess.graph)
    
    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()
    
    def run_epoch(self, sess, train_data, dev_data, ground_label_idx, utter_embed, epoch):
        """
        
        :param train_data: contains concatenated sentence(user and system list type) and ground_labels(O, T, X)
        :return: accuracy and f1 scroe
        """
        
        num_batches = (len(train_data) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=num_batches)
        
        for i, (concat_sentence_list, ground_label) in enumerate(minibatches(train_data, self.config.batch_size)):
            user_sentence = concat_sentence_list[0]
            system_sentence = concat_sentence_list[1]
            user_embedding = utter_embed.embed_utterance(user_sentence)
            system_embedding = utter_embed.embed_utterance(system_sentence)
            
            input_features = np.concatenate((user_embedding, system_embedding), axis=0)
            
            feed_dict = {
                self.input_features: input_features.reshape(1, self.input_size),
                self.init_state_c: self.init_state_c,
                self.init_state_h: self.init_state_h,
                self.ground_label: ground_label
            }
            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=feed_dict)

            prog.update(i + 1, [("train loss", train_loss)])
            
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * num_batches + i)

            acc, f1 = self.run_evaluate(sess, dev_data, ground_label_idx)
            self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
            return acc, f1
    
    def train(self, train, dev, ground_labels):
        saver = tf.train.Saver()
        
        best_score = 0
        nepoch_no_imprv = 0

        # load word2vec
        utter_embed = UtteranceEmbed()
        
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(self.init)
            if self.config.relod:
                self.logger.info("Reloading the latest trained model...")
                saver.restore(sess, self.config.model_output)
            self.add_summary(sess)
            
            
            for epoch in range(self.config.num_epochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.num_epochs))
                acc, f1 = self.run_epoch(sess, train, dev, ground_labels, utter_embed, epoch)

                # decay learning rate
                self.config.lr *= self.config.lr_decay