import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer as xav


class LstmModel():
    
    def __init__(self, input_size, num_hidden, num_classes):
        self.input_size = input_size
        self.num_hideen = num_hidden
        self.num_classes = num_classes
    
    def add_placeholders(self):
        self.input_features = tf.placeholder(tf.float32, [1, self.input_size], name='input_features')
        self.init_state_c, self.init_state_h = (tf.placeholder(tf.float32, [1, self.num_hideen]) for _ in range(2))
        self.ground_label = tf.placeholder(tf.int32, name='ground_label')
    
    def add_logits_op(self):
        with tf.variable_scope('lstm'):
            W_i = tf.get_varaiable('W_i', [self.input_size, self.num_hideen], initializer=xav())
            b_i = tf.get_varaiable('b_i', [self.num_hideen], initializer=tf.constant_initializer(0.))

            proj_input_features = tf.matmul(self.input_features, W_i) + b_i
            lstm_fw = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)
            
            lstm_op, state = lstm_fw(inputs=proj_input_features, state=(self.init_state_c, self.init_state_h))
            
            state_reshaped = tf.concat(axis=1, values=(state.c, state.h))
            
        with tf.variable_scope('output_projection'):
            W_o = tf.get_variable('Wo', [2 * self.num_hideen, self.num_classes],
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
            train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)
    
    def add_init_op(self):
        self.init = tf.global_variables_initializer()
    
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