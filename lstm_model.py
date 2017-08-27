import tensorflow as tf
import numpy as np
from general_utils import Progbar, print_sentence
from tensorflow.contrib.layers import xavier_initializer as xav
from data_process import minibatches
import joblib


class LstmModel():
    
    def __init__(self, input_size, num_hidden, num_classes, utter_embed, config):
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.utter_embed = utter_embed
        self.logger = config.logger
        self.config = config
        
        self.cate_mapping_dict = joblib.load('./dbdc3/data/cate_mapping_dict')
    
    def add_placeholders(self):
        # (batch_size, n_steps(10 utterance turn), input_size)
        self.input_features = tf.placeholder(tf.float32, [1, 10, self.input_size], name='input_features')
        # (batch_size, n_steps)
        self.ground_label = tf.placeholder(tf.int32, [1, 10], name='ground_label')
    
    def add_logits_op(self):
        with tf.variable_scope('lstm'):
            W_i = tf.get_variable('W_i', [self.input_size, self.num_hidden], initializer=xav())
            b_i = tf.get_variable('b_i', [self.num_hidden], initializer=tf.constant_initializer(0.))

            reshaped_features = tf.transpose(self.input_features, [1, 0, 2])
            print('reshaped_features: ', reshaped_features.shape)
            reshaped_features = tf.reshape(reshaped_features, [-1, self.input_size])
            # print('reshaped_features: ', reshaped_features.shape)
            
            proj_input_features = tf.matmul(reshaped_features, W_i) + b_i
            # print('proj_input_features: ', proj_input_features)

            proj_input_features = tf.split(proj_input_features, 10, 0)
            # print('split proj_input_features: ', proj_input_features)
            
            # define lstm cell
            lstm_fw = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)
            
            outputs, final_state = tf.contrib.rnn.static_rnn(lstm_fw, inputs=proj_input_features, dtype=tf.float32)

            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = tf.reshape(outputs, [-1, self.num_hidden])
            # print('outputs shape')
            # print(outputs.shape)
            
        with tf.variable_scope('output_projection'):
            W_o = tf.get_variable('Wo', [self.num_hidden, self.num_classes],
                                 initializer=xav())
            b_o = tf.get_variable('bo', [self.num_classes],
                                 initializer=tf.constant_initializer(0.))
            
            self.logits = tf.matmul(outputs, W_o) + b_o
            self.logits = tf.expand_dims(self.logits, 0)
            print('output logits: ',self.logits.shape)
        
    def add_pred_op(self):
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
    
    def add_loss_op(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.ground_label)
        self.loss = tf.reduce_mean(losses)
        
        tf.summary.scalar('loss', self.loss)
    
    def add_train_op(self):
        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            
            if self.config.clip > 0:
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, global_norm = tf.clip_by_global_norm(gradients, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(self.loss)
    
    def add_init_op(self):
        self.init = tf.global_variables_initializer()
        
    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)
    
    def build(self):
        self.add_placeholders()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()
    
    def run_epoch(self, sess, train_data, dev_data, test_data, epoch):
        """
        
        :param train_data: contains concatenated sentence(user and system list type) and ground_labels(O, T, X)
        :return: accuracy and f1 scroe
        """
        
        num_batches = (len(train_data) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=num_batches)

        for i, (concat_utter_list, ground_label) in enumerate(minibatches(train_data + dev_data + test_data[:300], self.config.batch_size)):
            input_features = []
            for each_utter_list in concat_utter_list:
                user_sentence = each_utter_list[0]
                system_sentence = each_utter_list[1]
                
                if self.config.embed_method == 'word2vec':
                    user_embedding = self.utter_embed.embed_utterance(user_sentence)
                    system_embedding = self.utter_embed.embed_utterance(system_sentence)
                    input_feature = np.concatenate((user_embedding, system_embedding), axis=0)
                    input_features.append(input_feature)
              
            if self.config.embed_method == 'word2vec':
                input_features = np.array([input_features])
                


            ground_label_list = []
            for label in ground_label:
                ground_label_list.append(self.cate_mapping_dict[label.strip().encode('utf-8')])
            ground_label_list = np.array([ground_label_list])
            
            feed_dict = {
                self.input_features: input_features,
                self.ground_label: ground_label_list
            }
            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=feed_dict)

            prog.update(i + 1, [("train loss", train_loss)])
            
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * num_batches + i)

        # acc, f1 = self.run_evaluate(sess, dev_data, ground_label_idx)
        # self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
        # return acc, f1
    
    def train(self, train_data, dev_data, test_data):
        saver = tf.train.Saver()
        
        best_score = 0
        nepoch_no_imprv = 0

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(self.init)
            
            if self.config.reload:
                self.logger.info("Reloading the latest trained model...")
                saver.restore(sess, self.config.model_output)
            self.add_summary(sess)
            
            for epoch in range(self.config.num_epochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.num_epochs))
                # acc, f1 = self.run_epoch(sess, train, dev, ground_labels, utter_embed, epoch)
                self.run_epoch(sess, train_data, dev_data, test_data, epoch)
                
                # decay learning rate
                self.config.lr *= self.config.lr_decay
                
                # need to add early stopping

            saver.save(sess, self.config.model_output)