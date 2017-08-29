import tensorflow as tf
import numpy as np
from general_utils import Progbar, print_sentence
from tensorflow.contrib.layers import xavier_initializer as xav
from data_process import minibatches
import joblib
import os


class CnnLstmModel():
    
    def __init__(self, num_rnn_hidden, num_classes, utter_embed,
                 sequence_length, filter_sizes, num_filters, config, l2_reg_lambda=0.0):
        
        self.num_hidden = num_rnn_hidden
        self.num_classes = num_classes
        self.utter_embed = utter_embed
        self.logger = config.logger
        
        self.sequence_length = sequence_length
        self.embedding_size = utter_embed.get_vector_size()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        
        self.config = config
        
        self.cate_mapping_dict = joblib.load('./dbdc3/data/cate_mapping_dict')
        
    def add_placeholders(self):
        """
        20 utterance means 10(user, system pair) rnn step.
        Utterance Input shape : (20(utterance batch), sequence_length, embed_size)
        
        Input shape : (utterance batch_size, sequence_length,
            embed_size(300 word2vec))

        CNN Input shape : [batch_size, in_height, in_width, in_channels]
                            (batch_size, sequence_length, embed_size, 1)
        """
        
        # input for CNN
        self.input_x = tf.placeholder(tf.float32, [20, self.sequence_length, self.embedding_size], name='input_x')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # (batch_size, n_steps(10 utterance turn), input_size)
        # self.input_features = tf.placeholder(tf.float32, [1, 10, self.input_size], name='input_features')
        
        # (batch_size, n_steps)
        self.ground_label = tf.placeholder(tf.int32, [1, 10], name='ground_label')

    def add_logits_op(self):
        with tf.variable_scope('cnn'):
            self.cnn_input = tf.expand_dims(self.input_x, -1)
            # print('self.cnn_input shape')
            # print(self.cnn_input.shape)
            
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    
                    # print('W shape')
                    # print(W.shape)
                    
                    conv = tf.nn.conv2d(
                        self.cnn_input,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    
                    # print('conv shape')
                    # print(conv.shape)
                    
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")

                    # print('pooled shape')
                    # print(pooled.shape)
                    # print('\n\n')
                    pooled_outputs.append(pooled)

            # print('pooled_outputs shape')
            # print(len(pooled_outputs))
            # print(pooled_outputs[0].shape)
            
            # Combine all the pooled features
            num_filters_total = self.num_filters * len(self.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # shape : (20, 384)
            input_features = tf.reshape(self.h_drop, [10, 384 * 2])
            self.input_features = tf.expand_dims(input_features, 0)
            
        with tf.variable_scope('lstm'):
            # need to change input_size to user and system concatenated h_pool_flat dimension.

            rnn_input_size = 384 * 2
            
            W_i = tf.get_variable('W_i', [rnn_input_size, self.num_hidden], initializer=xav())
            b_i = tf.get_variable('b_i', [self.num_hidden], initializer=tf.constant_initializer(0.))
        
            reshaped_features = tf.transpose(self.input_features, [1, 0, 2])
            # print(type(reshaped_features))
            # print('reshaped_features: ', reshaped_features.shape)
            reshaped_features = tf.reshape(reshaped_features, [-1, rnn_input_size])
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
            # print('output logits: ', self.logits.shape)

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
        :return: accuracy and f1 score
        """
    
        num_batches = (len(train_data) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=num_batches)
    
        for i, (concat_utter_list, ground_label) in enumerate(
                minibatches(train_data + dev_data + test_data[:100], self.config.batch_size)):
            
            input_features = []
            for each_utter_list in concat_utter_list:
                user_sentence = each_utter_list[0]
                system_sentence = each_utter_list[1]

                user_words_embedding = self.utter_embed.embed_utterance(user_sentence, sequence_length=50,
                                                                       is_mean=False)
                system_words_embedding = self.utter_embed.embed_utterance(system_sentence, sequence_length=50,
                                                                          is_mean=False)
                
                input_features.append(np.array(user_words_embedding))
                input_features.append(np.array(system_words_embedding))

            # maybe (20, sequence_length, 300)
            input_x = np.array(input_features)
        
            ground_label_list = []
            for label in ground_label:
                ground_label_list.append(self.cate_mapping_dict[label.strip().encode('utf-8')])
            ground_label_list = np.array([ground_label_list])

            dropout_keep_prob = 0.8
            
            feed_dict = {
                self.input_x: input_x,
                self.ground_label: ground_label_list,
                self.dropout_keep_prob : dropout_keep_prob
            }
            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=feed_dict)
        
            prog.update(i + 1, [("train loss", train_loss)])
        
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * num_batches + i)

        accuracy, precision_X, recall_X, f1_score_X, precision_T, recall_T, f1_score_T = self.run_evaluate(sess, test_data[100:])
        self.logger.info("accuracy : {:f}".format(accuracy))
        self.logger.info("precision_X : {:f}".format(precision_X))
        self.logger.info("recall_X : {:f}".format(recall_X))
        self.logger.info("f1_score_X : {:f}".format(f1_score_X))

        self.logger.info("precision X + T : {:f}".format(precision_X + precision_T))
        self.logger.info("recall X + T : {:f}".format(recall_X + recall_T))
        self.logger.info("f1_score X + T : {:f}".format(f1_score_X + f1_score_T))
        # self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * accuracy, 100 * f1_score))
        # self.logger.info("- test acc {:04.2f} - pre {:f} - recall {:0f}- f1 {:04.2f}".format(100 * accuracy, 100 * precision, 100 * recall, 100 * f1_score))
        return accuracy, f1_score_X
        
    def run_evaluate(self, sess, test_data):
        # create confusion matrix to evaluate precision and recall
        confusion_matrix = np.zeros(shape=(3,3))
        
        accuracy_list = []
        for i, (concat_utter_list, ground_label) in enumerate(
                minibatches(test_data, self.config.batch_size)):
            input_features = []
    
            for each_utter_list in concat_utter_list:
                user_sentence = each_utter_list[0]
                system_sentence = each_utter_list[1]
        
                user_words_embedding = self.utter_embed.embed_utterance(user_sentence, sequence_length=50,
                                                                   is_mean=False)
                system_words_embedding = self.utter_embed.embed_utterance(system_sentence, sequence_length=50,
                                                                     is_mean=False)
        
                input_features.append(np.array(user_words_embedding))
                input_features.append(np.array(system_words_embedding))
    
            input_x = np.array(input_features)
    
            ground_label_list = []
            for label in ground_label:
                ground_label_list.append(self.cate_mapping_dict[label.strip().encode('utf-8')])
            ground_label_list = np.array([ground_label_list])
    
            feed_dict = {
                self.input_x: input_x,
                self.dropout_keep_prob: 1.0
            }
    
            labels_pred = sess.run([self.labels_pred], feed_dict=feed_dict)

            predict_list = list(labels_pred)[0][0]
            ground_list = ground_label_list[0]

            correct_pred = 0.
            for pred_ele, ground_ele in zip(predict_list, ground_list):
                confusion_matrix[pred_ele][ground_ele] += 1
                if pred_ele == ground_ele:
                    correct_pred += 1
                else:
                    continue
            accuracy_list.append(correct_pred / len(ground_list))
        accuracy = np.mean(accuracy_list)

        
        # O : Not a breakdown, T : Possible breakdown, X : Breakdown
        tp_O = confusion_matrix[0][0]
        tp_X = confusion_matrix[1][1]
        tp_T = confusion_matrix[2][2]
        
        fp_O = (sum(confusion_matrix[:][0]) - confusion_matrix[0][0])
        fp_X = (sum(confusion_matrix[:][1]) - confusion_matrix[1][1])
        fp_T = (sum(confusion_matrix[:][2]) - confusion_matrix[2][2])
        
        fn_O = (sum(confusion_matrix[0][:]) - confusion_matrix[0][0])
        fn_X = (sum(confusion_matrix[1][:]) - confusion_matrix[1][1])
        fn_T = (sum(confusion_matrix[2][:]) - confusion_matrix[2][2])
        
        precision_X = tp_X / (tp_X + fp_X)
        recall_X = tp_X / (tp_X + fn_X)
        f1_score_X = (2 * precision_X * recall_X) / (precision_X + recall_X)

        precision_T = tp_T / (tp_T + fp_T)
        recall_T = tp_T / (tp_T + fn_T)
        f1_score_T = (2 * precision_T * recall_T) / (precision_T + recall_T)
        
        return accuracy, precision_X, recall_X, f1_score_X, precision_T, recall_T, f1_score_T
    
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
                accuracy, f1_score = self.run_epoch(sess, train_data, dev_data, test_data, epoch)
            
                # decay learning rate
                self.config.lr *= self.config.lr_decay
            
                # add for early stopping
                if f1_score >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1_score
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                                        nepoch_no_imprv))
                        break
        