import numpy as np


class Data():
    
    def __init__(self, train_file_name, dev_file_name, test_file_name):
        self.train_set = self.prepare_data(train_file_name)
        self.dev_set = self.prepare_data(dev_file_name)
        self.test_set = self.prepare_data(test_file_name)
        
    def prepare_data(self, file_name):
        data_set = []
        with open(file_name, 'r') as f_r:
            for line in f_r:
                line_split = line.split('||')
                user_utter = line_split[0]
                system_utter = line_split[1]
                ground_label = line_split[2]
                # concate user and system utterance.
                concat_utter_list = [user_utter, system_utter]
                
                data_set.append((concat_utter_list, ground_label))
                
        return data_set


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch
