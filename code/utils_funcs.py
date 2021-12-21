import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import os
import time

#ARD kernel
MATRIX_JITTER = 1e-6
NN_MAX = 20.0
FLOAT_TYPE = tf.float32
DELTA_JITTER= 1e-6
LOG_JITTER = 1e-4

class DataGenerator:
    def __init__(self, X, y=None, shuffle = True):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        #self.repeat = repeat

        self.num_elems = len(X)
        self.curr_idx = 0

        if self.shuffle:
            self.random_idx = np.random.permutation(self.num_elems)
        else:
            self.random_idx = np.arange( self.num_elems)


    def draw_last(self, return_idx = False):
        '''
        draw last batch sample
        :return:
        '''
        if self.y is not None:
            if return_idx:
                return self.X[self.last_arg_idx], self.y[self.last_arg_idx], self.last_arg_idx
            else:
                return self.X[self.last_arg_idx], self.y[self.last_arg_idx]
        else:
            if return_idx:
                return self.X[self.last_arg_idx], self.last_arg_idx
            else:
                return self.X[self.last_arg_idx]


    def draw_next(self, batch_size, return_idx = False):
        if batch_size > self.num_elems:
            raise NameError("Illegal batch size")

        if batch_size + self.curr_idx > self.num_elems:
            # shuffle

            if self.shuffle:
                self.random_idx = np.random.permutation(self.num_elems)
            else:
                self.random_idx = np.arange(self.num_elems)
            self.curr_idx = 0

        arg_idx = self.random_idx[self.curr_idx: self.curr_idx + batch_size]
        self.last_arg_idx = arg_idx

        self.curr_idx += batch_size

        if self.y is not None:
            if return_idx:
                return self.X[arg_idx], self.y[arg_idx], arg_idx
            else:
                return self.X[arg_idx], self.y[arg_idx]
        else:
            if return_idx:
                return self.X[arg_idx], arg_idx
            else:
                return self.X[arg_idx]


def init_log_file( log_file_path, data_name, model_config, mode = 'a'):
    log_file = open( log_file_path, mode = mode)

    date = time.asctime()

    log_file.write('\n\n' + date + '\n')
    log_file.write("data set = %s\n" %data_name)
    log_file.write('model config:\n%s\n' %( str( model_config)))

    return log_file













