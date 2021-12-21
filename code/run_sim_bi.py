import numpy as np
import os
import tensorflow as tf

from NPPDecay_model import LSPP
from data_loader import load_data
from utils_funcs import init_log_file

def run_SIM_BI_REG():
    fold = 0
    data_name = 'BiVariate_Case0'
    data = load_data(data_name, fold=fold)
    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    config = {
        'fold' : fold,
        'learning_rate': 1e-4,
        'embedding_dim': 4,
        'num_types': data['num_types'],
        'num_integral_points': 4,
        'num_segment_integral_points': 2,
        'num_test_integral_points': 200,
        'weights_reg_lambda':0.005,
        'softplus_scale': -0.2, # this parameter does matter a lot
        'epochs': 20,
        'batch_size': 4,
        'hidden_size': 16,
        'additional_layer': False,
        'use_rff' : False,
        'hard_max_length': 64,
        'time_convert_scale': 1,
        'test_deltas_multiplier': 10
    }

    log = init_log_file( 'results_log_v4_%s.txt' % data_name, data_name, model_config= config)

    np.random.seed(47)
    tf.random.set_random_seed(47)

    model = LSPP( )
    model.build_graph( config )


    model.train_eval( train_data, dev_data, test_data, num_epochs=config['epochs'], batch_size= config['batch_size'],
                      verbose=True, print_every=100, log_file =log )


def run_SIM_BI_CLF():
    fold = 0
    data_name = 'BiVariate_Case0'
    data = load_data(data_name, fold=fold)
    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    config = {
        'fold' : fold,
        'learning_rate': 1e-4,
        'embedding_dim': 4,
        'num_types': data['num_types'],
        'num_integral_points': 4,
        'num_segment_integral_points': 2,
        'num_test_integral_points': 200,
        'weights_reg_lambda':0.005,
        'softplus_scale': -0.2, # this parameter does matter a lot
        'epochs': 20,
        'batch_size': 4,
        'hidden_size': 16,
        'additional_layer': False,
        'use_rff' : False,
        'hard_max_length': 64,
        'time_convert_scale': 1,
        'test_deltas_multiplier': 10
    }

    log = init_log_file( 'results_log_v4_%s.txt' % data_name, data_name, model_config= config)

    np.random.seed(47)
    tf.random.set_random_seed(47)

    model = LSPP( )
    model.build_graph( config )
    model.train_eval( train_data, dev_data, test_data, num_epochs=config['epochs'], batch_size= config['batch_size'],
                      verbose=True, print_every=100, log_file =log )


if __name__ == '__main__':
    #run_SIM_BI_REG()
    run_SIM_BI_CLF()