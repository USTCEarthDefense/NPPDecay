import numpy as np
import joblib as jbl # The data were saved using 0.16.0
import os

def load_data( data_name, data_folder = '../data', fold = 0, print_info = True):
    valid_data_names = ['retweet', 'so', 'mimic', 'BiVariate_Case0']
    if data_name not in valid_data_names:
        raise NameError('Invalid data name. Available data names: ', valid_data_names)

    data_path = os.path.join( data_folder, '%s_5_folds.jbl' % data_name)
    data = jbl.load( data_path)
    data = data[fold]

    if print_info:
        print('loaded data %s' % data_name)
        print('fold = %g' % fold)
        print('num Types = %d' % data['num_types'])
        totalNumEvents = 0
        totalNumSeqs = 0
        for sub in ['train', 'dev', 'test']:
            lengths = [ len( seq) for seq in data[sub]['arr_list_timestamps']]
            min_len = np.min( lengths)
            max_len = np.max( lengths)
            mean_len = np.mean( lengths)
            subTotalEvents = np.sum( lengths)
            totalNumEvents += subTotalEvents
            totalNumSeqs += len( lengths)
            print('data_name = %s, sub = %s, num_seqs = %d, num_events = %d, min_len = %d, max_len =%d, mean_len =%d' % (
                data_name,sub, len( lengths), subTotalEvents, min_len, max_len, mean_len))
        print( 'totalNumSeqs = %d, totalNumEvents= = %d' % ( totalNumSeqs, totalNumEvents))
    return data

# See the following code snippet above how to load data
if __name__ == '__main__':

    data = load_data('mimic',)
    train_data = data['train']
    dev_data = data['dev']
    train_lists_timestamps = train_data['arr_list_timestamps']



















