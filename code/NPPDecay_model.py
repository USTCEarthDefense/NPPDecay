# Device
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from utils_funcs import FLOAT_TYPE, init_log_file,DataGenerator, LOG_JITTER
from numpy.polynomial.legendre import leggauss
from tensorflow.keras.preprocessing.sequence import pad_sequences


np.set_printoptions( suppress=True)

def scale_softplus( x, scale):
    return scale * tf.nn.softplus( x / scale)

class RandomFourierLayers:
    def __init__(self, numFreqs, kernelReg = None, kernelInit = None, additionalLayer = False, use_rff = False ):
        self.use_rff = use_rff
        self.projectLayer = tf.layers.Dense( numFreqs, activation=None, kernel_initializer=kernelInit, kernel_regularizer=kernelReg)
        if additionalLayer:
            self.reluLayer = tf.layers.Dense( numFreqs // 2, activation=tf.nn.leaky_relu, kernel_regularizer=kernelReg)
        self.outputLayer = tf.layers.Dense(1, activation=None, kernel_regularizer=kernelReg )

    def __call__(self,x, *args, **kwargs):
        out = self.projectLayer(x)
        if self.use_rff:
            out = tf.concat( [ tf.cos( out),tf.sin( out)], -1)
        else:
            out = tf.nn.relu(out)
        if hasattr(self, 'reluLayer'):
            out = self.reluLayer( out)
        out = self.outputLayer(out)
        return out


class LSPP:
    def __init__(self):
        self.is_built = False
        self.is_test_graph_built = False


    def build_graph(self, config):
        self.global_learning_rate = config['learning_rate']
        self.embedding_dim = config['embedding_dim']
        self.num_event_types = config['num_types']
        self.num_intg_points = config['num_integral_points']
        self.num_segment_intg_points = config['num_segment_integral_points']  # Num of integral points in one segment
        self.num_test_integral_points = config['num_test_integral_points']
        self._weights_reg_lambda = config['weights_reg_lambda']
        self._init_softplus_scale = config['softplus_scale']
        self._hidden_size = config['hidden_size']
        self._additional_layer = config['additional_layer']
        self._use_rff = config['use_rff']
        self._hard_max_length = config['hard_max_length']
        self._time_convert_scale = config['time_convert_scale']
        self._test_deltas_multiplier = config['test_deltas_multiplier']


        # Constant
        points, weights = leggauss( self.num_intg_points )
        self.leggauss_points = tf.constant( points, dtype=FLOAT_TYPE)
        self.leggauss_weights = tf.constant( weights,dtype=FLOAT_TYPE)
        del points, weights

        points, weights = leggauss( self.num_segment_intg_points)
        self.segment_leggauss_points = tf.constant( points, dtype=FLOAT_TYPE)
        self.segment_leggauss_weights = tf.constant( weights, dtype=FLOAT_TYPE)
        del points, weights

        self.all_event_types = tf.constant(np.arange(self.num_event_types), dtype=tf.int32)

        # Parameters
        # Embedding event types
        self.softplus_scale = tf.exp( [tf.Variable( self._init_softplus_scale,dtype=FLOAT_TYPE)] * self.num_event_types)
        self.emb_table = tf.Variable( np.random.randn( self.num_event_types, self.embedding_dim) / np.sqrt( self.embedding_dim), dtype=  FLOAT_TYPE )

        l2_reg = tf.keras.regularizers.l2( self._weights_reg_lambda)
        normal_init = tf.keras.initializers.glorot_normal()

        self.backgoundFunction_RFF = RandomFourierLayers( self._hidden_size, kernelInit= normal_init, kernelReg=l2_reg, use_rff = self._use_rff)
        self.g_RFF = RandomFourierLayers( self._hidden_size, kernelReg=l2_reg, additionalLayer= self._additional_layer, use_rff = self._use_rff)
        self.ita_RFF = RandomFourierLayers( self._hidden_size, kernelReg=l2_reg,  additionalLayer= self._additional_layer, use_rff = self._use_rff)

        # Placeholders
        self.max_seq_length = tf.placeholder( shape = [], dtype=tf.int32)
        self.batch_seqs_event_types = tf.placeholder( shape = [ None, None], dtype=tf.int32)
        self.batch_seqs_timestamps = tf.placeholder( shape = [None, None], dtype=FLOAT_TYPE)
        self.batch_seqs_deltas = tf.placeholder( shape = [ None, None], dtype=FLOAT_TYPE)
        self.batch_seqs_mask = tf.placeholder( shape=[None, None], dtype=FLOAT_TYPE) # [ B, L]

        self.is_training = tf.placeholder( shape=[], dtype=tf.bool)
        self.batch_size = tf.shape( self.batch_seqs_timestamps)[0]


        # Compute Graphs

        # Fetch Embedding
        self.seqs_type_ebd = tf.gather( self.emb_table, self.batch_seqs_event_types) #[B,L,D]
        self.event_types_ebds = tf.gather( self.emb_table, self.all_event_types) # [ K, D]
        self.event_types_base_rate = self.get_base_intensity( self.event_types_ebds) #[K,1]

        self.seqs_base_intensity =  tf.gather( self.event_types_base_rate, self.batch_seqs_event_types) #[ B, L, 1]
        self.seqs_base_intensity = self.seqs_base_intensity * self.batch_seqs_mask[:,:,None]

        self.seqs_type_ebd_ii = tf.tile( self.seqs_type_ebd[:,:,None, :], multiples=[ 1,1, self.max_seq_length, 1])  # [B, L, L, D]
        self.seqs_type_ebd_jj = tf.tile( self.seqs_type_ebd[:,None,:, :], multiples=[ 1, self.max_seq_length,1, 1])  # [B, L, L, D]
        self.seqs_type_ebd_iijj = tf.concat( [ self.seqs_type_ebd_ii, self.seqs_type_ebd_jj], axis=-1) #[B, L, L, 2D]

        # initial triggering function g
        # iijj means j's effect on i
        self.deltas_iijj = self.batch_seqs_timestamps[:,:,None,None] - self.batch_seqs_timestamps[:, None, :, None] # [B,L,L,1]
        print('deltas_iijj', self.deltas_iijj.shape)
        self.deltas_iijj_mask = tf.cast( self.deltas_iijj > 0,FLOAT_TYPE) * self.batch_seqs_mask[:, None, :, None]
        self.deltas_iijj = self.deltas_iijj * self.deltas_iijj_mask
        self.init_trig = self.get_init_trig(self.seqs_type_ebd_iijj)  # [ B, L, L, 1]

        # decayed function ita
        self.decayed_effect = self.get_decayed_mutual_effect(self.seqs_type_ebd_iijj, self.deltas_iijj) #[ B, L, L, 1]

        # Total Mutual Effect
        self.masked_decay_effect = self.init_trig * ( 1.0 - tf.tanh(  self.decayed_effect) ) * self.deltas_iijj_mask
        self.cum_effect = tf.reduce_sum(  self.masked_decay_effect, axis=2, keepdims=False) #[ B,L,1]

        self.lambda_raw = self.seqs_base_intensity + self.cum_effect #[ B,L,1]
        self.events_intensity = scale_softplus( self.lambda_raw, tf.gather(self.softplus_scale, self.batch_seqs_event_types)[...,None] )[...,0] # [ B, L]

        # Masking padded event with 1  (log 1 = 0)
        self.premask_log_events_intensity = tf.where( tf.cast( self.batch_seqs_mask,tf.bool), self.events_intensity, tf.ones_like( self.events_intensity,dtype=FLOAT_TYPE) )
        self.log_events_intensity = tf.log( self.premask_log_events_intensity + LOG_JITTER)
        self.log_events_intensity = self.log_events_intensity * self.batch_seqs_mask #[ B, L]

        # Log intensity term
        self.log_intensity = tf.reduce_mean(tf.reduce_sum( self.log_events_intensity, axis = -1), axis=-1)
        print('log_intensity.shape = ', self.log_intensity.shape)


        # To calculate integral term
        self.times_starts = self.batch_seqs_timestamps - self.batch_seqs_deltas  # [B,L]
        self.int_times_iimm = self.batch_seqs_deltas[:, :, None] / 2.0 * self.segment_leggauss_points  # [B, L, M']
        self.int_times_iimm = self.int_times_iimm + (self.times_starts + self.batch_seqs_timestamps)[:, :, None] / 2.0
        self.int_deltas_iimmjj =self.int_times_iimm[:,:,:, None, None] - self.batch_seqs_timestamps[:, None, None,:, None]  # [ B, L, M', L, 1]
        self.int_deltas_iimmjj_mask = tf.cast( self.int_deltas_iimmjj >0, FLOAT_TYPE) * self.batch_seqs_mask[ : , None,None,:,None] # [ B, L, M', L, 1]

        # Prevent overflow
        self.int_deltas_iimmjj = self.int_deltas_iimmjj * self.int_deltas_iimmjj_mask

        event_types_ebds_kk = tf.tile( self.event_types_ebds[None, :, None,:], multiples = [self.batch_size, 1, self.max_seq_length, 1]) #[B, K, L, D]
        event_types_ebds_jj = tf.tile( self.seqs_type_ebd[:, None,:,:], multiples=[1, self.num_event_types, 1, 1])
        self.event_types_ebds_kkjj = tf.concat( [ event_types_ebds_kk, event_types_ebds_jj], axis = -1) #[ B, K, L, D * 2]

        #
        self.init_trig_kkjj = self.get_init_trig(self.event_types_ebds_kkjj) #[ B, K, L, 1]

        # Input should have shape = [B, L, K, M', L, 1 or D *2]
        print( tf.tile(self.int_deltas_iimmjj[:, :, None, :, :, :], multiples=[1, 1, self.num_event_types, 1, 1, 1]).shape)

        self.decayed_effect_ikmj = self.get_decayed_mutual_effect(
            tf.tile( self.event_types_ebds_kkjj[:, None, :, None, :, :], multiples=[1,self.max_seq_length,1, self.num_segment_intg_points,1,1]),
            tf.tile( self.int_deltas_iimmjj[ :, :, None, :,:,: ], multiples=[1,1, self.num_event_types, 1,1,1]), record=True
        ) #[B,L,K,M',L,1]



        self.intg_total_effect = tf.reduce_sum(
            ( self.init_trig_kkjj[:, None,:,None,:,: ] * ( 1 - tf.tanh( self.decayed_effect_ikmj) )) * self.int_deltas_iimmjj_mask[:, :, None, :, :,:],
            axis=-2) # [B, L, K, M',1]

        self.intg_raw_lambda =  self.event_types_base_rate[ None, None, :, None,:] + self.intg_total_effect # [B, L, K, M',1]
        self.intg_intensity_at_points_kk = scale_softplus( self.intg_raw_lambda, self.softplus_scale[None,None,:,None,None])[...,0] # [ B, L, K, M']
        self.intg_intensity_at_points = tf.reduce_sum( self.intg_intensity_at_points_kk, axis=-2) #[ B, L, M']

        # Move the delta out of summation
        self.premask_intg_intensity = tf.reduce_sum( self.intg_intensity_at_points * self.segment_leggauss_weights, axis= - 1) * self.batch_seqs_deltas / 2.0 # [ B, L]
        self.intg_intensity = self.premask_intg_intensity * self.batch_seqs_mask

        # Integral Term
        self.integral_term = tf.reduce_mean( tf.reduce_sum( self.intg_intensity, axis=-1), axis= -1)
        print('integral_term.shape = ', self.integral_term.shape)

        # average negative log likelihood per sequence
        self.reg_loss = tf.losses.get_regularization_loss( ) + self._weights_reg_lambda * tf.reduce_sum( self.emb_table * self.emb_table)
        self.nll = self.integral_term - self.log_intensity  + self.reg_loss

        # build test graph
        self.build_test_graph()

        # Used for data vis
        self.embeddingMatrix_iijj = tf.concat( [ tf.tile( self.event_types_ebds[:, None, :], multiples=[1, self.num_event_types,1]),
                                               tf.tile( self.event_types_ebds[None, :, :], multiples=[self.num_event_types, 1, 1])],
                                             axis=-1)#[K,K,2D]
        self.initTrigMatrix_iijj = self.get_init_trig( self.embeddingMatrix_iijj) #[ K, K, 1]
        self.visDelta = tf.placeholder( dtype=FLOAT_TYPE, shape=[])
        self.visDelta_iijj = tf.ones( shape=[self.num_event_types, self.num_event_types,1],dtype=FLOAT_TYPE) * self.visDelta # [K,K,1]
        self.visDecayedEffect = self.get_decayed_mutual_effect( self.embeddingMatrix_iijj, self.visDelta_iijj)
        self.visDecayedEffect = ( 1.0 - tf.tanh( self.visDecayedEffect))


        # setting

        self.globalStep = tf.Variable(0, trainable=False)
        self.lr = tf.train.cosine_decay(self.global_learning_rate,self.globalStep,25000,alpha=1e-5)
        self.min_opt = tf.train.AdamOptimizer(self.lr)
        #self.min_opt = tf.train.AdamOptimizer(self.global_learning_rate)

        # Clipping Gradients
        # self.min_step = self.min_opt.minimize( self.nll )
        self.gradients = self.min_opt.compute_gradients(self.nll, gate_gradients=self.min_opt.GATE_NONE)
        self.clipped_grads = [(tf.clip_by_value(grads, -0.3 / self.global_learning_rate, 0.3 / self.global_learning_rate), vars) for
                              grads, vars in self.gradients]
        self.min_step = self.min_opt.apply_gradients(self.clipped_grads,global_step=self.globalStep)

        # GPU settings
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        self.sess.run(tf.global_variables_initializer())

        self.is_built = True

        return self

    def build_test_graph(self):

        self.test_times_stamps = tf.placeholder( dtype=FLOAT_TYPE, shape=[None]) #[L,]
        self.test_types = tf.placeholder( dtype=tf.int32, shape=[None] ) #[L,]
        self.test_T_max = tf.placeholder( dtype=FLOAT_TYPE,shape=[])
        test_seq_length = tf.shape( self.test_times_stamps)[0]


        self.test_duration = self.test_T_max - self.test_times_stamps[-1]
        self.test_time_interval = self.test_duration / float(self.num_test_integral_points)
        lin_space = tf.range( self.num_test_integral_points, dtype=FLOAT_TYPE) / float( self.num_test_integral_points)

        self.test_tm = self.test_times_stamps[-1] + self.test_duration * lin_space + self.test_time_interval / 2.0 #[ M'']
        self.test_delta_mmjj = tf.tile( self.test_tm[:, None, None], multiples=[1, test_seq_length,1]) \
                               - tf.tile( self.test_times_stamps[None,:, None], multiples=[self.num_test_integral_points,1,1]) # [M'', L, 1]


        self.test_seq_types_ebd = tf.gather( self.emb_table, self.test_types) # [L, D]
        self.test_type_ebd_kkjj = tf.concat( [
            tf.tile( self.event_types_ebds[:, None, :],multiples=[1, test_seq_length,1 ] ),
            tf.tile( self.test_seq_types_ebd[None, :, :], multiples=[self.num_event_types, 1,1])
        ], axis= -1) #[ K, L, D * 2]

        self.test_trig_init = self.get_init_trig(self.test_type_ebd_kkjj) # [K,L,1]

        self.test_trig_decay = self.get_decayed_mutual_effect(
            tf.tile( self.test_type_ebd_kkjj[:, None, :, :], multiples = [ 1, self.num_test_integral_points,1,1]),
            tf.tile( self.test_delta_mmjj[None, :, :, :], multiples=[self.num_event_types,1,1,1])
        ) #[ K,M'',L,1]

        self.test_cum_effect = tf.reduce_sum( self.test_trig_init[:, None, :,:] * ( 1.0 - tf.tanh(  self.test_trig_decay)), axis=-2)[...,0] #[K,M'']
        self.test_intensity_kkmm = scale_softplus( self.event_types_base_rate + self.test_cum_effect, self.softplus_scale[:,None]) #[K, M'']

        # Note the product with time interval here
        self.test_intensity_at_tm = tf.reduce_sum( self.test_intensity_kkmm, axis=0) #[ M'']
        self.test_intg_mm = tf.cumsum(self.test_intensity_at_tm * self.test_time_interval, axis=-1) #[M'']

        # To avoid overflow
        self.min_intg_mm = tf.reduce_min (self.test_intg_mm)
        self.test_intg_mm_shifted = self.test_intg_mm - self.min_intg_mm
        self.prob_tm_raw = self.test_intensity_at_tm * tf.exp( -self.test_intg_mm_shifted) #[M'']

        self.prob_sum = tf.reduce_sum(self.prob_tm_raw)
        self.prob_tm = self.prob_tm_raw / self.prob_sum

        self.test_t_est = tf.reduce_sum(self.prob_tm * self.test_tm)
        self.test_types_est = tf.reduce_sum( self.test_intensity_kkmm / self.test_intg_mm[None, :] * self.prob_tm[None,:], axis=-1) #[K]

        self.is_test_graph_built = True
        return self

    def eval_prediction(self, padded_data, t_max, verbose = False):
        if not self.is_test_graph_built:
            raise NameError('Test graph has not been built')

        seqs_timestamps = padded_data['timestamps']
        seqs_types = padded_data['types']
        lengths = padded_data['lengths']

        ts_true = []
        ts_pred = []
        types_true = []
        types_pred = []

        for i in range( len( seqs_types)):
            valid_len = lengths[i]

            valid_ts = seqs_timestamps[i][:valid_len]
            valid_types = seqs_types[i][:valid_len]

            ts_true.append( valid_ts[-1])
            types_true.append( valid_types[-1])

            pred_input_ts = valid_ts[:-1]
            pred_input_types = valid_types[:-1]

            # We don't need to use a over all max t
            if len(pred_input_ts) >= 2:
                max_duration = np.average( np.diff( pred_input_ts)) * self._test_deltas_multiplier
            elif pred_input_ts[-1] != 0:
                max_duration = pred_input_ts[-1] * self._test_deltas_multiplier
            else:
                max_duration = t_max - pred_input_ts[-1]

            pred_feed_dict = {  self.test_times_stamps : pred_input_ts,
                                self.test_types : pred_input_types,
                                self.test_T_max :pred_input_ts[-1] + max_duration,
                                self.is_training: False}

            t_est, types_prob_est,prob_tm, tm, intensity_k_at_tm = self.sess.run( [self.test_t_est, self.test_types_est,self.prob_tm, self.test_tm, self.test_intensity_kkmm], feed_dict=pred_feed_dict)

            type_pred = np.argmax( types_prob_est)
            if verbose:
                print('type prob: ', types_prob_est)

            ts_pred.append( t_est)
            types_pred.append( type_pred)

        ts_pred = np.array( ts_pred)
        ts_true = np.array( ts_true)

        types_pred = np.array( types_pred)
        types_true = np.array( types_true)

        rmse = np.sqrt( np.mean( ( ts_pred - ts_true) ** 2)) * self._time_convert_scale
        accuracy = np.sum( types_pred == types_true) / len( types_pred)

        return rmse, accuracy

    def eval_nll(self, padded_data, batch_size):
        lengths = padded_data['lengths']
        n_seqs = len( lengths)
        num_events = np.sum( lengths)

        cum_nll = 0

        start_idx = 0
        end_idx = start_idx + batch_size
        while( start_idx < n_seqs):
            end_idx = min( end_idx, n_seqs)

            batch_timestamps = padded_data['timestamps'][start_idx : end_idx]
            batch_deltas = padded_data['deltas'][start_idx : end_idx]
            batch_types = padded_data['types'][start_idx : end_idx]
            batch_lengths = padded_data['lengths'][start_idx : end_idx]
            batch_masks = padded_data['masks'][start_idx : end_idx]
            max_length = batch_timestamps.shape[1]

            feed_dict = {
                self.max_seq_length: max_length,
                self.batch_seqs_event_types: batch_types,
                self.batch_seqs_timestamps: batch_timestamps,
                self.batch_seqs_deltas: batch_deltas,
                self.batch_seqs_mask: batch_masks,
                self.is_training: False
                               }
            log_int_term, int_term = self.sess.run( [self.log_intensity, self.integral_term], feed_dict = feed_dict )

            nll_per_seq = int_term - log_int_term
            cum_nll += nll_per_seq * ( end_idx - start_idx)

            start_idx += batch_size
            end_idx =  start_idx + batch_size

        nll_per_events = cum_nll / num_events
        return nll_per_events


    def train_eval(self, train_data, dev_data, test_data, num_epochs, batch_size, verbose = True, print_every = 50, log_file = None ):
        '''
        :param train_data:
        :param dev_data:
        :param test_data:
        :return:
        '''

        # return : dict.keys() = ['times','delta', 'types', 'lengths']
        padded_train = self.padded_data( train_data['arr_list_timestamps'], train_data['arr_list_deltas'], train_data['arr_list_types'], hard_max_length=self._hard_max_length, convert_time= self._time_convert_scale )
        padded_dev = self.padded_data( dev_data['arr_list_timestamps'], dev_data['arr_list_deltas'], dev_data['arr_list_types'], hard_max_length=self._hard_max_length,  convert_time= self._time_convert_scale)
        padded_test = self.padded_data(test_data['arr_list_timestamps'], test_data['arr_list_deltas'], test_data['arr_list_types'], hard_max_length=self._hard_max_length, convert_time= self._time_convert_scale)

        t_max = max( np.max( padded_train['timestamps']), np.max( padded_dev['timestamps']), np.max( padded_test['timestamps']))

        # Training loop

        best_test_nll = None
        best_test_acc = None
        best_test_rmse = None
        best_dev_nll = float('inf')
        best_dev_acc = float('-inf')
        best_dev_rmse = float('inf')


        train_idx_gnrt = DataGenerator( np.arange( len( padded_train['timestamps'])))
        num_steps_per_epoch = int( len( padded_train['timestamps']) / batch_size)
        for epoch in range( 1, num_epochs + 1):
            if verbose:
                print('starting, epoch = %d' % epoch )

            for step in range( 1, num_steps_per_epoch + 1):
                batch_idx = train_idx_gnrt.draw_next( batch_size)

                batch_timestamps = padded_train['timestamps'][batch_idx]
                batch_deltas = padded_train['deltas'][batch_idx]
                batch_types = padded_train['types'][batch_idx]
                batch_masks = padded_train['masks'][batch_idx]
                max_length = batch_timestamps.shape[1]

                train_feed_dict = { self.max_seq_length : max_length,
                                    self.batch_seqs_event_types : batch_types,
                                    self.batch_seqs_timestamps : batch_timestamps,
                                    self.batch_seqs_deltas : batch_deltas,
                                    self.batch_seqs_mask : batch_masks,
                                    self.is_training : True
                                    }
                # Warm Start
                if epoch==1:
                    train_feed_dict[self.lr] = 1e-5

                '''
                # Gradient Checks
                nll_eval, gradients_eval, premask_log_events_intensity, emb_table= self.sess.run(
                    [
                        self.nll, self.gradients, self.premask_log_events_intensity, self.emb_table,
                    ], feed_dict=train_feed_dict)
                is_grad_finite = [np.all(np.isfinite(grad_pair[0])) for grad_pair in gradients_eval]
                all_finite = np.all(is_grad_finite + [np.isfinite(nll_eval)])
                if not all_finite:
                    print('***')
                    # raise  NameError('Invalid Gradients')
                '''

                _, nll_per_seq, log_term_per_seq, intg_term_per_seq, soft_scale, lr = self.sess.run([
                    self.min_step, self.nll, self.log_intensity, self.integral_term, self.softplus_scale, self.lr
                ], feed_dict=train_feed_dict)

                to_print = step % print_every == 0 or step == 1
                if verbose and to_print:
                    print(
                        '\nepoch = %d, step = %d, lr = %f, nll_per_seq = %g, log_term = %g, intg_term = %g, soft_scale_min = %g, max = %g' % (
                            epoch, step,lr, nll_per_seq, log_term_per_seq, intg_term_per_seq, np.min(soft_scale),
                            np.max(soft_scale)))

            if verbose:
                print('eval train nll')
            train_nll = self.eval_nll( padded_train, batch_size = batch_size)
            if verbose:
                print('eval dev nll')
            dev_nll = self.eval_nll( padded_dev, batch_size = batch_size)
            if verbose:
                print('eval test nll')
            test_nll = self.eval_nll( padded_test, batch_size= batch_size)

            train_rmse, train_acc = -1,-1#self.eval_prediction( padded_train, t_max = t_max,verbose=False)
            dev_rmse, dev_acc = self.eval_prediction( padded_dev, t_max = t_max, verbose = False)
            test_rmse, test_acc = self.eval_prediction( padded_test, t_max = t_max,  verbose = False)

            if dev_nll <= best_dev_nll:
                best_dev_nll = dev_nll
                best_test_nll = test_nll
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                best_test_acc = test_acc
            if dev_rmse <= best_dev_rmse:
                best_dev_rmse = dev_rmse
                best_test_rmse = test_rmse

            if verbose:
                print('\n***epoch =%d\ntrain_nll_per_event = %g, dev = %g, test = %g' % (epoch, train_nll, dev_nll, test_nll))
                print('train_rmse = %g, dev = %g, test = %g' %( train_rmse, dev_rmse, test_rmse))
                print('trian_acc = %g%%, dev = %g%%, test = %g%%' % ( train_acc * 100, dev_acc* 100, test_acc * 100))
                print('best_test_nll = %g, best_test_rmse = %g, best_test_acc = %g' % (
                    best_test_nll, best_test_rmse, best_test_acc))

                log_file.write( 'epoch =%d, train_nll = %g, dev_nll = %g, test_nll = %g, ' % (epoch, train_nll, dev_nll, test_nll))
                log_file.write( 'train_rmse = %g, dev_rmse = %g, test_rmse = %g, train_acc = %g, dev_acc = %g, test_acc = %g\n' %( train_rmse, dev_rmse, test_rmse, train_acc, dev_acc, test_acc))

                print('')
                log_file.flush()
                os.fsync(log_file.fileno())


        return self


    def padded_data(self, list_seqs_timestamps, list_seqs_deltas, list_seqs_types, hard_max_length = None, convert_time = 1):
        if len( list_seqs_types) == 0:
            raise NameError('Empty list')

        lengths = np.array( [ len( seq) for seq in list_seqs_types ] )
        max_length = max( lengths)
        if hard_max_length is not None:
            max_length = min( hard_max_length, max_length)

        times = pad_sequences( list_seqs_timestamps, maxlen = max_length, dtype= np.float32, padding= 'post', value=0.0)
        # scale time
        times = times / convert_time

        deltas = pad_sequences( list_seqs_deltas, maxlen = max_length, dtype=np.float32, padding='post', value = 0.0)
        deltas = deltas / convert_time

        types = pad_sequences( list_seqs_types, maxlen = max_length, dtype=np.int32, padding='post', value=0)

        masks = pad_sequences( list_seqs_types, maxlen = max_length, dtype=np.int32, padding='post', value=-1)
        masks= ( masks != -1 ).astype( np.float32)
        lengths = np.sum( masks, axis=-1).astype( np.int32)

        return { 'timestamps': times, 'deltas' : deltas, 'types' : types, 'lengths' : lengths, 'masks' : masks}

    # MLP as derivative
    # Ita
    def get_decayed_mutual_effect(self, types_embeddings, deltas, record = False):
        #:param types_embeddings: [...,L,L, D * 2]
        #:param deltas: [...,L,L,1]
        #:return:  [..., L,L,1]

        deltas_mm = deltas / 2.0
        deltas_mm = deltas_mm + deltas_mm * self.leggauss_points
        deltas_mm = deltas_mm[...,None] #[...,M,1]


        shape_len = len( types_embeddings.shape)
        multiples = [ 1 for _ in range( shape_len + 1)]
        multiples[-2] = self.num_intg_points
        types_embeddings = tf.tile( types_embeddings[...,None,:], multiples=multiples) #[ ..., M,D*2]

        MLP_out = tf.concat([types_embeddings,tf.cos(deltas_mm) * 2, tf.sin(deltas_mm) * 2, tf.log(1.0 + deltas_mm), tf.tanh(deltas_mm) * 2], axis=-1)
        MLP_out = self.ita_RFF( MLP_out)
        MLP_out = tf.exp( MLP_out)[...,0] #[...,M]

        MLP_out = tf.reduce_sum( MLP_out * self.leggauss_weights, axis= -1,keepdims=True) # [ ..., 1]
        MLP_out = MLP_out * deltas / 2.0

        return MLP_out


    def get_init_trig(self, types_embeddings):
        '''
        :param types_embeddings: types_embeddings: [...,L,L,D * 2]
        :return: [...,L,L,1]
        '''
        MLP_out = types_embeddings
        MLP_out = self.g_RFF( MLP_out)
        # could be negative
        return MLP_out

    def get_base_intensity(self, types_ebd):
        MLP_out = types_ebd
        MLP_out = self.backgoundFunction_RFF( MLP_out)

        # Could be negative
        #MLP_out = tf.exp(MLP_out)
        return MLP_out















