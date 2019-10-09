import numpy as np
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

class DecoderType:

	best_path = 0
	beam_search = 1

class Model:
    """ simple model for Todo HTR """

    # Model constants
    batch_size = 50
    image_size = (32, 192)
    max_text_len = 24

    def __init__(self, char_list, decoder_type = DecoderType.best_path, must_restore = False):
        """ init CNN, RNN, CTC and TensorFlow """
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_id = 0


        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.input_images = tf.placeholder(tf.float32, shape=(None, Model.image_size[0], Model.image_size[1]))


        self.setup_CNN()
        self.setup_RNN()
        self.setup_CTC()

        # setup optimizer to train NN
        self.batches_trained = 0
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        with tf.control_dependencies(self.update_ops):
        	self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setup_TF()

    def setup_CNN(self):
        """ init CNN layers """
        cnn_in = tf.expand_dims(input=self.input_images, axis=3)

        # Parameters of layers
        kernel_size = [5, 5, 3, 3, 3]
        features_num = [1, 32, 64, 128, 128, 256]
        pooling_ksize = pooling_stride = [(2,2), (2,2), (2,2), (2,1), (2,1)]
        layers_num = len(pooling_ksize)

        pool = cnn_in

        for i in range(0, layers_num):
            kernel = tf.Variable(tf.truncated_normal([kernel_size[i], kernel_size[i], features_num[i], features_num[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
            conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, pooling_ksize[i][0], pooling_ksize[i][1], 1), (1, pooling_stride[i][0], pooling_stride[i][1], 1), 'VALID')
            
        self.cnn_out = pool
            
    
    def setup_RNN(self):
        """ init RNN layers """
        #remove dimension with size = 1 (width)
        rnn_in = tf.squeeze(self.cnn_out, axis=[1])


        #basic LSTM cells
        num_hidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in range(2)]

        #stack 2 layers
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        #bidirectional RNN
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in, dtype=rnn_in.dtype)
        
        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, num_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_out = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    
    def setup_CTC(self):
        """ init CTC layer """
        # BxTxC -> TxBxC
        self.ctc_in = tf.transpose(self.rnn_out, [1, 0, 2])
		# ground truth text as sparse tensor
        self.gt_texts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		# calc loss for batch
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gt_texts, inputs=self.ctc_in, sequence_length=self.seq_len, ctc_merge_repeated=True))

		# calc loss for each element to compute label probability
        self.saved_ctc_input = tf.placeholder(tf.float32, shape=[Model.max_text_len, None, len(self.char_list) + 1])
        self.loss_per_element = tf.nn.ctc_loss(labels=self.gt_texts, inputs=self.saved_ctc_input, sequence_length=self.seq_len, ctc_merge_repeated=True)

		# decoder: either best path decoding or beam search decoding
        if self.decoder_type == DecoderType.best_path:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctc_in, sequence_length=self.seq_len)
        elif self.decoder_type == DecoderType.beam_search:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctc_in, sequence_length=self.seq_len, beam_width=50, merge_repeated=False)
    
    def setup_TF(self,):
        """ init TensorFlow """
        sess=tf.Session()

        # saver saves model to file
        saver = tf.train.Saver(max_to_keep=1) 
        model_dir = 'model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir) # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found in: ' + model_dir)

		# load saved model if available
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())
        
        return (sess,saver)

    def to_sparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0] # last entry must be max(labelList[i])

	    # go over all texts
        for (batch_element, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for (i, label) in enumerate(label_str):
                indices.append([batch_element, i])
                values.append(label)
        
        return (indices, values, shape)
    
    def decoder_output_to_text(self, ctc_output, batch_size):
        "extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
        encoded_label_strs = [[] for i in range(batch_size)]
        
        # ctc returns tuple, first element is SparseTensor 
        decoded=ctc_output[0][0] 

        # go over all indices and save mapping: batch -> values
        
        for (idx, idx2d) in enumerate(decoded.indices):
        	label = decoded.values[idx]
        	batch_element = idx2d[0] # index according to [b,t]
        	encoded_label_strs[batch_element].append(label)
        
        # map labels to chars for all batch elements
        return [str().join([self.char_list[c] for c in label_str]) for label_str in encoded_label_strs]

    def train_batch(self, batch):
        """ feed a batch into the NN to train it"""

        num_batch_elements = len(batch.images)
        sparse = self.to_sparse(batch.gt_texts)
        rate = 0.01 if self.batches_trained < 10 else (0.001 if self.batches_trained < 10000 else 0.0001) # decay learning rate
        eval_list = [self.optimizer, self.loss]
        feed_dict = {self.input_images : batch.images, self.gt_texts : sparse , self.seq_len : [Model.max_text_len] * num_batch_elements, self.learning_rate : rate, self.is_train: True}
        (_, loss_val) = self.sess.run(eval_list, feed_dict)
        self.batches_trained += 1
        return loss_val

    def infer_batch(self, batch):
        """ feed a batch into the NN to recognize the texts"""
		
		# decode 
        num_batch_elements = len(batch.images)
        eval_list = [self.decoder]
        feed_dict = {self.input_images : batch.images, self.seq_len : [Model.max_text_len] * num_batch_elements, self.is_train: False}
        eval_res = self.sess.run(eval_list, feed_dict)
        decoded = eval_res[0]
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        return texts

    def save(self):
        """ save model to file """
        
        self.snap_id += 1
        self.saver.save(self.sess, 'model/snapshot', global_step=self.snap_id)