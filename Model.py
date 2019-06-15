import numpy as np
import tensorflow as tf

class DecoderType:

	best_path = 0
	beam_search = 1

class Model:
    """ simple model for Todo HTR """

    # Model constants
    batch_size = 50
    image_size = (32, 192)
    max_text_len = 24

    def __init__(self, char_list, decoder_type=DecoderType.best_path):
        """ init CNN, RNN, CTC and TensorFlow """
        self.char_list = char_list
        self.decoder_type = decoder_type


        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.input_images = tf.placeholder(tf.float32, shape=(None, Model.image_size[0], Model.image_size[1]))


        self.setup_CNN()
        self.setup_RNN()
        self.setup_CTC()

    def setup_CNN(self):
        """ init CNN layers """
        cnn_in = tf.expand_dims(input=self.input_images, axis=3)

        # Parameters of layers
        kernel_size = [5, 5, 3, 3, 3]
        features_num = [1, 32, 64, 128, 128, 256]
        pooling_ksize = pooling_stride = [(2,2), (2,2), (2,2), (1,2), (1,2)]
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
        rnn_in = tf.squeeze(self.cnn_out)


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