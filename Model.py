import numpy as np
import tensorflow as tf

class Model:
    """ simple model for Todo HTR """

    # Model constants
    batch_size = 50
    image_size = (32, 192)
    max_text_len = 24

    def __init__(self, char_list):
        """ init CNN, RNN, CTC and TensorFlow """
        self.char_list = char_list

        self.setup_CNN()
        self.setup_RNN()
        self.setup_CTC()

    def setup_CNN(self):
        """ init CNN layers """

        # Parameters of layers
        kernel_size = [5, 5, 3, 3, 3]
        features_num = [1, 32, 64, 128, 128, 256]
        pooling_ksize = pooling_stride = [(2,2), (2,2), (2,2), (1,2), (1,2)]
        layers_num = len(pooling_ksize)
    
    def setup_RNN(self):
        """ init RNN layers """
    
    def setup_CTC(self):
        """ init CTC layer """