import numpy as np
import tensorflow as tf

class Model:
    """ simple model for Todo HTR """

    # Model constants
    batch_size = 50
    image_size = (32, 192)
    max_text_len = 16

    def __init__(self, char_list):
        """ init CNN, RNN, CTC and TensorFlow """
        self.char_list = char_list

    def setup_CNN(self):
        """ init CNN layers """
    
    def setup_RNN(self):
        """ init RNN layers """
    
    def setup_CTC(self):
        """ init CTC layer """