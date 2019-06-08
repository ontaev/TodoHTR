import numpy as np
import cv2 as cv
import os

class Sample:
    """ sample from dataset containing image and ground truth text """
    def __init__(self, gt_text, file_path):
        self.text = gt_text
        self.file_path = file_path

class Batch:
    
