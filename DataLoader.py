import numpy as np
import cv2 as cv
import os
import random
from SamplePreprocessor import preprocess

class Sample:
    """ sample from dataset containing image and ground truth text """
    def __init__(self, gt_text, file_path):
        self.text = gt_text
        self.file_path = file_path

class Batch:
    """ batch containing images and corresponding ground truth texts """
    def __init__(self, gt_texts, images):
        self.images = np.stack(images, axis=0)
        self.gt_texts = gt_texts
    
class DataLoader:
    """ loads data from Todo dataset """
    def __init__(self, file_path, batch_size, image_size, max_text_len):
        
        self.curr_idx = 0
        self.batch_size = batch_size
        self.image_size = image_size
        self.samples = []

        chars = set()

        with open(file_path + "words.txt", 'r') as input_file:
            for line in input_file:
                line_split = line.strip().split('\t')
                file_name = file_path+"/words/"+line_split[1]
                gt_text = line_split[0]
                chars = chars.union(set(list(gt_text)))
                self.samples.append(Sample(gt_text, file_name))
        input_file.close()

        # split into train and validation     
        split_idx = int(0.8 * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        self.num_train_samples_per_epoch = 5000
        self.char_list = sorted(list(chars))

        print(self.char_list)

    def train_set(self):
        """ switch to randomly chosen subset of training set """
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples[:self.num_train_samples_per_epoch]
    
    def validation_set(self):
        """ switch to validation set """
        self.curr_idx = 0
        self.samples = self.validation_samples
    
    def get_iterator_info(self):
        """ current batch index and overall number of batches """
        return (self.curr_idx // self.batch_size + 1, len(self.samples) // self.batch_size)

    def has_next(self):
        """ check existence of next batch """
        return self.curr_idx + self.batch_size <= len(self.samples)
    
    def get_next(self):
        """ get next batch """
        batch_range = range(self.curr_idx, self.curr_idx + self.batch_size)
        gt_texts = [self.samples[i].text for i in batch_range]
        images = [preprocess(cv.imread(self.samples[i].file_path, cv.IMREAD_GRAYSCALE), self.image_size, True) for i in batch_range]
        self.curr_idx += self.batch_size
        return Batch(gt_texts, images)
