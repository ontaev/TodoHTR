import sys
import argparse
import cv2 as cv
from DataLoader import DataLoader, Batch
from Model import Model

class Params:
    char_list = ""
    input_image = "input/test.jpg"
    dataset = "DATASET/"

def train(model, loader):
    """ train NN """
    print("Train model\n")

def validate(model, loader):
    """ validate NN """
    print("Validate model\n")

def recognize(model, input_image):
    """ recognize input image """

def main():
    """ main function """
    # define some command line arguments
    parser = argparse.ArgumentParser(description = 'Simple Todo handwritten text recognition')
    parser.add_argument('--train', action='store_true', help='train the NN')
    parser.add_argument('--validate', action='store_true', help='validate the NN')

    args = parser.parse_args()

    if args.train or args.validate:

        loader = DataLoader(Params.dataset, Model.batch_size, Model.image_size, Model.max_text_len)

        if args.train:

            print("Training NN!\n")
            model = Model(loader.char_list)
            train(model, loader)
        
        elif args.validate:
            print("Validating NN!\n")
            model = Model(loader.char_list)
            validate(model, loader)

    else:
        print("Recognizing image from "+ Params.input_image + " file!\n")
        pass

if __name__ == '__main__':
    main()