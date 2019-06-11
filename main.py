import sys
import argparse
import cv2 as cv

def main():
    # define some command line arguments
    parser = argparse.ArgumentParser(description = 'Simple Todo handwritten text recognition')
    parser.add_argument('--train', action='store_true', help='train the NN')
    parser.add_argument('--validate', action='store_true', help='validate the NN')

    args = parser.parse_args()

    if args.train:
        print("Train NN!\n")
        pass
    elif args.validate:
        print("Validate NN!\n")
        pass
    else:
        print("No args!\n")
        pass

if __name__ == '__main__':
    main()