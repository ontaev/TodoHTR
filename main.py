import sys
import argparse
import editdistance
import cv2 as cv
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

class Params:
    
    input_image = "data/test.jpg"
    dataset = "DATASET/"
    file_accuracy = "model/accuracy.txt"
    char_list = "model/charlist.txt"

def train(model, loader):
    """ train NN """

    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 10 # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iterInfo = loader.get_iterator_info()
            batch = loader.get_next()
            loss = model.train_batch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)
        
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(Params.file_accuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break

def validate(model, loader):
    """ validate NN """
    print('Validate NN')
    loader.validation_set()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.has_next():
        iterInfo = loader.get_iterator_info()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.get_next()
        recognized = model.infer_batch(batch)
        
        print('Ground truth -> Recognized')    
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gt_texts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            numCharErr += dist
            numCharTotal += len(batch.gt_texts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gt_texts[i] + '"', '->', '"' + recognized[i] + '"')
    
    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate

def infer(model, input_image):
    """ recognize input image """

    img = preprocess(cv.imread(input_image, cv.IMREAD_GRAYSCALE), Model.image_size)
    batch = Batch(None, [img])
    recognized = model.infer_batch(batch)
    print('Recognized:', '"' + recognized[0] + '"')
    

def main():
    """ main function """
    # define some command line arguments
    parser = argparse.ArgumentParser(description = 'Simple Todo handwritten text recognition')
    parser.add_argument('--train', action='store_true', help='train the NN')
    parser.add_argument('--validate', action='store_true', help='validate the NN')

    args = parser.parse_args()

    decoder_type = DecoderType.best_path

    if args.train or args.validate:

        loader = DataLoader(Params.dataset, Model.batch_size, Model.image_size, Model.max_text_len)

        # save characters of model for inference mode
        open(Params.char_list, 'w').write(str().join(loader.char_list))

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
        model = Model(open(Params.char_list).read(), decoder_type, must_restore=True)
        infer(model, Params.input_image)

if __name__ == '__main__':
    main()