import numpy as np
import cv2 as cv
import os
from glob import glob

#Define variables
PATH_TO_IMAGES = 'prepared/'
PATH_TO_LINES = 'DATASET/lines/images'
PATH_TO_BOUNDARIES = 'boundaries_words/'
PATH_TO_WORDS = 'words/'

images = sorted(glob(os.path.join(PATH_TO_LINES, '*.*')))

def detect_words(image_file_path, path_to_words, path_to_bound):

    file_full = str(image_file_path).split('/')[-1]
    file_name = str(file_full).split('.')[0]
    file_type = str(file_full).split('.')[1]
    # Read image and convert to binary
    img = cv.imread(image_file_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_median = cv.GaussianBlur(gray_img, (1,3), cv.BORDER_DEFAULT)
    thresh = cv.adaptiveThreshold(gray_median, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,9,3)

    # Save image to copy
    boundaries = np.copy(img)
    crop = np.copy(img)

    # Dilation
    kernel = np.ones((1,3), np.uint8)
    dilation_img = cv.dilate(thresh, kernel, iterations=1)

    # Find contours of words and bounding rectangles
    image, contours, hier = cv.findContours(dilation_img, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    bounding_rect = np.array([cv.boundingRect(c) for c in contours])

    # Delete small rectangles
    idx_to_delete = [item[0] for item in enumerate(bounding_rect) if item[1][3] < 20] 
    bounding_rect = np.delete(bounding_rect, idx_to_delete, 0)

    # Sort rectangles by X-coordinate
    bounding_rect = np.sort(bounding_rect.view('i8,i8,i8,i8'), order=['f1'], axis=0).view(np.int)

    # Draw a white rectangle to visualize the bounding rect
    width = boundaries.shape[1]
    for b in bounding_rect:
        x, y, w, h = b 
        cv.rectangle(boundaries, (0, y), (width, y + h), 0, 1)

    # Write image with bounding rectangles
    cv.imwrite(path_to_bound + file_name+ str('_bound.')+file_type, boundaries)

    with open(path_to_words+"count_words.txt", 'a') as outfile:
        outfile.write(file_name+";"+str(len(bounding_rect))+"\n")
    outfile.close()

    # Crop image on lines and save
    for idx, b in enumerate(bounding_rect):
        # Get the bounding rect
        x, y, w, h = b
        # Draw a white rectangle to visualize the bounding rect
        crop_img = crop[y:y+h, :width]
        cv.imwrite(path_to_words+file_name+'_'+str(idx)+'.'+file_type, crop_img)


# Delete old words
for the_file in os.listdir(PATH_TO_WORDS):
    file_path = os.path.join(PATH_TO_WORDS, the_file)
    try:      
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


for image in images:
    detect_words(image, PATH_TO_WORDS, PATH_TO_BOUNDARIES)