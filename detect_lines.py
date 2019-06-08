import numpy as np
import cv2 as cv
import os
import statistics
from glob import glob
from collections import Counter

#Define variables
PATH_TO_IMAGES = 'prepared/'
PATH_TO_LINES = 'lines/'
PATH_TO_BOUNDARIES = 'boundaries/'

images = sorted(glob(os.path.join(PATH_TO_IMAGES, '*.*')))

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if (x.ndim != 1):
        raise (ValueError, "Smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def detect_lines(image_file_path, path_to_lines, path_to_bound):

    file_full = str(image_file_path).split('/')[-1]
    file_name = str(file_full).split('.')[0]
    file_type = str(file_full).split('.')[1]
    # Read image and convert to binary
    img = cv.imread(image_file_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Calculate density of dark pixels
    density = np.sum(gray_img, axis=0)

    # Save image to copy
    boundaries = np.copy(img)
    crop = np.copy(img)

    #Smooth density graph
    d = smooth(density, 10, window='flat')
    #Additional smothing around local minimums
    mins = np.r_[True, d[10:] > d[:-10]] & np.r_[d[:-10] > d[10:], True]
    bound = [idx for idx, item in enumerate(mins) if item == True]
    
    height = boundaries.shape[0]
    # Draw a white rectangle to visualize the bounding rect
    for idx, item in enumerate(bound[:-1]):
        cv.rectangle(boundaries, (bound[idx], 0), (bound[idx+1], height), 0, 1)
    
    # Write image with bounding rectangles
    cv.imwrite(path_to_bound + file_name+ str('_bound.')+file_type, boundaries)

    # Crop image on lines and save
    for idx, b in enumerate(bound[:-1]):
        # Draw a white rectangle to visualize the bounding rect
        crop_img = crop[0:height, bound[idx]:bound[idx+1]]
        cv.imwrite(path_to_lines+file_name+'_'+str(idx)+'.'+file_type, crop_img)
    
    with open(path_to_lines+"count_lines.txt", 'a') as outfile:
        outfile.write(file_name+";"+str(len(bound)-1)+"\n")
    outfile.close()

# Delete old lines
for the_file in os.listdir(PATH_TO_LINES):
    file_path = os.path.join(PATH_TO_LINES, the_file)
    try:      
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


for image in images:
    detect_lines(image, PATH_TO_LINES, PATH_TO_BOUNDARIES)