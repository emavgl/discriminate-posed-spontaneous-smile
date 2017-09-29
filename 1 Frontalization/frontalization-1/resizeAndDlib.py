__author__ = 'Iacopo'
import renderer
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
import sys
import myutil
import ThreeD_Model
import config
import glob
import multiprocessing
    
"""
Usage: frontalize.py input_folder output_folder
Output: frontalized image + file with landmarks

Options are specified in config.ini files.
Note. Do not use the script with images larger than cnnSize of config.ini
Here we are using images with width of 540.
"""

# redict stdout to null
null_loc = open(os.devnull, 'w')
sys.stdout = null_loc

inputFolder = sys.argv[1]
outputFolder = sys.argv[2]
this_path = os.path.dirname(os.path.abspath(__file__))
opts = config.parse()
## 3D Models we are gonna use to to the rendering {0, -40, -75}
pose_models = ['model3D_aug_-00']
## In case we want to crop the final image for each pose specified above/
## Each bbox should be [tlx,tly,brx,bry]
resizeCNN = opts.getboolean('general', 'resizeCNN')
cnnSize = opts.getint('general', 'cnnSize')
nSub = opts.getint('general', 'nTotSub')
allModels = myutil.preload(this_path,pose_models,nSub)
if not opts.getboolean('general', 'resnetON'):
    crop_models = [None,None,None]  # <-- with this no crop is done.     
else:
    #In case we want to produce images for ResNet
    resizeCNN=False #We can decide to resize it later using the CNN software or now here.
    ## The images produced without resizing could be useful to provide a reference system for in-plane alignment
    cnnSize=224
    crop_models = [[23,0,23+125,160],[0,0,210,230],[0,0,210,230]]  # <-- best crop for ResNet     

def process_file(f):
    image_key = os.path.basename(f)
    image_path = f
    img = cv2.imread(image_path, 1)
    rendered_raw = cv2.resize(img, (540, 540), interpolation=cv2.INTER_CUBIC)
    lmarks = feature_detection.get_landmarks(rendered_raw, this_path)

    ## Saving if required
    savingString = outputFolder +  '/frontalized/' + image_key
    savingLandmarks = outputFolder +  '/landmarks/' + image_key
    with open(savingLandmarks + ".lm", 'w') as f:
        for lm in lmarks[0]:
            f.write(str(int(lm[0])) + " ")

        f.write("\n")
        for lm in lmarks[0]:
            f.write(str(int(lm[1])) + " ") 

    cv2.imwrite(savingString, rendered_raw)

def demo():
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    check.check_dlib_landmark_weights()
    search_model = inputFolder + "*.jpg"
    fileList = glob.glob(search_model)
    print len(fileList)
    pool = multiprocessing.Pool(processes=6)
    pool.map(process_file, fileList)
        
if __name__ == "__main__":
    demo()
