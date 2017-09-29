import os, glob, sys, re
import scipy.io as sp #to export the file in .mat

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

landmarks=[]
for folder in os.listdir(sys.argv[1]): #for each folder 
    path=os.path.join(sys.argv[1], folder)
    if os.path.isdir(path):
        print("Processing folder: {}".format(folder))
		
        file_list=glob.glob(os.path.join(path, "*.lm"))
        file_list.sort(key=natural_keys)

        for f in file_list: #for each file in the folder 
            #print("Processing file: {}".format(f))
            x = None
            y = None
            try:
                with open(f, "r") as fo:
                    x = (fo.readline()).split(" ")
                    y = (fo.readline()).split(" ")
            except:
                print("Error: no lm file", landmarks_file_path)
                exit(-1)

            #parse to float
            x = [float(i) for i in x[:-1]] #making it float (Matlab need this type)
            y = [float(i) for i in y[:-1]] #making it float (Matlab need this type)
            
            landmarks.append(x)
            landmarks.append(y)
        #print landmarks
        print len(landmarks)
        folder='LM_'+folder
        sp.savemat(path, mdict={folder: landmarks}) #writes the .mat file with the name of the video
        del landmarks[:] #clear the current landmarks
