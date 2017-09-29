import os, glob, sys, re
import scipy.io as sp #to import the .mat file

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def lm_range(start, end, step):
    while start <= end:
        yield start
        start += step

file_list=glob.glob(os.path.join(sys.argv[1], "*.mat"))
file_list.sort(key=natural_keys)
for f in file_list: #for each folder
    print("Processing file: {}".format(f))
    
    #read the file    
    try:
        mat = (sp.loadmat(f))['Aligned_S3']
    except:
        print("Error: no mat file", f)
        exit(-1)
     
     #create a folder with that name
    folderName=f[:-4]
    if not os.path.exists(folderName):
        os.makedirs(folderName)
        
    
    for i in lm_range(0,(len(mat)-3),3): #for every three rows

        lm = open(os.path.join(folderName, 'frame'+str(i/3)+'.jpg.lm'), 'w') #will contain the landmarks
        for x in range(0,len(mat[i])): #write the x(s)
            lm.write(str(mat[i][x]))
            lm.write(' ')
        lm.write('\n')
        for y in range(0,len(mat[i+1])): #write the y(s)
            lm.write(str(mat[i+1][y]))
            lm.write(' ')
        lm.write('\n')
    lm.close()
    #print mat
    os.remove(f) #remove the mat file
