import cv2
import sys
import glob, os

# folder with videos
folder = sys.argv[1]

os.chdir(folder)
for file_path in glob.glob("*.mp4"):
    print("processing ", file_path)
    folder_name = file_path[:-4]
    
    # create a new folder with the same filename
    os.mkdir(folder_name)
    vidcap = cv2.VideoCapture(file_path)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        jpeg_name = "%s/frame%d.jpg" % (folder_name, count)
        cv2.imwrite(jpeg_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])     # save frame as JPEG file
        count += 1
    # remove video at the end
    os.remove(file_path)
