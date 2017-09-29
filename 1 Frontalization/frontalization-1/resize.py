import sys, os, glob
import PIL
from PIL import Image

"""
Usage:
python resize.py basewidth input_folder output_folder
- basewidth: width
- input_folder: folder that contains images to resize
- output_folder: output folder, if not exists, it's created.
"""

basewidth = int(sys.argv[1])
input_folder = sys.argv[2]
output_folder = sys.argv[3]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

search_model = input_folder + "*.jpg"
file_list = glob.glob(search_model)

for file in file_list:
    image_key = os.path.basename(file)
    img = Image.open(file)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
    savingString = output_folder +  '/' + image_key
    img.save(savingString)
