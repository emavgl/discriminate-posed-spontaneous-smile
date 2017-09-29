from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import sys

def getMiddlePoint(p1, p2):
    """
    returns the middle point between p1 and p2
    """
    x1, y1 = p1
    x2, y2 = p2
    return (int((x1 + x2)/2), int((y1 + y2)/2))

def extractPoints(landmarks_file_path):
    """
    input: path of the file with landmarks
    landmark file should have two rows, the first with "x" coordinates
    the second with the "y". 
    output: This function create the landmark point and get only the points
    described in the paper.
    """
    x_co = None
    y_co = None
    try:
        with open(landmarks_file_path, "r") as fo:
            x_co = (fo.readline()).split(" ")
            y_co = (fo.readline()).split(" ")
    except:
        print("Error: no lm file", landmarks_file_path)
        exit(-1)

    # parse to int
    x_co = [float(i) for i in x_co[:-1]]
    y_co = [float(i) for i in y_co[:-1]]

    x_co = [int((i + abs(min(x_co)*1.3))*1000) for i in x_co]
    y_co = [int((i + abs(min(y_co)*1.3))*1000) for i in y_co]
    
    indexes_to_get = [36, 37, 38, 39, 42, 43, 44, 45, 30, 48, 54, 8, 62]
    points = []
    for i in indexes_to_get:
        points.append((x_co[i], y_co[i]))

#    return list(zip(x_co, y_co))
    return [(0, 0), points[0], getMiddlePoint(points[1], points[2]), points[3], points[4],
            getMiddlePoint(points[5], points[6]), points[7], (0, 0), (0, 0),
            points[8], points[9], points[10], points[11], points[12]]

landmark_path = sys.argv[1]
landmarks = extractPoints(landmark_path);

im = Image.new('RGB', (500, 500))
draw = ImageDraw.Draw(im)
for landmark in landmarks:
    r = 5
    x, y = landmark
    ellipse = (x - r, y - r, x + r, y + r)
    draw.ellipse(ellipse, fill=128)
im.show()
