from PIL import Image, ImageDraw
import numpy as np
import sys, traceback
import os, glob, math
import re
import copy
import csv
from scipy.signal import medfilt
import operator

'''
features_extractions.py
input: folder with frontalized images and landmarks (.lm) file for each image

What it does:
Takes all the images inside the input directory and, for each image, gets the landmarks.
Using the landmarks, the script extracts the dlip and eyelid
using the formulas in the paper "Recognition of Genuine Smiles".
Then, it divides the functions (frame, dlip) and (frame, eyelid)
in temporal phases: onset, apex, offset and extracts the lip features (25)
and eye features (25).
'''

### HELPER FUNCTIONS
def atoi(text):
    return int(text) if text.isdigit() else text
    
def safe_div(x, y):
    if y==0: return 0
    return x/y

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def getMiddlePoint(p1, p2):
    """
    returns the middle point between p1 and p2
    """
    x1, y1 = p1
    x2, y2 = p2
    return (round(float((x1 + x2)/2), 2), round(float((y1 + y2)/2), 2))

def distance(p1, p2):
    """
    Input: two points (x, y), (x2, y2)
    Output: (float) Euclidean distance
    """
    x1, y1 = p1
    x2, y2 = p2
    base = math.pow(x1-x2, 2) + math.pow(y1-y2, 2)
    return math.sqrt(base)

def segmentize(xs, ys):
    """
    Divide the function in segments
    input:
        - xs is the array with all the x values of the function
        - ys is the array with the y values of the functions
        In our case; ys are the amplitudes, xs are the frames
    output: amplitudes (list), speeds (list)
    """
    points = list(zip(xs, ys))

    list_1 = points[:-1]
    list_2 = points[1:]
    
    amplitudes = []
    speeds = []

    # iterate over cuples of points
    for point1, point2 in zip(list_1, list_2):
        x1, y1 = point1
        x2, y2 = point2
        speed = float((y2 - y1))*50
        amplitudes.append(y2 - y1)
        speeds.append(speed)

    return amplitudes, speeds
    
def division_kmeans(amplitudes):
    """
    Divide the function (frame, amplitudes) in temporal phases
    using the kmeans algorithm.
    """
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    f1 = range(1, len(amplitudes) + 1) # (x) frames
    f2 = amplitudes # (y) amplitudes
    X  = np.matrix(zip(f1,f2))
    standardized_X = preprocessing.scale(X)
    
    # since I want the distance of frames to be
    # more relevant than the second features (amplitudes)
    # I have to multiply, a weight, say 2, the second features
    standardized_X[:,0] *= 1.4
    standardized_X[:,1] *= 1
    
    kmeans = KMeans(n_clusters=3, max_iter=500).fit(X)
    return kmeans.labels_.tolist()

def eyeLidAmplitude(p1, p2, upper_middle):
    """
    Calculate the eyelid amplitude
    formula: distance(lower_middle, upper_middle)*vertical_location_function
    """
    lower_middle = getMiddlePoint(p1, p2)
    x_upper_middle, y_upper_middle = upper_middle
    x_lower_middle, y_lower_middle = lower_middle
    vertical_location_function = 1
    # Check if the upper_middle stays below the lower_middle
    # if so, set the vertical_location_function to -1
    if y_upper_middle > y_lower_middle: vertical_location_function = -1
    return distance(lower_middle, upper_middle)*vertical_location_function

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

    x_co = [int((i + abs(min(x_co)*1.5))*1000) for i in x_co]
    y_co = [int((i + abs(min(y_co)*1.5))*1000) for i in y_co]

    indexes_to_get = [36, 37, 38, 39, 42, 43, 44, 45, 30, 48, 54, 8, 62]
    points = []
    for i in indexes_to_get:
        points.append((x_co[i], y_co[i]))

    return [(0, 0), points[0], getMiddlePoint(points[1], points[2]), points[3], points[4],
            getMiddlePoint(points[5], points[6]), points[7], (0, 0), (0, 0),
            points[8], points[9], points[10], points[11], points[12]]

def checkMax(li):
    """
    Check if the list is empty.
    If the list is empty, returns 0
    """
    if len(li) > 0:
       return max(li)
    return 0

def checkMean(li):
    """
    Check if the list is empty.
    If the list is empty, returns 0
    """
    if len(li) > 0:
       return np.mean(li)
    return 0

# Definition of the functions
# - find_longest_positive_sequence
# - find_longest_negative_sequence
# This function are useful to divide the function in temporal segments
# The onset is the phase that starts from 0 and ends
# with the last frame of the longest positive sequence of segments
def find_longest_positive_sequence(arr, limit):
    """
    returns the start frame and the last frame of
    the longest sequence of consecutive positive segments
    A limit is specified because we want to choose only
    the sequence of positive segments that starts before
    the start of the longest sequence of negative segments
    """
    sequences = []
    sequence = 0
    indexes = []
    first_index = None
    last_index = None
    for i, element in enumerate(arr):    
        if element > 0 and i < limit:
            if first_index == None: first_index = i
            sequence += 1
        elif sequence > 0:
            last_index = i
            sequences.append(sequence)
            indexes.append((first_index, last_index))
            sequence = 0
            first_index = None	
   	
    max_sequence = max(sequences)
    max_index = sequences.index(max_sequence)
    return indexes[max_index]

def find_longest_negative_sequence(arr):
    """
    returns the start frame and the last frame of
    the longest sequence of consecutive negative segments
    """
    sequences = []
    sequence = 0
    indexes = []
    first_index = None
    last_index = None
    for i, element in enumerate(arr):    
#        if element < 0 and i > limit:
        if element < 0:
            if first_index == None: first_index = i
            sequence += 1
        elif sequence > 0:
            last_index = i
            sequences.append(sequence)
            indexes.append((first_index, last_index))
            sequence = 0
            first_index = None
     
    max_sequence = max(sequences)
    max_index = sequences.index(max_sequence)
    return indexes[max_index]

def extractFeatures(amplitudes, speeds, left_amplitude, right_amplitude):
    """
    input:
        - amplitudes: list of segments amplitude (dlip2 - dlip1) from segmatization
        - speeds: list of speeds
        - left_amplitude: list of left_amplitude values
        - right_amplitude: a list of right_amplitude values

    Extracts 25 features described in the paper and writes
    4 CSV (one per temporal phase + one for all the phases together)
    """

    # Get the ascending_segments and descending segments
    # if the value of the segment is positive, is ascending
    ascending_segments, descending_segments = [i for i in amplitudes if i > 0 ], [j for j in amplitudes if j < 0]
 
    # Divides into ascending speeds and descending speeds
    speeds_asc, speeds_des = [i for i in speeds if i > 0 ], [j for j in speeds if j < 0]
 
    # Get the absolute values of the descending speeds
    speeds_des_abs = [abs(number) for number in speeds_des]

    # Get the absolute values of the descending segments
    descending_segments_abs = [abs(number) for number in descending_segments]

    # Gets the number of ascending and descending segments
    nascending = len(ascending_segments)
    ndescending = len(descending_segments)
    ntotal = nascending + ndescending

    # Gets the sum of ascending values and abs(descending) values
    sum_ascending = sum(ascending_segments)
    sum_descending = sum(descending_segments_abs)

    ### Duration ###
    # Description: duration of the ascendent, descendent, both phases
    # Formula: number_of_frames / frame_rate
    #          since each segment is composed of two frames
    #          the number of frames is n_segments*2
    # Output: duration = [duration+, duration-, duration_tot]
    ################
    durationP = float(nascending*2)/fps
    durationN = float(ndescending*2)/fps
    durationT = float(ntotal*2)/fps
    duration  = [durationP, durationN, durationT]

    ### DurationRatio ###
    # Description: how many positive frames respect all the frames
    # Formula: number_of_segments_in_phase / number_of_all_segments
    #          since each segment is composed of two frames
    #          the number of frames is n_segments*2
    #          but the *2 at the numerator and *2 at denominator
    #          cancel out.
    # Output: durationRatio
    #####################
    durationRatioP = float(nascending)/ntotal 
    durationRatioN = float(ndescending)/ntotal 
    durationRatio  = [durationRatioP, durationRatioN]

    ### max (value of a segment)
    # Description: max value of a segment
    # Formula: max(amplitudes)
    # Output: maximum
    ####################
    maximum = checkMax(amplitudes)

    ### mean
    # Description: max value of a segment
    # Formula: mean(amplitudes)
    # Output: mean
    ####################
    meanA = checkMean(amplitudes)
    meanP = checkMean(ascending_segments)
    meanD = checkMean(descending_segments_abs)
    mean = [meanA, meanP, meanD]


    ### Standard Deviation
    # Description: standard deviation of the amplitude values
    # Formula: standard deviation, use grade of freedom = 1
    # Output: std
    #####################
    std = np.std(amplitudes, ddof=1)


    ### Total Amplitude
    # Description: sum of the amplitude of the ascending and descending segments
    # Formula: /
    # Output: total_amplitude[sumD+, sumD-]
    ####################
    total_amplitude = [sum_ascending, sum_descending]


    ### Net Amplitude
    # Description: difference of ascending's amplitude and descending's
    # Formula: sumD+ - sumD-
    # Output: total_amplitude[sumD+, sumD-]
    ####################
    net_amplitude = sum_ascending - sum_descending


    ### Amplitude Ratio
    # Description: difference of ascending's amplitude and descending's
    # Formula: sumD+ / sum_ascending + sum_descending and viceversa
    # Output: total_amplitude[sumD+, sumD-]
    ####################
    amplitude_ratio_asc = safe_div(sum_ascending, sum_ascending + sum_descending)
    amplitude_ratio_des = safe_div(sum_descending, sum_ascending + sum_descending)
    amplitude_ratio = [amplitude_ratio_asc, amplitude_ratio_des]

    ### Max speed
    # Description: get max speed
    # Formula: get both asc and desc speeds (abs), get the max
    # Output: max_speed
    ###################
    max_speed = [checkMax(speeds_asc), checkMax(speeds_des_abs)]

    ### Mean speed
    # Description: get mean speed
    # Formula: sum_of_speeds_asc / len(speeds_asc)
    # Output: mean_speed
    ####################
    mean_speed = [checkMean(speeds_asc), checkMean(speeds_des_abs)]

    ### Maximum Acceleration
    # How to calc acceleration, we have "speeds" that contains the speeds
    # of the segments, speeds[0] is the speed of the movement from dlip(0) and dlip(1).
    # So we use the acceleration formula is acceleration = v2 - v1 / t
    # We have to divide the acceleration into acceleration of D+ and D-
    # and we have two different interpretations of this:
    # - put inside acceleration_asc all the positive acceleration and viceversa
    # - acceleration_asc will contain the acceleration, both positive and negative
    #   of the ascending temporal phase, relativlely to ascending segments in that
    #   temporal phase.
    # We chose this last one interpretation. How to identifies if a segment
    # is ascending or descending using the speed values? to do this, it is
    # necessary to check the sings of v2 and v1. 
    # if both are positive, D+
    # if both are negative, D-
    # if v2+ and v1- D+
    # if v2- and v1+, D-
    # we can compress the rules, checking only the sign of v2
    # t is in seconds, and, since we have 50 fps, t = 1 / 50
    acceleration_asc = []
    acceleration_des = []
    list_1 = speeds[:-1]
    list_2 = speeds[1:]
    for v1, v2 in zip(list_1, list_2):
        acc = v2 - v1
        if v2 >= 0:
            acceleration_asc.append(acc)
        else:
            acceleration_des.append(acc)
    
    max_acceleration = [checkMax(acceleration_asc), checkMax(acceleration_des)]

    ### Mean Acceleration
    # Description: mean between acceleration values (all acceleration are positive)
    # Formula: np.mean
    # Output: mean_acceleration
    ###########################
    mean_acceleration = [checkMean(acceleration_asc), checkMean(acceleration_des)]


    ### Net. Amplitude Duration Ratio
    # Description: mean between acceleration values (all acceleration are positive)
    # Formula: (sum D+ - sum |D-|) * fps / len(D)
    # Output: Amplitude Duration Ratio
    ###########################
    net_amplitude_ratio = ((sum_ascending - sum_descending)*50)/ ntotal

    ### Left/Right Amplitude Difference
    # Description: /
    # Formula: abs(sum(left_amplitude) - sum(right_amplitude)) / ntotal
    # Output: left_right_amplitude_diff
    ###########################
    left_right_amplitude_diff = abs(sum(left_amplitude) - sum(right_amplitude)) / ntotal

    # Aggregate all features in a single list (25 features in total)
    features_set = [durationP, durationN, durationT, durationRatioP, durationRatioN,
                    maximum, meanA, meanP, meanD, std, sum_ascending, sum_descending, 
                    net_amplitude, amplitude_ratio_asc, amplitude_ratio_des,
                    checkMax(speeds_asc), checkMax(speeds_des_abs), checkMean(speeds_asc), checkMean(speeds_des_abs),
                    checkMax(acceleration_asc), checkMax(acceleration_des), checkMean(acceleration_asc),
                    checkMean(acceleration_des), net_amplitude_ratio, left_right_amplitude_diff]

    return features_set


def writeCSV(features, features_name, input_folder):
    """
    Write CSV with a single line that represent the set of features
    of a particular folder (video)
    """
    foldername = input_folder.split('/')[-2]
    output_file = '{}{}.{}.csv'.format(input_folder, foldername, features_name)
    with open(output_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(features)

def extractDlipFeaturesFromFolder(input_folder):
    """
    Extract lip landmarks, and use them to calculate the dlip value
    aftr we have the function (frame, dlip), it first calls the
    segmatize functions and then extract featurs function.
    At the end, it writes the extracted value into CSV files.
    """

    # get the list of all the landmarks files
    search_model = input_folder + "*.lm"
    file_list = glob.glob(search_model)

    # sort file_list in natural order
    file_list.sort(key=natural_keys)

    # extract points from the first frame
    first_frame = input_folder + "frame0.jpg.lm"
    landmarks = extractPoints(first_frame)
    f_central_point = landmarks[13]
    f_l10 = landmarks[10]
    f_l11 = landmarks[11]

    # for each file in the list, extract l10, l11
    # and central point of the month
    # calculate dlip and save it in a list "amplitudes"
    amplitudes = []
    left_amplitudes = []
    right_amplitudes = []
    for f in file_list:
        landmarks = extractPoints(f)
        central_point = landmarks[13]
        l10 = landmarks[10]
        l11 = landmarks[11]

        right_amplitude = distance(f_central_point, l10)
        left_amplitude = distance(f_central_point, l11)
        dlip = right_amplitude + left_amplitude
        dlip = float(dlip) / (2*distance(f_l10, f_l11))
        amplitudes.append(dlip)
        left_amplitudes.append(left_amplitude)
        right_amplitudes.append(right_amplitude)


    # apply smoothing median filter
    amplitudes = medfilt(amplitudes, 25)
    left_amplitudes = medfilt(left_amplitudes, 25)
    right_amplitudes = medfilt(right_amplitudes, 25)

    # segmatize 
    seg_amplitudes, speed = segmentize(range(len(amplitudes)), amplitudes)

    # Read from file and find the division algorithm to use
    with open('division_algorithm_to_use.txt') as f:
        filter_result = filter(lambda x: x.split('\t')[0] == input_folder.split('/')[1] + '.png', f.read().split('\n'))
        match = list(filter_result)[0]
        algorithm_to_use = (match.split('\t')[1]).rstrip()

    tries = 0
    while True:
        # NOTE: you could insert a try-catch here, but this part is not problematic
        try:
            if 'paper' == algorithm_to_use:
                # Method 1 - Paper like
                max_negative_sequence = find_longest_negative_sequence(seg_amplitudes)
                limit = max_negative_sequence[0]
                max_positive_sequence = find_longest_positive_sequence(seg_amplitudes, limit)
                
                onset_indexes = (0, max_positive_sequence[1])
                offset_indexes = (max_negative_sequence[0], len(seg_amplitudes) + 1)
                
                onset_frames = range(onset_indexes[0], onset_indexes[1])
                offset_frames = range(offset_indexes[0], offset_indexes[1])
                
                onset_index = 0
                apex_index = 1
                offset_index = 2
                
                # fill "labels" list with the corresponding index for each frame
                labels = []
                for i in range(len(amplitudes)):
                    if i in onset_frames:
                        labels.append(onset_index)
                    elif i in offset_frames:
                        labels.append(offset_index)
                    else:
                        labels.append(apex_index)

                clusters = [[], [], []]
                xs = [[], [], []]
                cluster_left_amplitudes = [[], [], []]
                cluster_right_amplitudes = [[], [], []]
                for i, l in enumerate(labels):
                    clusters[l].append(amplitudes[i])
                    cluster_left_amplitudes[l].append(left_amplitudes[i])
                    cluster_right_amplitudes[l].append(right_amplitudes[i])
                    xs[l].append(i)
            else:
                # Method 2: cluster based
                labels = division_kmeans(amplitudes)
                onset_index = labels[0]
                offset_index = labels[-1]
                apex_index = 3 - (onset_index + offset_index)

                # get index of the last green element
                last_green = list(filter(lambda x: x[1] == offset_index, enumerate(labels)))[0]

                # fix incorrect blue point in other sections
                labels =  [ (labels[x[0]-1] if x[0] < last_green[0] and x[1] == offset_index else x[1]) for x in enumerate(labels) ]

                clusters = [[], [], []]
                xs = [[], [], []]
                cluster_left_amplitudes = [[], [], []]
                cluster_right_amplitudes = [[], [], []]
                for i, l in enumerate(labels):
                    clusters[l].append(amplitudes[i])
                    cluster_left_amplitudes[l].append(left_amplitudes[i])
                    cluster_right_amplitudes[l].append(right_amplitudes[i])
                    xs[l].append(i)

        except Exception as e:
            if tries > 0: 
                raise Exception('It is the second time that something went wrong with ' + input_folder)
            print('There was an error here' + input_folder)
            print(e)
            # try with a different division algorithm
            algorithm_to_use = 'cluster' if algorithm_to_use == 'paper' else 'paper'
            tries += 1
            continue


        # Here we have the division in phases
        # Now it's time to run the extractions

        # NOTE: This step could raise an error
        # For example, if the phase division is not so good
        # You could force to use another method for the classification instead
        try:
            amplitudes_onset, speed_onset = segmentize(xs[onset_index], clusters[onset_index])
            amplitudes_apex, speed_apex = segmentize(xs[apex_index], clusters[apex_index])
            amplitudes_offset, speed_offset = segmentize(xs[offset_index], clusters[offset_index])

            left_amplitude_onset, _  = segmentize(xs[onset_index], cluster_left_amplitudes[onset_index])
            right_amplitude_onset, _   = segmentize(xs[onset_index], cluster_right_amplitudes[onset_index])
            left_amplitude_apex, _   = segmentize(xs[apex_index], cluster_left_amplitudes[apex_index])
            right_amplitude_apex, _    = segmentize(xs[apex_index], cluster_right_amplitudes[apex_index])
            left_amplitude_offset, _ = segmentize(xs[offset_index], cluster_left_amplitudes[offset_index])
            right_amplitude_offset, _  = segmentize(xs[offset_index], cluster_right_amplitudes[offset_index])

            onsetFeatures = extractFeatures(amplitudes_onset, speed_onset, left_amplitude_onset, right_amplitude_onset)
            apexFeatures = extractFeatures(amplitudes_apex, speed_apex, left_amplitude_apex, right_amplitude_apex)
            offsetFeatures = extractFeatures(amplitudes_offset, speed_offset, left_amplitude_offset, right_amplitude_offset)
            totalFeaturesSet = onsetFeatures + apexFeatures + offsetFeatures    

            writeCSV(onsetFeatures, 'lip_onset', input_folder)
            writeCSV(apexFeatures, 'lip_apex', input_folder)
            writeCSV(offsetFeatures, 'lip_offset', input_folder)
            writeCSV(totalFeaturesSet, 'lip_total', input_folder)

            return labels
        except Exception as e:
            if tries > 0: 
                raise Exception('It is the second time that something went wrong with ' + input_folder)
            print('There was an error here' + input_folder)
            print(e)
            # try with a different division algorithm
            algorithm_to_use = 'cluster' if algorithm_to_use == 'paper' else 'paper'
            tries += 1


def extractEyeLidFeaturesFromFolder(input_folder, labels):
    """
    Extract Eyelid Features given an input folder with landmarks
    """
    # get the list of all the landmarks files
    search_model = input_folder + "*.lm"
    file_list = glob.glob(search_model)

    # sort file_list in natural order
    file_list.sort(key=natural_keys)

    # for each file in the list, extract l1, l2, l3, l4, l5
    # calculate EyeLid and save it in a list "amplitudes"
    amplitudes = []
    left_amplitudes = []
    right_amplitudes = []
    for f in file_list:
        landmarks = extractPoints(f)
        l1 = landmarks[1]
        l2 = landmarks[2]
        l3 = landmarks[3]
        l4 = landmarks[4]
        l5 = landmarks[5]
        l6 = landmarks[6]

        # Calculate deyelid
        left_amplitude = eyeLidAmplitude(l1, l3, l2)
        right_amplitude = eyeLidAmplitude(l4, l6, l5)
        dyeyelid = (left_amplitude + right_amplitude) / (2*distance(l1, l3))

        amplitudes.append(dyeyelid)
        left_amplitudes.append(left_amplitude)
        right_amplitudes.append(right_amplitude)

    # We already have "labels" that defines where frames are located

    clusters = [[], [], []]
    xs = [[], [], []]
    cluster_left_amplitudes = [[], [], []]
    cluster_right_amplitudes = [[], [], []]
    for i, l in enumerate(labels):
        clusters[l].append(amplitudes[i])
        cluster_left_amplitudes[l].append(left_amplitudes[i])
        cluster_right_amplitudes[l].append(right_amplitudes[i])
        xs[l].append(i)

    # still a valid way to compute the indexes for both
    # longest sequence and kmeans algorithms
    onset_index = labels[0]
    offset_index = labels[-1]
    apex_index = 3 - (onset_index + offset_index)

    # Also this block can raise an error, but better that it doesn't
    amplitudes_onset, speed_onset = segmentize(xs[onset_index], clusters[onset_index])
    amplitudes_apex, speed_apex = segmentize(xs[apex_index], clusters[apex_index])
    amplitudes_offset, speed_offset = segmentize(xs[offset_index], clusters[offset_index])

    left_amplitude_onset, _  = segmentize(xs[onset_index], cluster_left_amplitudes[onset_index])
    right_amplitude_onset, _   = segmentize(xs[onset_index], cluster_right_amplitudes[onset_index])
    left_amplitude_apex, _   = segmentize(xs[apex_index], cluster_left_amplitudes[apex_index])
    right_amplitude_apex, _    = segmentize(xs[apex_index], cluster_right_amplitudes[apex_index])
    left_amplitude_offset, _ = segmentize(xs[offset_index], cluster_left_amplitudes[offset_index])
    right_amplitude_offset, _  = segmentize(xs[offset_index], cluster_right_amplitudes[offset_index])

    onsetFeatures = extractFeatures(amplitudes_onset, speed_onset, left_amplitude_onset, right_amplitude_onset)
    apexFeatures = extractFeatures(amplitudes_apex, speed_apex, left_amplitude_apex, right_amplitude_apex)
    offsetFeatures = extractFeatures(amplitudes_offset, speed_offset, left_amplitude_offset, right_amplitude_offset)
    totalFeaturesSet = onsetFeatures + apexFeatures + offsetFeatures

    writeCSV(onsetFeatures, 'eye_onset', input_folder)
    writeCSV(apexFeatures, 'eye_apex', input_folder)
    writeCSV(offsetFeatures, 'eye_offset', input_folder)
    writeCSV(totalFeaturesSet, 'eye_total', input_folder)

# Get command line args
# sys.argv[1] is the folder that contains .lm files
input_folder = sys.argv[1]

# Global Variable to be initialized
fps = 50

# Extract Dlip
# Labels contains the temporal division in onset, offset, and apex
# the same temporal division is then used in eyelid features extractions
labels = extractDlipFeaturesFromFolder(input_folder)

# Extract DEyeLid
extractEyeLidFeaturesFromFolder(input_folder, labels)
