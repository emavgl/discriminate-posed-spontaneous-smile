import os
import sys
import glob
import csv

"""
How to use:
merge_csv.py input_folder detail_file header_csv output
"""

def read_detail_file(file_path):
    """
    Read the detail file and get information such as:
    gender, age, classification (posed or not)
    """
    text = None
    details = {}
    with open(file_path, 'r') as f:
        text = f.read()
    lines = text.split("\n")
    for line in lines:
        if not line: continue
        file_name, index, gender, age, classification = line.split("\t")
        details[file_name] = (gender.strip(), int(age), classification.strip())
    return details

def extract_csv_line(file_path):
    """
    Helper function: read csv line
    """
    line = None
    with open(file_path, 'r') as f:
        line = f.read()
    return line.split(',')

def convertToInteger(value):
    """
    Map:
    'spontaneous': 0
    'posed': 1
    'male': 0
    'female': 1
    """
    if value == 'spontaneous' or value == 'male':
        return 0
    else:
        return 1

def selectFeatures(rows, indexes_to_save):
    """
    To select only a subset of features
    """
    if not indexes_to_save:
        return rows
    else:
        new_rows = []
        for row in rows:
            new_row = []
            for idx, feature in enumerate(row):
                if idx in indexes_to_save:
                    new_row.append(feature)
            new_rows.append(new_row)
        return new_rows
            
# Takes input from command line args
input_folder = sys.argv[1]
detail_file = sys.argv[2]
header_file = sys.argv[3]
output_file  = sys.argv[4]

# get the list of all the files
search_model = input_folder + "*.csv"
file_list = glob.glob(search_model)

# search for all the landmarks csv files
details = read_detail_file(detail_file)

# extract header information and put it
csv_header = ['class', 'gender', 'age']
with open(header_file) as f:
    headers = f.readline().split(",")
    for h in headers:
        if not h: continue
        csv_header.append(h.strip())

# add csv_header information as the first line
# of the output file
rows = [csv_header]

# for each csv files
# - extract additional information from detail_file
# - extract features from csv
# - add a new line
#   [class, gender, age, feature1, f.2 ...]
for file_path in file_list:
    print(file_path)
    filename = file_path.split("/")[-1]
    if not filename[0].isdigit(): continue # got wrong csv
    basename = filename.split(".")[0] + ".mp4"
    gender, age, classification = details[basename]
    gender = convertToInteger(gender)
    classification = convertToInteger(classification)
    features = extract_csv_line(file_path)
    features = [float(x) for x in features]
    row = [classification, gender, age] + features
    rows.append(row)

# Select only the first col
only_classes = selectFeatures(rows, [0])

# All - class
full_features = selectFeatures(rows, range(1, len(rows[0])))

# All - [class, gender, age]
base_features = selectFeatures(rows, range(3, len(rows[0])))

# All
complete = selectFeatures(rows, None)

# Edit it, pass as second parameter a list of col to select
# index are taken
features_selections = selectFeatures(rows, [26, 30, 62, 53, 14, 20, 39, 17, 33, 16, 9])

###
# Write on file sections
###

# Write only classes (no labels and features)    
with open(output_file + "_classes.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(only_classes[1:])    

# Write CSV complete (features and class and labels)
with open(output_file + ".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(complete)
    
# Write base features (no class, gender, age and labels)
with open(output_file + "_base_features.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(base_features[1:])
    
# Write full features (no class, no labels)
with open(output_file + "_full_features.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(full_features[1:])
    
# Write only selected features
with open(output_file + "_selected_features.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(features_selections[1:])
