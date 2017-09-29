# Create folders
mkdir features
mkdir features/eye
mkdir features/lip

# Copy eye files into subfolder
cp -r merge_csv.py detail_file.txt features/eye
cp -r eye_total_features_files features/eye
cp -r eye_apex_features_files features/eye
cp -r eye_offset_features_files features/eye
cp -r eye_onset_features_files features/eye

# Enter in subfolder
cd features/eye

# Merge Eye CSVs
python merge_csv.py eye_total_features_files/ detail_file.txt eye_total_features_files/csv_header.csv eye_total_features_merged
python merge_csv.py eye_apex_features_files/ detail_file.txt eye_apex_features_files/csv_header.csv eye_apex_features_merged
python merge_csv.py eye_offset_features_files/ detail_file.txt eye_offset_features_files/csv_header.csv eye_offset_features_merged
python merge_csv.py eye_onset_features_files/ detail_file.txt eye_onset_features_files/csv_header.csv eye_onset_features_merged

# back to project-root
cd ../..

# Copy eye files into subfolder
cp -r merge_csv.py detail_file.txt features/lip
cp -r lip_total_features_files detail_file.txt features/lip
cp -r lip_apex_features_files detail_file.txt features/lip
cp -r lip_offset_features_files detail_file.txt features/lip
cp -r lip_onset_features_files detail_file.txt features/lip

# Enter in subfolder
cd features/lip

# Merge Lip CSVs
python merge_csv.py lip_total_features_files/ detail_file.txt lip_total_features_files/csv_header.csv lip_total_features_merged
python merge_csv.py lip_apex_features_files/ detail_file.txt lip_apex_features_files/csv_header.csv lip_apex_features_merged
python merge_csv.py lip_offset_features_files/ detail_file.txt lip_offset_features_files/csv_header.csv lip_offset_features_merged
python merge_csv.py lip_onset_features_files/ detail_file.txt lip_onset_features_files/csv_header.csv lip_onset_features_merged
