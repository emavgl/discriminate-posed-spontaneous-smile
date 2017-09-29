#!/bin/bash

#set -e

FILES=$1/\*

mkdir lip_onset_features_files
mkdir lip_apex_features_files
mkdir lip_offset_features_files
mkdir lip_total_features_files
mkdir eye_onset_features_files
mkdir eye_apex_features_files
mkdir eye_offset_features_files
mkdir eye_total_features_files

for f in $FILES
do
  echo "Processing $f folder..."
  python features_extraction.py $f/
  FOLDERNAME=$(basename $f)
  cp $f/"$FOLDERNAME".eye_onset.csv eye_onset_features_files/
  cp $f/"$FOLDERNAME".eye_apex.csv eye_apex_features_files/
  cp $f/"$FOLDERNAME".eye_offset.csv eye_offset_features_files/
  cp $f/"$FOLDERNAME".eye_total.csv eye_total_features_files/
  cp $f/"$FOLDERNAME".lip_onset.csv lip_onset_features_files/
  cp $f/"$FOLDERNAME".lip_apex.csv lip_apex_features_files/
  cp $f/"$FOLDERNAME".lip_offset.csv lip_offset_features_files/
  cp $f/"$FOLDERNAME".lip_total.csv lip_total_features_files/
done

# Copy headers
cp headers/onset.csv lip_onset_features_files/csv_header.csv
cp headers/onset.csv eye_onset_features_files/csv_header.csv

cp headers/apex.csv lip_apex_features_files/csv_header.csv
cp headers/apex.csv eye_apex_features_files/csv_header.csv

cp headers/offset.csv lip_offset_features_files/csv_header.csv
cp headers/offset.csv eye_offset_features_files/csv_header.csv

cp headers/total.csv lip_total_features_files/csv_header.csv
cp headers/total.csv eye_total_features_files/csv_header.csv
