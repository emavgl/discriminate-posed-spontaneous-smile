# Features Extractions
In this step we want to extract the features for each video.

## Prerequisites:
- dataset: contains a directory for each videos, and, inside each directory, the extracted landmarks of each frame
(es. dataset/001\_deliberate\_smile2\_/frame0.jpg.lm, .../frame1.jpg.lm)

Since in the first part, frontalization or extractions of the landamarks may fails. It is required to run the script
`check.py dataset` to remove empty subfolders or folders with incomplete data.

## Scripts

Now we are ready to extract the landmarks using the command `bash extract_features_from_dataset.sh dataset`.
This command creates 8 folders:
	- eye_apex_features_files
	- eye_offset_features_files
	- eye_onset_features_files
	- eye_total_features_files
	- lip_apex_features_files
	...
	

Each folder, contains a CSV for each videos with the corresponding features. 
The folder contains also an addition CSV file with the header.

Finally, run the bash script `merge.sh` to merge the CSVs.
Note. you can edit the file merge.py to select only a subset of features.

## Output
You can find the features in the *./features* directory
which contains two sub-directory: *eye* and *lip*.

Files that ends wit: *_class*:
\_classes.csv -> no csv labels, no features
.csv -> complete (features and class and labels)
\_base_features -> base features (no class, gender, age and labels)
\_full_features.csv ->  full features (no class, no labels)
\_selected_features.csv -> only selected features

## Results
The best results was obtained using *medianfilter* with k=25 (instead of 5)
And selecting manually the best algorithm to divide the dlip function in temporal phases (instead of using an implementation of the algorithm described in the paper).

## Wrap-up

```
// remove empty folders or incomplete data
python3 check.py dataset

// extract the features
bash extract_features_from_dataset.sh dataset

// merge CSV
bash merge.sh
```

## Notes
- In the folder *other_test_codes* you can find the source-code used in our tests (for example. using landmarks without frontalization).
- During the features extraction there could be some errors like: division by zero, max() on an empty sequence etc. These errors occur because
of a non optimal temporal phase division (for example, if there is no onset_phase). If an error occurs, the program will try with a different temporal phase division algorithm.
