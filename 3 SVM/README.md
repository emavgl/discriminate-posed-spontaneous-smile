# SVM

Run the jupyter notebook which performs 10 fold cross validation, with inner cross fold to select the correct hyperparameter C.
It use the features extracted in the step 2 with the configuration that gave us the best results.

Configuration:
- frontalization (using matlab algorithms)
- median filter (k = 25)
- manually selected division algorithm (paper or cluster-based)

## Results
The best accuracy obtained is about 78%.
All the other configurations (except the one using the first frontalization algorithm) gave us an accuracy around 74-75%.
 
