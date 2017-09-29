from sklearn.svm import SVC
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.externals import joblib
import sys, csv

# To retrieve back the object
svm = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

features = []
features_path = sys.argv[1]
with open(features_path, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for r in spamreader:
        features.append(r)

X_test = scaler.transform(features)
result = svm.predict(X_test)
print('result', int(result[0]))
result = int(result[0])
if result == 1:
    print("POSED")
else:
    print("SPONTANEOUS")

sys.exit(0)

