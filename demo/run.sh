# Use: ./run.sh VIDEO_PATH AGE GENDER
# Example ./run.sh spontaneous_seba.avi 20 male

VIDEO=$1
AGE=$2
GENDER_S=$3

FILENAME=$(basename "$VIDEO" | cut -d. -f1)
FPS=$(ffmpeg -i $VIDEO 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p")

GENDER=0
if [ $GENDER_S = "female" ]
then
	GENDER=1
fi

# delete if exists video folder
echo "=== settin up"
rm -rf $FILENAME

# extract landmarks
echo "=== extracting landmarks"
./main $VIDEO

# extract features
echo "=== extracting features"
python3 features_extraction.py $FILENAME/ $FPS $GENDER $AGE

# classify
echo "=== classifing"
python3 discriminate.py $FILENAME/$FILENAME.total.csv
