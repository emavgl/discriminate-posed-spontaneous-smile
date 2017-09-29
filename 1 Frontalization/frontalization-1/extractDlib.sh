# params: folder with original images
echo "computing $1"

for f in $1/*
do
	date
	echo "processing $f"

	mkdir $f/frontal
	mkdir $f/frontal/landmarks_original
	
	# frontalize and extract landmarks
	python dlib_ema.py $f/ $f/frontal/
done
