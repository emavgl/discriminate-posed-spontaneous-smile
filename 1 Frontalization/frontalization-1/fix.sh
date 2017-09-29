# params: folder with original images
# example resize.sh
for f in ./train/*
do
	date
	echo "processing $f"
	rm -rf $f/resized $f/frontal/*.lm 
	rm -rf $f/frontalized $f/landmarks
	mkdir $f/frontalized
	mkdir $f/landmarks
	python resizeAndDlib.py $f/frontal/ $f
done
