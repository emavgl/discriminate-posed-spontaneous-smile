# params: folder with original images
echo "computing $1"

for f in $1/*
do
	date
	echo "processing $f"

	# creating output folders
	mkdir $f/frontal/
	mkdir $f/frontal/frontalized
	mkdir $f/frontal/landmarks
	
	# resize
	python resize.py 540 $f/ $f/resized
	
	# frontalize and extract landmarks
	python frontalizeFolder.py $f/resized/ $f/frontal/
done
