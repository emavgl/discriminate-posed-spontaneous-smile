# params: folder with frontalized image and landmarks
mkdir landmarks_orig_only
for f in $1/*
do
	date
	fbname=$(basename "$f")
	mkdir landmarks_orig_only/$fbname
	echo "processing $fbname"
	#rm -rf $f/resized $f/*.jpg $f/*.lm
	cp $f/frontal/landmarks_original/* landmarks_orig_only/$fbname/
	#cp -r $f/frontal/* $f/
	#rm -rf $f/frontal
done
