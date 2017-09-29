# params: folder with original images
# example resize.sh
mkdir trainingset
for f in ./train/*
do
	date
	echo $f
	fbname=$(basename "$f")
	mkdir traningset/$fbname
	echo "processing $fbname"
	#rm -rf $f/resized $f/frontal/ $f/*.jpg $f/*.lm
	#cp $f/landmarks/* traningset/$fbname/
done
