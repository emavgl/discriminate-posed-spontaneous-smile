This part of the project use the code from an on-going face recognition project [4]. Please, check [this project page](http://www.openu.ac.il/home/hassner/projects/augmented_faces/) for updates and more data.

## Dependencies

* [Dlib Python Wrapper](http://dlib.net/)
* [OpenCV Python Wrapper](http://opencv.org/) <= 3.0.0
* [SciPy](http://www.scipy.org/install.html)
* [Matplotlib](http://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Scikit-Learn](http://scikit-learn.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)

## Usage
### Run it

`bash frontalize.sh dataset`

where dataset is a folder that contains subfolder of videoframes
frontalize.sh will resize the frame's image with a width of 540px
(https://github.com/dougsouza/face-frontalization/issues/7)

Then,
frontalizeFolder.py is called, it uses multicore cpu to
frontalize the folder using the library described in the paper.

At the end, we will have, for each subfolder in *dataset*
a folder */frontalized* that contains frontalized image
and */landmarks* which contains 68 landmarks from dlib library

If you are interested only in landmark files you can run
`bash clean.sh dataset`
that will copy all the */landmarks* folders into a single directory
*landmarks_only*.

