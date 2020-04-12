# Urban Release

Modifying and running code of Urban Level recognition

[Vicente Ordonez](http://www.cs.virginia.edu/~vicente/urban/index.html) @ UNC Chapel Hill 2013

- Code for Train/Test Classification of Urban Images.
- Code for Train/Test Regression of Urban Images.
- Code for visualizing predictions on a heatmap overlay.

## Download dataset

To download the dataset composed by images from 2011, 2013 and 2019 press [here](https://drive.google.com/open?id=1bqja-qX_y_LWrbjcxefMdPspdkrdoehC).

## Octave dependencies
 
You should install:
 
    pkg install -forge struct

Image package download from [here](https://octave.sourceforge.io/image/):

    pkg install image-2.12.0.tar.gz
    
## Install libraries for Octave

To use, install:
   
    sudo apt-get install liboctave-dev octave-image

To setup vlfeat, go to root directory:

     MKOCTFILE=mkoctfile make

After the MEX files are successfully compiled (look for them into toolbox/mex/octave/).

## Install libraries for Python

To use, install:

    pip install --user numexpr scipy-stack scikit-image mpi4py networkx
    pip install --user mat4py hdf5storage tables
