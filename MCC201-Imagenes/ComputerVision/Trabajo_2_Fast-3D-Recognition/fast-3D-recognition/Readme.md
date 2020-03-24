# Fast 3D recognition using viewpoint feature histogram

## KINECT CAMERA INSTALL LIBFREENECT

* First, You should need java installed in your computer.

* Then, We need to install libfreenect2 to use Kinect camera to get data in real time.

* Install all this things:

      sudo apt-get install libva-dev libjpeg-dev build-essential cmake pkg-config libusb-1.0-0-dev libturbojpeg libjpeg-turbo8-dev libglfw3-dev libopenni2-dev libturbojpeg0-dev libopencv-dev meshlab libproj-dev

* Then, clone:

      git clone https://github.com/OpenKinect/libfreenect2.git
      cd libfreenect2

* Next, build and install:

      mkdir build && cd build
      cmake ..  -DENABLE_CXX11=ON -DENABLE_OPENCL=ON
      make
      sudo make install

* Then go to src and compile with:

      ./compileFile

## Point Cloud Generation - INSTALL PCL

*  Install:

       sudo apt install libpcl-dev

* Then, run:

      ./compileFile
