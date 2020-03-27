#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <signal.h>

#include <opencv2/opencv.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

// g++ fast3DRecognition.cpp -std=c++11 -o out `pkg-config opencv --cflags --libs` `pkg-config freenect2 --cflags --libs` 

using namespace std;
using namespace cv;

int WIDTH = 650;
int HEIGHT = 550;

Size newsz = Size(WIDTH, HEIGHT);

bool protonect_shutdown = false; // Whether the running application should shut down.

void sigint_handler(int s){
  protonect_shutdown = true;
}

int main(){
  cout << "Streaming from Kinect One sensor!" << endl;

  //! [context]
  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *dev = 0;
  libfreenect2::PacketPipeline *pipeline = 0;
  //! [context]

  //! [discovery]
  if(freenect2.enumerateDevices() == 0){
    cout << "no device connected!" << endl;
    return -1;
  }

  string serial = freenect2.getDefaultDeviceSerialNumber();

  cout << "SERIAL: " << serial << endl;

  if(pipeline){
    //! [open]
    dev = freenect2.openDevice(serial, pipeline);
    //! [open]
  } else {
    dev = freenect2.openDevice(serial);
  }

  if(dev == 0){
    cout << "failure opening device!" << endl;
    return -1;
  }

  signal(SIGINT, sigint_handler);
  protonect_shutdown = false;

  //! [listeners]
  libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color |
                                                libfreenect2::Frame::Depth |
                                                libfreenect2::Frame::Ir);
  libfreenect2::FrameMap frames;

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);
  //! [listeners]

  //! [start]
  dev->start();

  cout << "device serial: " << dev->getSerialNumber() << endl;
  cout << "device firmware: " << dev->getFirmwareVersion() << endl;
  //! [start]

  //! [registration setup]
  libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4); // check here (https://github.com/OpenKinect/libfreenect2/issues/337) and here (https://github.com/OpenKinect/libfreenect2/issues/464) why depth2rgb image should be bigger
  //! [registration setup]

  Mat rgbMat, depthMat, irMat, undistMat, regisMat, depth2rgbMat;
  
  vector<int> compressionParams_PNG;
  compressionParams_PNG.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compressionParams_PNG.push_back(9);

  while(!protonect_shutdown){
    
    /* =================================
            Defining Variables
    ===================================*/
  
    listener.waitForNewFrame(frames);
    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo(rgbMat);
    Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(irMat);
    Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depthMat);

    resize(rgbMat, rgbMat, newsz, 0, 0, INTER_CUBIC);
    resize(irMat, irMat, newsz, 0, 0, INTER_CUBIC);
    resize(depthMat, depthMat, newsz, 0, 0, INTER_CUBIC);

    /* =================================
            Make RGB and DEPTH
    ===================================*/

    registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);
    
    /* =================================
               Frames to Mat
    ===================================*/

    Mat(undistorted.height, undistorted.width, CV_32FC1, undistorted.data).copyTo(undistMat);
    Mat(registered.height, registered.width, CV_8UC4, registered.data).copyTo(regisMat);
    Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data).copyTo(depth2rgbMat);

    /* =================================
                   Resize
    ===================================*/

    resize(regisMat, regisMat, newsz);
    resize(depth2rgbMat, depth2rgbMat, newsz);
    resize(undistMat, undistMat, newsz);

    /* =================================
            CV_32FC1 to CV8UC4
    ===================================*/
    
    Mat aux(irMat.rows, irMat.cols, CV_8UC4, irMat.data);
    Mat aux1(depthMat.rows, depthMat.cols, CV_8UC4, depthMat.data);
    Mat aux2(depth2rgbMat.rows, depth2rgbMat.cols, CV_8UC4, depth2rgbMat.data);
    Mat aux3(undistMat.rows, undistMat.cols, CV_8UC4, undistMat.data);

    /* =================================
               Saving Images
    ===================================*/
    
    imwrite("../images/rgb.png", rgbMat);
    imwrite("../images/depth.png", aux1 / 4096.0f, compressionParams_PNG);

    /* =================================
               Showing Images
    ===================================*/
    
    imshow("rgb", rgbMat);
    imshow("depth", depthMat / 4096.0f);
    imshow("ir", irMat / 4096.0f);
    
    imshow("registered", regisMat);
    imshow("undistorted", undistMat / 4096.0f);
    imshow("depth2RGB", depth2rgbMat / 4096.0f);
    
    /* =================================
         Waiting keys to do something
    ===================================*/
    
    int key = waitKey(1);
    // shutdown on escape
    protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); 
    listener.release(frames);
  }

  dev->stop();
  dev->close();

  delete registration;

  cout << "Streaming Ends!" << endl;
  return 0;
}
