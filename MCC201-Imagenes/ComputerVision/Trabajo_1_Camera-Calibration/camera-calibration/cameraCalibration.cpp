#include"utils/ellipsesFunctions.h"

// g++ cameraCalibration.cpp `pkg-config --cflags --libs opencv`

// g++ cameraCalibration.cpp `pkg-config --cflags --libs opencv` && ./a.out

int main(int argc, char* argv[]){
  /* =================================
            Reading File
  ===================================*/
  Settings s;
  FileStorage fs("default.xml", FileStorage::READ);
  
  if (!fs.isOpened()){
    cout << "Could not open the configuration file."<< endl;
    return -1;
  }
  fs["Settings"] >> s;
  fs.release();

  if (!s.goodInput){
    cout << "Invalid input detected. Application stopping. " << endl;
    return -1;
  }

  /* =================================
            Defining Variables
  ===================================*/
  
  // Half of search window for cornerSubPix
  int winSize = 11;
  // grid of patterns
  float grid_width = s.squareSize * (s.boardSize.width - 1);
  // centers vector for each frame
  vector<vector<Point2f> > imagePoints;
  // Represet a matrix using Mat ( camera, distorsion coef)
  Mat cameraMatrix, distCoeffs;
  // RMS
  double rms;
  // camera mode
  int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
  // time elapsed between taked frame
  clock_t prevTimestamp = 0;
  // bounding box of all patterns
  int Xmax = 1000;
  int Ymax = 1000;
  int Xmin = 0;
  int Ymin = 0;
  int XmaxFronto = 1000;
  int YmaxFronto = 1000;
  int XminFronto = 0;
  int YminFronto = 0;
  int XmaxWarp = 1000;
  int YmaxWarp = 1000;
  int XminWarp = 0;
  int YminWarp = 0;
  // Average size of ellipses
  float sizeAvgEllipses = 1000.0;
  // number of frames
  int frameCounts = 0;
  // time elapsed
  double acumulatedTime = 0;
  // size of each frame
  Size imageSize;
  // frames
  Mat view, uncalibrate, calibrate, frame, fronto, warp;
  // center of patterns
  vector<Point2f> sortedPoints;
  vector<Point3f> positionPoints;
  vector<Point2f> objectPointsProjected;
  // if find all patterns
  bool found;
  // Array of frames to calib iterative
  vector<Mat> imgToCalib;

  /* =================================
         Reading Video
  ===================================*/

  while(true){
    // 
    bool blinkOutput = false;
    // next frame
    s.inputCapture.read(view);
    // Evalute if is necessary to mirror vertically
    if(s.flipVertical) flip( view, view, 0 );
    // copy to frame do warp and fronto Parallel
    view.copyTo(frame);
    // frame to work
    view.copyTo(uncalibrate);
    // Format input image.
    imageSize = uncalibrate.size();

    /* =================================
             Calibration Process
    ===================================*/

    //-----  If no more image, or got enough, then stop calibration and show result -------------
    if( mode == CAPTURING && imagePoints.size() >= (size_t)s.nrFrames ){
      if(runIterativeCalibration(s, imgToCalib, imageSize, rms, cameraMatrix, distCoeffs, imagePoints, positionPoints, grid_width, false)){
        mode = CALIBRATED;
      } else {
        mode = DETECTION;
      }
    }
    
    // If there are no more images stop the loop
    if(uncalibrate.empty()) {
      // if calibration threshold was not reached yet, calibrate now
      if(runIterativeCalibration(s, imgToCalib, imageSize, rms, cameraMatrix, distCoeffs, imagePoints, positionPoints, grid_width, false)){
        mode = CALIBRATED;
      }
      cout << "Video finished" << endl;
      cout << "Total Frames: " << frameCounts << endl;
      cout << "Execution time per frame average: " << acumulatedTime/frameCounts << endl;
      break;
    }

    /* =================================
                 Set timer
    ===================================*/
    
    auto startTime = chrono::high_resolution_clock::now();

    /* =================================
       Patterns Recognition heuristics
    ===================================*/
    
    switch( s.calibrationPattern ) {
      case Settings::CHESSBOARD:
        found = findChessboardCorners( uncalibrate, s.boardSize, sortedPoints);
        break;
      case Settings::CIRCLES_GRID:
        found = findCirclesGrid( uncalibrate, s.boardSize, sortedPoints );
        break;
      case Settings::ASYMMETRIC_CIRCLES_GRID:
        found = findCirclesGrid( uncalibrate, s.boardSize, sortedPoints, CALIB_CB_ASYMMETRIC_GRID );
        break;
      case Settings::RINGS_GRID:
        //found = findRingsGrid( uncalibrate, s.boardSize, sortedPoints, Xmax, Ymax, Xmin, Ymin, sizeAvgEllipses);
        found = findRingsGrid1( uncalibrate, s.boardSize, sortedPoints);
        break;
      default:
        found = false;
        break;
    }
    
    /* =================================
          Stop timer and count frames
    ===================================*/
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime-startTime).count();
    acumulatedTime += (double)duration;
    frameCounts += 1;
    
    /* =================================
         When all patterns are found
    ===================================*/
    
    if(found) {
      /* =================================
         Capturing Frames to Calibrate
      ===================================*/
      // improve the found corners' coordinate accuracy for chessboard
      if( s.calibrationPattern == Settings::CHESSBOARD){
        Mat gray;
        cvtColor(uncalibrate, gray, COLOR_BGR2GRAY);
        cornerSubPix( gray, sortedPoints, Size(winSize,winSize), Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001 ));
      }
      // Get frames to calibrate
      if( mode == CAPTURING &&  (!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC) && frameCounts % s.frameFrecuencyCount == 0) {
        imagePoints.push_back(sortedPoints);
        prevTimestamp = clock();
        blinkOutput = s.inputCapture.isOpened();
        imgToCalib.push_back(frame);
      }
      /* =================================
            Draw Corners of Patterns
      ===================================*/
      drawChessboardCorners( uncalibrate, s.boardSize, Mat(sortedPoints), found);

      /* =================================
            Doing Warp Perspective
      ===================================*/
    
      if(s.showWarpPerspective) {
        Mat warpMatrix, lambda;
        warp = frame.clone();
        doWarpPerspective(s, warp, lambda, warpMatrix, sortedPoints);
        /*vector<Point2f> sortedPointsWarp;
        bool foundWarp = findRingsGrid( frontoMatrix, s.boardSize, sortedPointsWarp, XmaxWarp, YmaxWarp, XminWarp, YminWarp, sizeAvgEllipses);
        if (foundwarp){
          drawChessboardCorners( warpMatrix, s.boardSize, Mat(sortedPointsWarp), foundWarp);
        }*/
        playVideo(warpMatrix, "Warp Perspective", WIDTH, HEIGHT);
      }
      
      /* =================================
            Doing Fronto Parallel
      ===================================*/
    
      if(mode == CALIBRATED && s.showFrontoParallel){
        Mat frontoMatrix, H;
        fronto = uncalibrate.clone();
        doFrontoParallel(s, fronto, H, frontoMatrix, sortedPoints, positionPoints, s.boardSize, cameraMatrix, distCoeffs, objectPointsProjected);
        /*if(H.cols == 3 && H.rows == 3){
          vector<Point2f> sortedPointsFronto;
          bool foundFronto = findRingsGrid( frontoMatrix, s.boardSize, sortedPointsFronto, XmaxFronto, YmaxFronto, XminFronto, YminFronto, sizeAvgEllipses);
          if (foundFronto){
            drawChessboardCorners( frontoMatrix, s.boardSize, Mat(sortedPointsFronto), foundFronto);
          }
        }*/
        playVideo(frontoMatrix, "Fronto Parallel", WIDTH, HEIGHT);
      }
    }
    
    /* =================================
                Set Text
    ===================================*/

    string msg = (mode == CALIBRATED) ? "Uncalibrated" : "Press 'g' to start";
    int baseLine = 0;
    Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
    Point textOrigin(uncalibrate.cols - 2*textSize.width - 10, uncalibrate.rows - 2*baseLine - 10);

    if( mode == CAPTURING ){
      if(s.showUndistorsed){
        msg = format( "%d/%d Undist", (int)imagePoints.size(), s.nrFrames );
      } else {
        msg = format( "%d/%d", (int)imagePoints.size(), s.nrFrames );
      }
    }
      
    /* =================================
       Show Undistort after Calibrate
    ===================================*/
    
    if( mode == CALIBRATED && s.showUndistorsed ) {
      calibrate = uncalibrate.clone();
      if (s.useFisheye){
        fisheye::undistortImage(uncalibrate, calibrate, cameraMatrix, distCoeffs);
      } else {
        undistort(uncalibrate, calibrate, cameraMatrix, distCoeffs);
      } 
      putText(calibrate, "Calibrated", textOrigin, 1, 1, GREEN);
      playVideo(calibrate, "Camera Calibrated (undistorted)", WIDTH, HEIGHT);
    }
    
    if( blinkOutput )
      bitwise_not(uncalibrate, uncalibrate);
    
    /* =================================
       Show Distort and angle to fronto
    ===================================*/
    if(mode == CALIBRATED && s.showFrontoParallel && found){
      drawProjectedDistances(s, uncalibrate, objectPointsProjected);
    }
    putText(uncalibrate, msg, textOrigin, 1, 1, RED);
    playVideo(uncalibrate, "Camera Uncalibrated (distorted)", WIDTH, HEIGHT);
    
    Mat infoVideo = Mat::zeros( WIDTH/2.5, HEIGHT*1.5, CV_8UC3 );
    putText(infoVideo, "Execution time per Frame (ms): " + to_string(acumulatedTime/frameCounts), Point2f(15,70),FONT_HERSHEY_PLAIN,1.5, YELLOW, 2);
    putText(infoVideo, "Total No. Frames: " + to_string(frameCounts), Point2f(15,150),FONT_HERSHEY_PLAIN,1.5, YELLOW, 2);
    //imshow("Info Frames", infoVideo);
    
    /* =================================
         Waiting keys to do something
    ===================================*/
    
    char key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

    if( s.inputCapture.isOpened() && key == 'g' ) {
      mode = CAPTURING;
      imagePoints.clear();
    }
    
    if(key == 27 || key == 'q' || key == 'Q' )
      break;
  }
  cout << "Reading Ends!" << endl;
  return 0;
}
