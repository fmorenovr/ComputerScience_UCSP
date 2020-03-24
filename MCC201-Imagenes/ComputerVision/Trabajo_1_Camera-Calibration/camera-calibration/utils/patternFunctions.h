#include"calibrateFunctions.h"

int D = 45.0;
float BIAS = 200;

struct compareClass {
  bool operator() (Point pt1, Point pt2) { return (pt1.x < pt2.x);}
} comparePoints;

void drawLines(Mat& img, vector<Point>& SortedPoints){
  for (int ip = 0; ip < SortedPoints.size(); ip++){
    circle(img, Point(SortedPoints[ip].x, SortedPoints[ip].y), 2, BLUE, 2, 8);  
    putText(img,to_string(ip+1),Point(SortedPoints[ip].x, SortedPoints[ip].y),FONT_ITALIC,0.8,ORANGE,2);           
  }
}

void playVideo(Mat video, string name, int w, int h){
  resize(video, video, Size(w, h));
  imshow(name, video);
}

void drawProjectedDistances(Settings& s, Mat view, vector<Point2f> objectPointsProjected){
  vector<Point> objectPointsProjected2Image;
  
  bool neg = false;
  for(int i = 0; i < objectPointsProjected.size(); i++){
    if (objectPointsProjected[i].x < 0 || objectPointsProjected[i].y < 0 ){
      neg = true;
      break;
    }
    objectPointsProjected2Image.push_back(Point((int)objectPointsProjected[i].x, (int)objectPointsProjected[i].y));
  }

  if (neg == false){
    line(view, objectPointsProjected2Image[0], objectPointsProjected2Image[4], Scalar(0,255,255), 4, 8);
    line(view, objectPointsProjected2Image[0 ], objectPointsProjected2Image[ 8], Scalar(255,0,0), 4, 8);
    line(view, objectPointsProjected2Image[0 ], objectPointsProjected2Image[ 9], Scalar(0,255,0), 4, 8);
  }
}

void doFrontoParallel(Settings& s, Mat view, Mat &H, Mat &output, vector<Point2f> sortedPoints, vector<Point3f> positionPoints, Size boardSize, Mat& cameraMatrix, Mat& distCoeffs, vector<Point2f>& objectPointsProjected){
  int pos1 = 0;
  int pos2 = boardSize.width * boardSize.height - 1;
  int pos3 = boardSize.width - 1;
  int pos4 = boardSize.width * (boardSize.height - 1);
  
  // Obtained the sorted points of patterns
        
  vector<Point2f> imagePointsModel;
  imagePointsModel.push_back(sortedPoints[pos1]);
  imagePointsModel.push_back(sortedPoints[pos2]);
  imagePointsModel.push_back(sortedPoints[pos3]);
  imagePointsModel.push_back(sortedPoints[pos4]);

  // Obtaining 3D points from camera calibration
  
  vector<Point3f> objectPointsModel;
  objectPointsModel.push_back(positionPoints[pos1]);
  objectPointsModel.push_back(positionPoints[pos2]);
  objectPointsModel.push_back(positionPoints[pos3]);
  objectPointsModel.push_back(positionPoints[pos4]);
  
  // Estimates the object pose given
  
  Mat rvec(3,1,DataType<double>::type);
  Mat tvec(3,1,DataType<double>::type);
  
  solvePnP(objectPointsModel, imagePointsModel, cameraMatrix, distCoeffs, rvec, tvec);

  objectPointsModel.push_back(Point3f(0.0,0.0,D));
  objectPointsModel.push_back(Point3f(D,0.0,D));
  objectPointsModel.push_back(Point3f(0.0,D,D));
  objectPointsModel.push_back(Point3f(D,D,D));

  objectPointsModel.push_back(Point3f(D  ,0.0,0));
  objectPointsModel.push_back(Point3f(0.0,  D,0));
  objectPointsModel.push_back(Point3f(  D,  D,0));
  
  objectPointsProjected.clear();
  
  projectPoints(objectPointsModel, rvec, tvec, cameraMatrix, distCoeffs, objectPointsProjected);
  
  // verify projected points
  
  vector<Point2f> objectPoints2D;
  for(int i = 0; i < positionPoints.size(); i++){
    objectPoints2D.push_back(Point2f(positionPoints[i].x + 200, positionPoints[i].y + 100));
  }
  
  vector<Point2f> imagePoints2D;
  for(int i = 0; i < sortedPoints.size(); i++){
    imagePoints2D.push_back(Point2f(sortedPoints[i].x, sortedPoints[i].y));
  }

  // Find homography
  
  H = findHomography(imagePoints2D, objectPoints2D);
  if(s.printHMatrix){
    cout << "Homography Matrix: \n" << H << endl;
    // Normalization to ensure that ||c1|| = 1
    double norm = sqrt(H.at<double>(0,0)*H.at<double>(0,0) +
                       H.at<double>(1,0)*H.at<double>(1,0) +
                       H.at<double>(2,0)*H.at<double>(2,0));
    H /= norm;
    Mat c1  = H.col(0);
    Mat c2  = H.col(1);
    Mat c3 = c1.cross(c2);
    Mat R(3, 3, CV_64F);
    for (int i = 0; i < 3; i++){
      R.at<double>(i,0) = c1.at<double>(i,0);
      R.at<double>(i,1) = c2.at<double>(i,0);
      R.at<double>(i,2) = c3.at<double>(i,0);
    }
    
    cout << "R (before polar decomposition):\n" << R << "\ndet(R): " << determinant(R) << endl;
    Mat W, U, Vt;
    SVDecomp(R, W, U, Vt);
    R = U*Vt;
    cout << "R (after polar decomposition):\n" << R << "\ndet(R): " << determinant(R) << endl;
  }
  if(H.cols == 3 && H.rows == 3){
    warpPerspective(view, output, H, view.size());
    flip(output,output,0);
  }
}

void doWarpPerspective(Settings& s, Mat view, Mat &lambda, Mat &output, vector<Point2f> &corners){
  // Input Quadilateral or Image plane coordinates
  Point2f inputQuad[4]; 
  // Output Quadilateral or World plane coordinates
  Point2f outputQuad[4];
  
  // These four pts are the sides of the rect box used as input 
  inputQuad[0]=corners[15]; // left up corner
  inputQuad[1]=corners[19]; // right up corner
  inputQuad[2]=corners[4]; // right down corner
  inputQuad[3]=corners[0]; // left down corner
  
  // The 4 points where the mapping is to be done , from top-left in clockwise order Y, X
  outputQuad[0]=Point2f( MARGIN, MARGIN );
  outputQuad[1]=Point2f( 555, MARGIN);
  outputQuad[2]=Point2f( 555, 430);
  outputQuad[3]=Point2f( MARGIN, 430);
  
  // Get the Perspective Transform Matrix i.e. lambda 
  lambda = getPerspectiveTransform(inputQuad, outputQuad);
  if(s.printLambdaMatrix)
    cout << "Lambda Matrix: \n" << lambda << endl;
  // Apply the Perspective Transform just found to the src image
  warpPerspective(view, output, lambda, output.size());
}

Mat integralThreshold(Mat src, double thresParam){
  int width = src.rows;
  int height = src.cols;

  Mat sumMat, dst;
  src.copyTo(dst);
  integral(src, sumMat);

  //int s = (int) size.width/(16);
	double T = 1 - 0.01*thresParam;
  // perform thresholding
  int S = MAX(width, height)/16;
  int s = S/2;
  int x1, y1, x2, y2, count, sum;

  int *p_y1, *p_y2;
  uchar *p_src, *p_dst;

  for( int i = 0; i < width; ++i){
    y1 = i-s;
    y2 = i+s;

    if (y1 < 0){
      y1 = 0;
    }
    if (y2 >= width) {
      y2 = width-1;
    }

    p_y1 = sumMat.ptr<int>(y1);
    p_y2 = sumMat.ptr<int>(y2);
    p_src = src.ptr<uchar>(i);
    p_dst = dst.ptr<uchar>(i);

    for ( int j = 0; j < height; ++j){
      x1 = j-s;
      x2 = j+s;

      if (x1 < 0) {
        x1 = 0;
      }
      if (x2 >= height) {
        x2 = height-1;
      }

      count = (x2-x1)*(y2-y1);
      sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

      if ((int)(p_src[j] * count) < (int)(sum*(T)))
        p_dst[j] = 0;
        //dst.at<uchar>(i,j) = uchar(255);
      else
        p_dst[j] = 255;
    }
  }
  return dst;
}

bool findRingsGrid(Mat& frame, Size boardSize, vector<Point2f> &sortedPoints, int& Xmax, int& Ymax, int& Xmin, int& Ymin, float& sizeAvgEllipses){
  /* =================================
          Defining Variables
  ===================================*/
  // All Mats
  Mat gray, gaussian, thres, adapThres, intThres;
  // To get all points in all contours
  vector<vector<Point>> contours;
  // To get all hierarchy in contours
  vector<Vec4i> hierarchy;
  // To store all contours that have parent and child
  vector<int> pointsContours;
  // To store index of contours after apply the heuristics
  vector<int> pointsHeuristics;
  // To Store the Center of each Pattern
  vector<Point2f> centerPatterns;
  
  // Difference between centers and sum of ratios
  double centersDiff = 3.0, sumRatiosDiff = 95.0;;
  // Compare bounding box with previous to scale
  double bbDiff = 60.0;
  // Computer corners
  double eps = 20.0;
  // total of patterns to detect
  int numPatterns = boardSize.width*boardSize.height;
  //Define the first bounding box of all patterns
  int xmin = 3000;
  int ymin = 3000;
  int xmax = 0;
  int ymax = 0;
  
  /* =================================
          Re-adjust Parameters
  ===================================*/
  
  if (sizeAvgEllipses > 17.0 && sizeAvgEllipses!=1000.0){
    centersDiff = 3.0;
    bbDiff = 45;
    eps = 30.0;
  }
  else if (sizeAvgEllipses < 9.0) {
    bbDiff = 20;
    eps = 16.0;
  }
  else{
    centersDiff = 2.0;
    bbDiff = 45;
    eps = 20.0;
  }

  /* =================================
              Filters
  ===================================*/
  
  // convert to gray
  cvtColor( frame, gray, CV_BGR2GRAY );
  // suaviza con filtro gaussiano
  GaussianBlur(gray, gaussian, Size(5,5), 0, 0);
  // Threshold
  threshold(gaussian, thres, 100, 255, CV_THRESH_BINARY);
  // Adaptive Threshold
  adaptiveThreshold(gaussian, adapThres, 255,ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,13,12);
  // Integral Threshold
  intThres = integralThreshold(gaussian, 10);
  
  /* =================================
            Find Contours
  ===================================*/
  
  // find contours
  findContours(intThres, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
  // Ellipses contours rect
  vector<RotatedRect> minEllipse(contours.size());
  
  /* =================================
      Find Last Hierarchy Objects
  ===================================*/
  
  // Get all contours with child and parents, [next, previous, first_child, parent]
  for(int i = 0; i < contours.size(); i++){
    if((hierarchy[i][2] >= 0 || hierarchy[i][3] >= 0) && contours[i].size() > 5)
      pointsContours.push_back(i);
  }

  /* =================================
     Heuristics to Detect Ellipses
  ===================================*/

  // For heuristic of weighted size of elipse.
  double sizeAvg = 0.0;

  // To verify which ellipses are detected
  bool ellipseDetected[contours.size()] = {false};

  // Iterate all contours with child and parents
  for(int i = 0; i < pointsContours.size(); i++){
    int currIndex = pointsContours[i];
    if(hierarchy[currIndex][2] == -1 ){ // dont have child
      // This is the last circles in hierarchy
      if(!ellipseDetected[currIndex] && contours[currIndex].size() > 5){
        minEllipse[currIndex] = fitEllipse(Mat(contours[currIndex]));
        ellipseDetected[currIndex] = true;
      }
      // Verify is his father is ellipse
      int parentIndex = hierarchy[pointsContours[i]][3];
      if(!ellipseDetected[parentIndex] && contours[parentIndex].size() > 5){
        minEllipse[parentIndex] = fitEllipse(Mat(contours[parentIndex]));
        ellipseDetected[parentIndex] = true;
      }
      // Verify if current and his father are close
      double val = norm(minEllipse[currIndex].center - minEllipse[parentIndex].center);
      if(val < centersDiff){
        // Ponderate size of father.
        double szFather = (minEllipse[parentIndex].size.height + minEllipse[parentIndex].size.width)/2.0;
        double szChild = (minEllipse[currIndex].size.height + minEllipse[currIndex].size.width)/2.0;
        // Heuristic: if size of father is too bigger in comparizon to child. 2.9 setted experimentally
        if(szFather/szChild < 2.9){
          // Heuristic: If Size of father ellipse increasse dramatically in comparison with previous frame.
          if(szFather - sizeAvgEllipses < sumRatiosDiff ){
            //Heuristic: M. Bounding Box Heuristic:
            Point curPoint = minEllipse[currIndex].center;
            if((curPoint.x - Xmax < bbDiff) &&(Xmin - curPoint.x < bbDiff) && (curPoint.y - Ymax < bbDiff) && (Ymin - curPoint.y < bbDiff) ){
              // Here it is the centerpoint of pattern.
              pointsHeuristics.push_back(currIndex);
              pointsHeuristics.push_back(parentIndex);
              // Save the Center point
              centerPatterns.push_back(minEllipse[currIndex].center);
              // Compute the MBB of Current Frame;
              if(minEllipse[currIndex].center.x < xmin)
                xmin = minEllipse[currIndex].center.x;
              if(minEllipse[currIndex].center.x > xmax)
                xmax = minEllipse[currIndex].center.x;
              if(minEllipse[currIndex].center.y < ymin)
                ymin = minEllipse[currIndex].center.y;
              if(minEllipse[currIndex].center.y > ymax)
                ymax = minEllipse[currIndex].center.y;
              // Acumm Size average
              sizeAvg += szFather;
            }
          }
        }
      }
    }
  }

  sizeAvgEllipses = sizeAvg/(float)pointsHeuristics.size();
  if(pointsHeuristics.size() != 0){
    Xmax = xmax;
    Xmin = xmin;
    Ymax = ymax;
    Ymin = ymin;
  }
  // Restart initial conditions
  if(pointsHeuristics.size() == 0){
    sizeAvgEllipses = 1000.0;
  }
  
  /* =================================
        Purge and sort patterns
  ===================================*/
  
  /* =================================
          Draw patterns
  ===================================*/

  //playVideo(gray, "Gray Scale", WIDTH, HEIGHT);
  //playVideo(gaussian, "Gaussian Filter", WIDTH, HEIGHT);
  //playVideo(thres, "Threshold", WIDTH, HEIGHT);
  //playVideo(adapThres, "Adaptive Threshold", WIDTH, HEIGHT);
  //playVideo(intThres, "Integral Threshold", WIDTH, HEIGHT);

  rectangle(frame, Point2f(Xmin, Ymin), Point2f(Xmax, Ymax), PURPLE, 5, 8);
  for(int i = 0; i < pointsHeuristics.size(); i++){
    ellipse( frame, minEllipse[pointsHeuristics[i]], SKYBLUE, 2, 8 );
    putText(frame, to_string(i+1), centerPatterns[i], FONT_HERSHEY_DUPLEX, 0.65, SOFTGREEN, 4);
  }
  sortedPoints = centerPatterns;
  return (sortedPoints.size() == numPatterns);
}

vector<Point> findConcentricCenters( vector<RotatedRect> minRect,
                            vector<Point> centers,
                            vector<float> diag,
                            int contours_filtered)
{
    vector<Point> tempCnts;
    Point2f rect_points[4];
    int centerx;
    int centery;
    float diagtemp;
    float errormaxDiam = 3.5;
    float errormax = 3; 
    float dist;
    

    for( int i = 0; i< contours_filtered; i++ )
    {
        
        minRect[i].points( rect_points );
        centerx = 0.5*(rect_points[0].x + rect_points[2].x);
        centery = 0.5*(rect_points[0].y + rect_points[2].y);
        centers[i] = Point(centerx,centery); 
        diag[i] =  sqrt((rect_points[2].x - rect_points[0].x)*(rect_points[2].x - rect_points[0].x)+(rect_points[2].y - rect_points[0].y)*(rect_points[2].y - rect_points[0].y));
    }

    for(int p = 0; p < contours_filtered; p++){
        minRect[p].points( rect_points );
        centerx = 0.5*(rect_points[0].x + rect_points[2].x);
        centery = 0.5*(rect_points[0].y + rect_points[2].y);
        diagtemp = sqrt(pow((rect_points[2].x - rect_points[0].x),2)+pow((rect_points[2].y - rect_points[0].y),2));

        for(int k = 0; k < contours_filtered; k++)
        {
            if(k != p){
                dist = sqrt((centerx - centers[k].x)*(centerx - centers[k].x) + (centery - centers[k].y)*(centery - centers[k].y));
                if((dist <= errormax) && (diag[k] - diagtemp)  > errormaxDiam)
                {   
                    tempCnts.push_back(Point(0.5*(centerx + centers[k].x), 0.5*(centery + centers[k].y)));
                    break;
                }
            }
        }
    }

    return tempCnts;   
}

int PointsInside(vector<Point>& P, vector<Point>& pIns,float m, float b, int np)
{
    int     count = 0;
    float   y_hat;
    float   K = 10.0;

    for (int i = 0; i < P.size(); i++)
    {
        y_hat = (float)P[i].x * m + b;  
        if (abs(y_hat - (float)P[i].y) < K)
        {   
            count++;
            pIns.push_back(Point(P[i].x, P[i].y));
        }           
    }

    if (count == np)
    {
        for (int i = 0; i < pIns.size(); i++)
        {
            if (find(P.begin(),P.end(),Point(pIns[i].x, pIns[i].y)) != P.end())
                P.erase(find(P.begin(),P.end(),Point(pIns[i].x, pIns[i].y)));           
        }
    }
    return count;
}

vector<vector<Point>> Points2RectOfPoints(vector<Point> allPoints2D, Size BoardSize)
{
    vector<Point> tempPoints(allPoints2D.size());

    for (int i = 0; i < allPoints2D.size(); i++)
    {
        tempPoints[i].x  = allPoints2D[i].x;
        tempPoints[i].y  = allPoints2D[i].y;
    }

    vector<Point> lhor(2);

    float m;
    float b;

    int npoints;

    vector<vector<Point>> RectOfPoinst;
    vector<Point> pIns;

    for (int i = 0; i < allPoints2D.size(); i++)
    {
        for (int j = 0; j < allPoints2D.size(); j++)
        {
            if (i != j)
            {
                m       = (float)(allPoints2D[i].y - allPoints2D[j].y) / ((float)(allPoints2D[i].x - allPoints2D[j].x));
                b       = (float)allPoints2D[i].y - allPoints2D[i].x * m;
                
                pIns.clear();

                npoints = PointsInside(tempPoints, pIns, m, b, BoardSize.width);
                if (npoints == BoardSize.width)
                {
                    RectOfPoinst.push_back(pIns);
                }
            }
        }
    }
    return RectOfPoinst;
}

void SortingPoints(vector<Point>  tempCnts, vector<Point>& nextCenters, Size BoardSize){
  if (tempCnts.size() == (BoardSize.width * BoardSize.height)){
    vector<vector<Point>> RofP = Points2RectOfPoints(tempCnts,BoardSize);
    vector<vector<int>> Ypos;

    vector<Point> minYVec;
    vector<int> Y;
    for (int ri = 0; ri < RofP.size(); ri++){
      Y.clear();
      for(int iY = 0; iY < RofP[ri].size(); iY++){
        Y.push_back(RofP[ri][iY].y);
      }

      int maxY = *max_element(Y.begin(), Y.end());
      minYVec.push_back(Point(ri, maxY));
    }

    vector<vector<Point>> SortedRofP;
    vector<Point> SortedR;

    while (minYVec.size() != 0){
      SortedR.clear();

      int Global_minY     = minYVec[0].y;
      int Global_minY_Pos = minYVec[0].x;

      if (minYVec.size() > 1){
        for (int iY = 1; iY < minYVec.size(); iY++){
          if (minYVec[iY].y < Global_minY){
            Global_minY     = minYVec[iY].y;
            Global_minY_Pos = minYVec[iY].x;
          }
        }  

        for (int iP = 0; iP < RofP[Global_minY_Pos].size(); iP++){
          SortedR.push_back(Point(RofP[Global_minY_Pos][iP].x, RofP[Global_minY_Pos][iP].y));
        }
        SortedRofP.push_back(SortedR);

        if (find(minYVec.begin(),minYVec.end(),Point(minYVec[Global_minY_Pos].x, minYVec[Global_minY_Pos].y)) != minYVec.end()){
          minYVec.erase(find(minYVec.begin(),minYVec.end(),Point(minYVec[Global_minY_Pos].x, minYVec[Global_minY_Pos].y)));
         }
      } else {
        for (int iP = 0; iP < RofP[0].size(); iP++){
          SortedR.push_back(Point(RofP[0][iP].x, RofP[0][iP].y));
        }
        SortedRofP.push_back(SortedR);
        minYVec.erase(minYVec.begin(),minYVec.begin());
        break;
      }
    }

    for (int ir = 0; ir < SortedRofP.size(); ir ++){
      sort(SortedRofP[ir].begin(), SortedRofP[ir].end(), comparePoints);
    }

    nextCenters.clear();
    for (int ir = 0; ir < SortedRofP.size(); ir ++){
      for (int iP = 0; iP < SortedRofP[ir].size(); iP++){
        nextCenters.push_back(SortedRofP[SortedRofP.size() - ir - 1][iP]);
      }
    }
  } else {
    nextCenters.clear();
  }
}

bool trancking(  vector<Point>  tempCnts, vector<Point>  prevCenters, vector<Point>& nextCenters, Size BoardSize, int countFrame ){
  int centerx;
  int centery;
  int dist_centers;
  int countc      = 0;
  
  int center_jump = 0;

  float minc;
  int   Nrst;

  Scalar color = Scalar( 255, 250, 50);

  bool jump = false;

  if (countFrame == 0){
    return true;
  } else{
    int nextX;
    int nextY;

    if (prevCenters.size() != (BoardSize.height * BoardSize.width)){
      return true;
    }
    for (int i = 0; i < prevCenters.size(); i++){
      centerx = prevCenters[i].x;
      centery = prevCenters[i].y;

      minc    = 1e5;

      for(int iT =  0; iT < tempCnts.size(); iT++){
        dist_centers = sqrt((centerx - tempCnts[iT].x)*(centerx - tempCnts[iT].x) + 
                            (centery - tempCnts[iT].y)*(centery - tempCnts[iT].y));
        if(dist_centers < minc){
          minc  = dist_centers;
          Nrst  = iT;
          nextX = tempCnts[iT].x;
          nextY = tempCnts[iT].y;
        }
      }    

      if (minc < 10){
        nextCenters[i] = Point(nextX, nextY);
      }     

      else 
      {
        return true;
      }
    }
    return false;
  }
}

Mat findCenters(Mat frame, Size BoardSize, vector<Point>& sortedPoints){
    int prevcenterx = 0;
    int prevcentery = 0;
    Mat prev;
    int maxdistance = 0;

    vector<Vec4i> hierarchy;
    vector<Point> newPoints(BoardSize.width * BoardSize.height);
    vector<Point> tempCnts;
    vector<vector<Point>> contours;

    Mat gray, gaussian, thres, intThres, adapThres, result;
    frame.copyTo(result);

    cvtColor( frame, gray, CV_BGR2GRAY );
    GaussianBlur(gray, gaussian, Size(5,5), 0, 0);
    threshold(gaussian, thres, 100, 255, CV_THRESH_BINARY);
    adaptiveThreshold(gaussian, adapThres, 255,ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,13,12);
    intThres = integralThreshold(gaussian, 10);

    findContours(intThres, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    vector<Point> centers(contours.size());
    vector<float>  diag(contours.size());

    vector<RotatedRect> minRect( contours.size() );
    int contours_filtered = 0;
    
    for( int i = 0; i < contours.size(); i++ ){
      if(contours[i].size() > 10  && contours[i].size() < 280){
        minRect[contours_filtered]    = minAreaRect( Mat(contours[i]) );
        contours_filtered++;
      }
    }

    Scalar color;
    int   centerx;
    int   centery;
    float minc;
    float dist_centers;

    color = Scalar( 255, 250, 50);
    int countc=0;

    tempCnts    = findConcentricCenters(minRect, centers, diag, contours_filtered);

    for(int i=0; i < tempCnts.size(); i++){
      circle(result, Point(tempCnts[i].x, tempCnts[i].y), 2, Scalar(0,255,0), 2, 8);  
    }

    bool Cjump = trancking(tempCnts, sortedPoints, newPoints, BoardSize, 1);   
    
    if (Cjump == true){
      if (tempCnts.size() < (BoardSize.width * BoardSize.height)){
        sortedPoints.clear();
        return result;
      }
      vector<Point> newPoints;

      SortingPoints(tempCnts, newPoints, BoardSize);

      if (newPoints.size() == (BoardSize.width * BoardSize.height)){
        sortedPoints = newPoints;
      } else{
        sortedPoints.clear();
      }
    } else {
      sortedPoints = newPoints;
    }
    return result;
}

// iterative calibration
bool IterativeRefinement(Settings& s, vector<Mat> imgsToCalib, double& rms, Mat& cameraMatrix , Mat& distCoeffs, vector<vector<Point2f>> imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs, vector<float>& reprojErrs, double totalAvgErr, vector<Point3f> objectPoints, float grid_width, bool release_object) {
  Mat dst, temp;

  vector<vector<Point3f>> NewObjectPointsModel(imagePoints.size());
  vector<vector<Point2f>> imagePointsReProyected(imagePoints.size());

  for(int i = 0; i < imagePoints.size(); i++){
    temp = imgsToCalib[i].clone();

    vector<Point2f> ObjectPointsModel2D;
    for(int j = 0; j < objectPoints.size(); j++){
      ObjectPointsModel2D.push_back(Point2f(objectPoints[j].x + BIAS, D * (s.boardSize.height - 1) -  objectPoints[j].y + BIAS/2));
    }

    vector<Point2f> imagePointsModel2D;

    for(int j = 0; j < imagePoints[i].size(); j++){
      imagePointsModel2D.push_back(Point2f(imagePoints[i][j].x, imagePoints[i][j].y));
    }

    Mat H = findHomography(imagePointsModel2D, ObjectPointsModel2D);

    if (H.cols != 3 || H.rows != 3){
      NewObjectPointsModel[i] = objectPoints;
      projectPoints(NewObjectPointsModel[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePointsReProyected[i]);
      continue;
    }
    undistort(temp, imgsToCalib[i], cameraMatrix, distCoeffs);
    warpPerspective(imgsToCalib[i], dst, H, imgsToCalib[i].size());
    //flip(dst,dst,0);

    vector<Point> SortedPoints2;
    
    int c = 0;

    bool foundCircles = findCirclesGrid(dst, s.boardSize, SortedPoints2, CALIB_CB_ASYMMETRIC_GRID );
    //Mat result2 = findCenters(dst, s.boardSize, SortedPoints2);

    if(foundCircles){// (SortedPoints2.size() == (s.boardSize.width * s.boardSize.height)){
      //drawLines(result2, SortedPoints2);
      vector<Point3f> FimgPoint; 
      for(int j = 0; j < SortedPoints2.size(); j++){
        FimgPoint.push_back(Point3f(SortedPoints2[j].x - BIAS , D * (s.boardSize.height - 1) - (SortedPoints2[j].y - BIAS/2), 0.0));
      }
      NewObjectPointsModel[i] = FimgPoint;
    }
    else{
      NewObjectPointsModel[i] = objectPoints;
    }
    
    projectPoints(NewObjectPointsModel[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePointsReProyected[i]);
    
  }

  for (int i = 0; i < imagePointsReProyected.size(); i++){
    for (int j = 0; j < imagePointsReProyected[i].size(); j++){
      imagePointsReProyected[i][j].x = (imagePoints[i][j].x)*0.5 + (imagePointsReProyected[i][j].x)*0.5;
      imagePointsReProyected[i][j].y = (imagePoints[i][j].y)*0.5 + (imagePointsReProyected[i][j].y)*0.5;
    }
  }
  
  // de aqui en adelante ya esta verificado
  
  vector<vector<Point3f>> objectPointsNew(1);
  calcBoardCornerPositions(s.boardSize, s.squareSize, objectPointsNew[0], s.calibrationPattern);
  objectPointsNew[0][s.boardSize.width - 1].x = objectPointsNew[0][0].x + grid_width;
  objectPoints = objectPointsNew[0];
  
  objectPointsNew.resize(imagePoints.size(),objectPointsNew[0]);

  bool ok = runCalibrationAndSave(s, temp.size(), rms, cameraMatrix, distCoeffs, imagePointsReProyected, objectPoints, grid_width, false);

  return ok;
}

bool runIterativeCalibration(Settings& s, vector<Mat> imgsToCalib, Size imgSize, double& rms, Mat& cameraMatrix, Mat& distCoeffs, vector<vector<Point2f>> imagePoints, vector<Point3f>& objectPoints, float grid_width, bool release_object) {
  vector<Mat> rvecs, tvecs;
  vector<float> reprojErrs;

  double totalAvgErr = 0;
  bool ok;
  
  ok = runCalibration(s, imgSize, rms, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs, totalAvgErr, objectPoints, grid_width, release_object);
  
  for (int i = 0; i < s.numItersToCalibrate; i ++){
    cout << " ====================================================== " << endl;
    cout << " ****************************************************** " << endl;
    cout << "    Iteration: " << i+1 << endl;
    cout << " ****************************************************** " << endl;
    ok = IterativeRefinement(s, imgsToCalib, rms, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs, totalAvgErr, objectPoints, grid_width, release_object);
    cout << " ======================================================= " << endl;
    if(!ok){
      cout << "Finalizo calibracion con defectos !"<<endl;
    }
  }
  
  return ok;
}
