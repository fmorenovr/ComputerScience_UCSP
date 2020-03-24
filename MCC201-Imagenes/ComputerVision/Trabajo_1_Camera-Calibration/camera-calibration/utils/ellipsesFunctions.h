#include"patternFunctions.h"

void ellipsePurge(Mat &morphology,
                  const vector<RotatedRect> &elipses,
                  vector<Point2f> &centers,
                  int width, int height){
    vector<int> ellipses_to_verify( elipses.size() );
    Point2f center;
    int counter_verify;
    float x_t0, y_t0;
    float x_t1, y_t1;
    float disX, disY;
    float sumx, sumy;

    int counter = 0;
    float holgura = 1;
    float distance;

    // Draw ellipse|
    Mat ellipses = Mat::zeros( morphology.size(), CV_8UC3 );

    for( size_t i = 0; i< elipses.size()-1; i++ ){
        if( elipses[i].size.width > width/8 || elipses[i].size.height > height/6 ||
            elipses[i].size.width < 6       || elipses[i].size.height < 6        ||
            ellipses_to_verify[i] == -1){}
        else{
            x_t0 = elipses[i].center.x;
            y_t0 = elipses[i].center.y;

            counter_verify = 1;
            sumx = x_t0;
            sumy = y_t0;

            for( size_t j = i+1; j< elipses.size()-1; j++ ){
                x_t1 = float(elipses[j].center.x);
                y_t1 = float(elipses[j].center.y);

                disX = x_t0 - x_t1;
                disY = y_t0 - y_t1;
                distance = sqrt(disX*disX + disY*disY);

                if(distance <= holgura){
                    counter++;
                    counter_verify++;
                    sumx += x_t1;
                    sumy += y_t1;
                    ellipses_to_verify[j] = -1;
                }
            }

            if(counter_verify > 1){
                center = Point2f(sumx/counter_verify, sumy/counter_verify);
                circle(ellipses, center, 2,Scalar(35,255,75), -1, 8, 0);
                ellipse( ellipses, elipses[i], Scalar(253,35,255), 2, 8 );

                // Add to ellipse list
                centers.push_back(center);
            }

        }
    }
}

bool DoesntRectangleContainPoint(RotatedRect &rectangle, Point2f &point) {
    //Get the corner points.
    Point2f corners[4];
    rectangle.points(corners);

    //Convert the point array to a vector.
    //https://stackoverflow.com/a/8777619/1997617
    Point2f* lastItemPointer = (corners + sizeof corners / sizeof corners[0]);
    vector<Point2f> contour(corners, lastItemPointer);

    //Check if the point is within the rectangle.
    double indicator = pointPolygonTest(contour, point, false);
    return (indicator < 0);
}


vector<Point2f> get_intermediate_sorted_points(Point p, vector<Point2f> intermediate_vectors)
{
    vector<Point2f> result;
    if(intermediate_vectors.size() == 1)
    {
        result.push_back(intermediate_vectors[0]);
        return result;
    }

    if(intermediate_vectors.size() == 3 )
    {
        float d1 = sqrt(pow(p.x - intermediate_vectors[0].x,2) + pow(p.y - intermediate_vectors[0].y,2) );
        float d2 = sqrt(pow(p.x - intermediate_vectors[1].x,2) + pow(p.y - intermediate_vectors[1].y,2) );
        float d3 = sqrt(pow(p.x - intermediate_vectors[2].x,2) + pow(p.y - intermediate_vectors[2].y,2) );

        if(d1 < d2 && d1 < d3)
        {
            result.push_back(intermediate_vectors[0]);

            if(d2 < d3)
            {
                result.push_back(intermediate_vectors[1]);
                result.push_back(intermediate_vectors[2]);
            }

            else
            {
                result.push_back(intermediate_vectors[2]);
                result.push_back(intermediate_vectors[1]);
            }
        }

        if(d2 < d1 && d2 < d3)
        {
            result.push_back(intermediate_vectors[1]);

            if(d1 < d3)
            {
                result.push_back(intermediate_vectors[0]);
                result.push_back(intermediate_vectors[2]);
            }

            else
            {
                result.push_back(intermediate_vectors[2]);
                result.push_back(intermediate_vectors[0]);
            }
        }

        if(d3 < d1 && d3< d2)
        {
            result.push_back(intermediate_vectors[2]);

            if(d1 < d2)
            {
                result.push_back(intermediate_vectors[0]);
                result.push_back(intermediate_vectors[1]);
            }

            else
            {
                result.push_back(intermediate_vectors[1]);
                result.push_back(intermediate_vectors[0]);
            }
        }

    }

    if(intermediate_vectors.size() == 2)
    {
        float d1 = sqrt(pow(p.x - intermediate_vectors[0].x,2) + pow(p.y - intermediate_vectors[0].y,2) );
        float d2 = sqrt(pow(p.x - intermediate_vectors[1].x,2) + pow(p.y - intermediate_vectors[1].y,2) );

        if(d1 < d2)
        {
            result.push_back(intermediate_vectors[0]);
            result.push_back(intermediate_vectors[1]);
        }

        else
        {
            result.push_back(intermediate_vectors[1]);
            result.push_back(intermediate_vectors[0]);
        }

    }

    return result;

}
vector<Point2f> get_intermediate_points(Point p1, Point p2, vector<Point2f> good_ellipses)
{
    vector<Point2f> intermediate_vectors;
    float m = (p2.y - p1.y)/(p2.x - p1.x + 0.0001);
    float y_eq, x_eq;
    int holgura = 5;

    for(uint i=0; i<good_ellipses.size(); i++)
    {
        if( !(
                (abs(good_ellipses[i].x-p1.x)<1 && abs(good_ellipses[i].y - p1.y)<1)
                || (abs(good_ellipses[i].x - p2.x)<1 && abs(good_ellipses[i].y - p2.y)<1)
                    ))
        {
            if(abs(good_ellipses[i].x - p1.x) > abs(good_ellipses[i].y - p1.y))
            {
                y_eq = m*(good_ellipses[i].x - p1.x) + p1.y;

                if(abs(y_eq - good_ellipses[i].y) <= holgura)
                {
                    intermediate_vectors.push_back(good_ellipses[i]);
                }

            }
            else
            {
                x_eq = (good_ellipses[i].y - p1.y)/m + p1.x;

                if(abs(x_eq - good_ellipses[i].x) <= holgura)
                {
                    intermediate_vectors.push_back(good_ellipses[i]);
                }

            }
        }
    }
    return intermediate_vectors;
}


vector<Point2f> ellipses_order20(vector<Point2f> good_ellipses)
{
    vector<Point2f> convex_hull;
    vector<Point2f> sorted_ellipses;


    convexHull(good_ellipses, convex_hull, true);
    convex_hull.push_back(convex_hull[0]);
    convex_hull.push_back(convex_hull[1]);

    float x_s = convex_hull[0].x;
    float y_s = convex_hull[0].y;
    float m = (convex_hull[1].y - y_s)/(convex_hull[1].x - x_s + 0.0001);
    float holgura = 5;
    float y_eq = 0;
    float x_eq = 0;

    for(uint i=1; i<convex_hull.size(); i++)
    {
        if(abs(convex_hull[i].x - x_s) > abs(convex_hull[i].y - y_s))
        {
            y_eq = m*(convex_hull[i].x - x_s) + y_s;

            if(abs(y_eq - convex_hull[i].y) >= holgura)
            {
                sorted_ellipses.push_back(convex_hull[i-1]);
                x_s = convex_hull[i-1].x;
                y_s = convex_hull[i-1].y;
                m = (convex_hull[i].y - y_s)/(convex_hull[i].x - x_s + 0.0001);
            }

        }
        else
        {
            x_eq = (convex_hull[i].y - y_s)/m + x_s;

            if(abs(x_eq - convex_hull[i].x) >= holgura)
            {
                sorted_ellipses.push_back(convex_hull[i-1]);
                x_s = convex_hull[i-1].x;
                y_s = convex_hull[i-1].y;
                m = (convex_hull[i].y - y_s)/(convex_hull[i].x - x_s + 0.0001);
            }

        }


    }
    //cout<<sorted_ellipses.size()<<endl;
    x_s = sorted_ellipses[0].x;
    y_s = sorted_ellipses[0].y;
    m = (sorted_ellipses[1].y - y_s)/(sorted_ellipses[1].x - x_s + 0.0001);

    int count_dist1 = 0;
    int count_dist2 = 0;

    for(uint i=0; i<good_ellipses.size(); i++)
    {
        if(abs(good_ellipses[i].x - x_s) > abs(good_ellipses[i].y - y_s))
        {
            y_eq = m*(good_ellipses[i].x - x_s) + y_s;

            if(abs(y_eq - good_ellipses[i].y) >= holgura)
            {
                count_dist1++;
            }

        }
        else
        {
            x_eq = (good_ellipses[i].y - y_s)/m + x_s;

            if(abs(x_eq - good_ellipses[i].x) >= holgura)
            {
                count_dist1++;
            }

        }
    }




    x_s = sorted_ellipses[1].x;
    y_s = sorted_ellipses[1].y;
    m = (sorted_ellipses[2].y - y_s)/(sorted_ellipses[2].x - x_s + 0.0001);

    for(uint i=0; i<good_ellipses.size(); i++)
    {
        if(abs(good_ellipses[i].x - x_s) > abs(good_ellipses[i].y - y_s))
        {
            y_eq = m*(good_ellipses[i].x - x_s) + y_s;

            if(abs(y_eq - good_ellipses[i].y) >= holgura)
            {
                count_dist2++;
            }

        }
        else
        {
            x_eq = (good_ellipses[i].y - y_s)/m + x_s;

            if(abs(x_eq - good_ellipses[i].x) >= holgura)
            {
                count_dist2++;
            }

        }
    }


    if(count_dist1 < count_dist2)
    {
        Point p = sorted_ellipses[2];
        sorted_ellipses[2] = sorted_ellipses[3];
        sorted_ellipses[3] = p;
        //cout<<"FORMA AMPLIA"<<endl;
    }
    else
    {
        vector<Point2f> sorted_ellipses2;
        sorted_ellipses2.push_back(sorted_ellipses[3]);
        sorted_ellipses2.push_back(sorted_ellipses[0]);
        sorted_ellipses2.push_back(sorted_ellipses[2]);
        sorted_ellipses2.push_back(sorted_ellipses[1]);
        //cout<<"FORMA ALTA"<<endl;
        sorted_ellipses = sorted_ellipses2;

    }

    //============================================FUNCTION, HOW?============================================================
    vector<Point2f> sorted_ellipses_final(20);

    //Adding first row 0 to 4
    vector<Point2f> intermediate_vectors = get_intermediate_points(sorted_ellipses[0], sorted_ellipses[1], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses[0], intermediate_vectors); // consider the most near to sorted

    sorted_ellipses_final[0] = sorted_ellipses[0];

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[i+1] = intermediate_vectors[i];
    }
    sorted_ellipses_final[4] = sorted_ellipses[1];




    //Adding first row 5, 10, 15
    intermediate_vectors = get_intermediate_points(sorted_ellipses[0], sorted_ellipses[2], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses[0], intermediate_vectors); // consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[(i+1)*5] = intermediate_vectors[i];
    }
    sorted_ellipses_final[15] = sorted_ellipses[2];


    //Adding first row 16 to 19
    intermediate_vectors = get_intermediate_points(sorted_ellipses[2], sorted_ellipses[3], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses[2], intermediate_vectors); //consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[i+16] = intermediate_vectors[i];
    }
    sorted_ellipses_final[19] = sorted_ellipses[3];



    //Adding first row 9 , 14
    intermediate_vectors = get_intermediate_points(sorted_ellipses[1], sorted_ellipses[3], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses[1], intermediate_vectors); //consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[9+(i*5)] = intermediate_vectors[i];
    }

    //Adding first row 6 to 8
    intermediate_vectors = get_intermediate_points(sorted_ellipses_final[5], sorted_ellipses_final[9], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses_final[5], intermediate_vectors); //consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[i+6] = intermediate_vectors[i];
    }

    //Adding first row 11 to 13
    intermediate_vectors = get_intermediate_points(sorted_ellipses_final[10], sorted_ellipses_final[14], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses_final[10], intermediate_vectors); //consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[i+11] = intermediate_vectors[i];
    }

    return sorted_ellipses_final;
}

vector<Point2f> ellipses_order12(vector<Point2f> good_ellipses)
{
    vector<Point2f> convex_hull;
    vector<Point2f> sorted_ellipses;


    convexHull(good_ellipses, convex_hull, true); // Get corners
    convex_hull.push_back(convex_hull[0]);  // To ensure the square, searching the change to add the last corner
    convex_hull.push_back(convex_hull[1]);

    float x_s = convex_hull[0].x;
    float y_s = convex_hull[0].y;
    float m = (convex_hull[1].y - y_s)/(convex_hull[1].x - x_s + 0.0001); //Pendiente
    float holgura = 5; // In order to say the element is inside the line (recta)
    float y_eq = 0; // Equation with respect y
    float x_eq = 0; // Equation with respect x

    for(uint i=1; i<convex_hull.size(); i++)
    {
        if(abs(convex_hull[i].x - x_s) > abs(convex_hull[i].y - y_s)) // X mayor -> calculate the normal eq in function to y
        {
            y_eq = m*(convex_hull[i].x - x_s) + y_s;

            if(abs(y_eq - convex_hull[i].y) >= holgura)
            {
                sorted_ellipses.push_back(convex_hull[i-1]);
                x_s = convex_hull[i-1].x;
                y_s = convex_hull[i-1].y;
                m = (convex_hull[i].y - y_s)/(convex_hull[i].x - x_s + 0.0001);
            }

        }
        else
        {
            x_eq = (convex_hull[i].y - y_s)/m + x_s;

            if(abs(x_eq - convex_hull[i].x) >= holgura)
            {
                sorted_ellipses.push_back(convex_hull[i-1]);
                x_s = convex_hull[i-1].x;
                y_s = convex_hull[i-1].y;
                m = (convex_hull[i].y - y_s)/(convex_hull[i].x - x_s + 0.0001);
            }

        }


    }

    //cout<<sorted_ellipses.size()<<endl;
    x_s = sorted_ellipses[0].x;
    y_s = sorted_ellipses[0].y;
    m = (sorted_ellipses[1].y - y_s)/(sorted_ellipses[1].x - x_s + 0.0001);

    int count_dist1 = 0;
    int count_dist2 = 0;

    // Count the number of points from 0 to 1 position
    for(uint i=0; i<good_ellipses.size(); i++)
    {
        if(abs(good_ellipses[i].x - x_s) > abs(good_ellipses[i].y - y_s))
        {
            y_eq = m*(good_ellipses[i].x - x_s) + y_s;

            if(abs(y_eq - good_ellipses[i].y) >= holgura)
            {
                count_dist1++;
            }

        }
        else
        {
            x_eq = (good_ellipses[i].y - y_s)/m + x_s;

            if(abs(x_eq - good_ellipses[i].x) >= holgura)
            {
                count_dist1++;
            }

        }
    }




    x_s = sorted_ellipses[1].x;
    y_s = sorted_ellipses[1].y;
    m = (sorted_ellipses[2].y - y_s)/(sorted_ellipses[2].x - x_s + 0.0001);

    // Count the number of points from 1 to 2 position
    for(uint i=0; i<good_ellipses.size(); i++)
    {
        if(abs(good_ellipses[i].x - x_s) > abs(good_ellipses[i].y - y_s))
        {
            y_eq = m*(good_ellipses[i].x - x_s) + y_s;

            if(abs(y_eq - good_ellipses[i].y) >= holgura)
            {
                count_dist2++;
            }

        }
        else
        {
            x_eq = (good_ellipses[i].y - y_s)/m + x_s;

            if(abs(x_eq - good_ellipses[i].x) >= holgura)
            {
                count_dist2++;
            }

        }
    }

    // Sort the corners
    if(count_dist1 < count_dist2)
    {
        Point p = sorted_ellipses[2];
        sorted_ellipses[2] = sorted_ellipses[3];
        sorted_ellipses[3] = p;
        //cout<<"FORMA AMPLIA"<<endl;
    }
    else
    {
        vector<Point2f> sorted_ellipses2;
        sorted_ellipses2.push_back(sorted_ellipses[3]);
        sorted_ellipses2.push_back(sorted_ellipses[0]);
        sorted_ellipses2.push_back(sorted_ellipses[2]);
        sorted_ellipses2.push_back(sorted_ellipses[1]);
        //cout<<"FORMA ALTA"<<endl;
        sorted_ellipses = sorted_ellipses2;

    }

    vector<Point2f> sorted_ellipses_final(12);

    //Adding first row 0 to 4
    vector<Point2f> intermediate_vectors = get_intermediate_points(sorted_ellipses[0], sorted_ellipses[1], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses[0], intermediate_vectors); // consider the most near to sorted

    sorted_ellipses_final[0] = sorted_ellipses[0];

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[i+1] = intermediate_vectors[i];
    }
    sorted_ellipses_final[3] = sorted_ellipses[1];


    //Adding first row 4
    intermediate_vectors = get_intermediate_points(sorted_ellipses[0], sorted_ellipses[2], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses[0], intermediate_vectors); // consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[4] = intermediate_vectors[0];
    }
    sorted_ellipses_final[8] = sorted_ellipses[2];


    //Adding first row 9, 10
    intermediate_vectors = get_intermediate_points(sorted_ellipses[2], sorted_ellipses[3], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses[2], intermediate_vectors); //consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[i+9] = intermediate_vectors[i];
    }
    sorted_ellipses_final[11] = sorted_ellipses[3];



    //Adding first row 7
    intermediate_vectors = get_intermediate_points(sorted_ellipses[1], sorted_ellipses[3], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses[1], intermediate_vectors); //consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[7] = intermediate_vectors[0];
    }

    //Adding first row 5, 6
    intermediate_vectors = get_intermediate_points(sorted_ellipses_final[4], sorted_ellipses_final[7], good_ellipses);
    intermediate_vectors  = get_intermediate_sorted_points(sorted_ellipses_final[4], intermediate_vectors); //consider the most near to sorted

    for (uint i=0; i<intermediate_vectors.size(); i++)
    {
        sorted_ellipses_final[i+5] = intermediate_vectors[i];
    }

    //=========================================================================================================================

    return sorted_ellipses_final;
}

vector<Point2f>getExtremePoints(Mat view, vector<Point2f> pointBuf, vector<Point2f> dst_vertices, int offset){

  vector<Point2f> src_vertices;
  src_vertices.push_back( Point(pointBuf[15].x, pointBuf[15].y ) );
  src_vertices.push_back( Point(pointBuf[19].x, pointBuf[19].y ) );
  src_vertices.push_back( Point(pointBuf[0 ].x, pointBuf[0 ].y ) );
  src_vertices.push_back( Point(pointBuf[4 ].x, pointBuf[4 ].y ) );

  Mat H = findHomography(src_vertices, dst_vertices);
  //Matx33f H = getPerspectiveTransform(src_vertices, dst_vertices);

  cv::Mat rotated;
  warpPerspective(view, rotated, H, rotated.size(), INTER_LINEAR, BORDER_CONSTANT);

  vector<Point2f> new_vertices;
  int width = view.cols, height = view.rows;
  new_vertices.push_back( Point(0-offset, 0-offset) );
  new_vertices.push_back( Point(width+offset, 0-offset) );
  new_vertices.push_back( Point(0-offset, height+offset) );
  new_vertices.push_back( Point(width+offset, height+offset) );


  perspectiveTransform( new_vertices, new_vertices, H.inv());


  //imshow("Fronto-Parallel-Corto", rotated);
  //waitKey(0);
  return new_vertices;
}

void ellipsePurge2(Mat &morphology, Mat &ellipses,
                  const std::vector<RotatedRect> &elipses,
                  vector<Point2f> &centers,
                  int width, int height)
{
    vector<int> ellipses_to_verify( elipses.size() );
    Point2f center;
    int counter_verify;
    float x_t0, y_t0;
    float x_t1, y_t1;
    float disX, disY;
    float sumx, sumy;

    int counter = 0;
    float holgura = 1;
    float distance;

    // Draw ellipse|
    ellipses = Mat::zeros( morphology.size(), CV_8UC3 );
    
    for( size_t i = 0; i< elipses.size()-1; i++ )
    {
        if( elipses[i].size.width < width/12 || elipses[i].size.height < width/12 ||
            elipses[i].center.x == 0 || elipses[i].center.y == 0
            
            ){}
        else
        {
            x_t0 = elipses[i].center.x;
            y_t0 = elipses[i].center.y;

            counter_verify = 1;
            sumx = x_t0;
            sumy = y_t0;

            for( size_t j = i+1; j< elipses.size()-1; j++ ){
                x_t1 = float(elipses[j].center.x);
                y_t1 = float(elipses[j].center.y);

                disX = x_t0 - x_t1;
                disY = y_t0 - y_t1;
                distance = sqrt(disX*disX + disY*disY);

                if(distance <= holgura){
                    counter++;
                    counter_verify++;
                    sumx += x_t1;
                    sumy += y_t1;
                    ellipses_to_verify[j] = -1;
                }
            }

            if(counter_verify >= 1){
                center = Point2f(sumx/counter_verify, sumy/counter_verify);
                circle(ellipses, center, 2,cv::Scalar(35,255,75), -1, 8, 0);
                ellipse( ellipses, elipses[i], Scalar(253,35,255), 2, 8 );

                // Add to ellipse list
                centers.push_back(center);
            }

        }
    }

}

bool findRingsGrid2( Mat &view, Size &boardSize, vector<Point2f> &pointBuf){
  int ellipseCount = 0;
  int width  = view.cols;
  int height = view.rows;
  vector<Vec4i> hierarchy;




/*
  Binarized Image
  ---------------
*/
  cv::Mat gray, binarized, morphology, ellipses, result;
  cvtColor(view, gray, CV_BGR2GRAY);
  GaussianBlur(gray, gray, Size(9, 9), 2, 2);
  
  

  // Binary Threshold
  //threshold(gray, binarized, 128, 255, THRESH_BINARY);
  adaptiveThreshold(gray, binarized, 255, THRESH_BINARY, THRESH_BINARY,11,3);
  //adaptiveThreshold(binarized, binarized, 255, THRESH_BINARY, THRESH_BINARY,11,3);


/*
  Morphological Transformations
  -----------------------------
*/
  cv::Mat element = getStructuringElement( MORPH_ELLIPSE, Size ( 5, 5 ),Point( 2, 2 ));
  erode ( binarized,  morphology, element );
  erode ( morphology,  morphology, element );
  //dilate( morphology, morphology, element );

  
/*
  Ellipse detection
  -----------------
*/
  std::vector<std::vector<cv::Point> > contours;
  findContours( morphology, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  std::vector<RotatedRect> minEllipse( contours.size() );

  for( size_t i = 0; i < contours.size(); i++ ){
      if( contours[i].size() > 5 ){
          minEllipse[i] = fitEllipse( Mat(contours[i]) );
      }
  }
  
  
/*
  Ellipse purge
  -------------
*/
  vector<Point2f> good_ellipses;
  ellipsePurge2(morphology,ellipses,minEllipse,
                  good_ellipses,width, height);

  
  //imshow("ellipses", ellipses);


/*
  Result
  ------
*/
  vector<Point2f> sorted_ellipses = good_ellipses;

  if(good_ellipses.size() == 20)
      sorted_ellipses = ellipses_order20(good_ellipses);

  if(good_ellipses.size() == 12)
      sorted_ellipses = ellipses_order12(good_ellipses);


  view.copyTo(result);
  /*for (size_t i = 0; i < sorted_ellipses.size(); ++i){
      circle(result, sorted_ellipses[i], 2, Scalar(253,35,255), -1, 8, 0); //-1 full circle
      putText(result,      std::to_string(i+1),
                            sorted_ellipses[i], // Coordinates
                     cv::FONT_HERSHEY_DUPLEX, // Font
                                        0.65, // Scale. 2.0 = 2x bigger
                       cv::Scalar(35,255,75), // BGR Color
                                          1); // Line Thickness (Optional)
  }*/

  if(sorted_ellipses.size() == 20 || sorted_ellipses.size() == 12)
      pointBuf = sorted_ellipses;

  // Save ellipse count
  ellipseCount = int(good_ellipses.size());

  for(int i=0; i<pointBuf.size(); i++){
    if(pointBuf[i].x == 0 && pointBuf[i].y == 0){
      pointBuf.clear();
      return false;
    }
  }
  if(pointBuf.size() == 12 || pointBuf.size() == 20)
      return true;
  return false;
}

// RINGS DETECTION
bool findRingsGrid1( Mat& frame, Size boardSize, vector<Point2f> &pointBuf){
  Mat gray, gaussian, thres, adapThres, intThres;
  Mat erodil, ellipses;

  // rect
  RotatedRect minRect = RotatedRect(Point(frame.rows,         0),
                              Point(         0,         0),
                              Point(         0, frame.cols));

  // ellipses count
  int ellipseCount = 0;
  int width  = frame.cols;
  int height = frame.rows;
  // hieararchy
  vector<Vec4i> hierarchy;
  // total of patterns to detect
  int numPatterns = boardSize.width*boardSize.height;
  // contours
  vector<vector<Point> > contours;

  // RGB to GRAY
  cvtColor(frame, gray, CV_BGR2GRAY);
  // Gaussian Filter
  GaussianBlur(gray, gaussian, Size(5,5), 0, 0);
  // Threshold
  threshold(gaussian, thres, 100, 255, CV_THRESH_BINARY);
  // Adaptive Threshold
  adaptiveThreshold(gaussian, adapThres, 255,ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,13,12);
  // Integral Threshold
  intThres = integralThreshold(gaussian, 10);

  Mat element = getStructuringElement( MORPH_ELLIPSE, Size ( 5, 5 ),Point( 2, 2 ));
  erode ( intThres,  intThres, element );
  dilate( intThres, erodil, element );

  findContours( erodil, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  vector<RotatedRect> minEllipse( contours.size() );

  for( size_t i = 0; i < contours.size(); i++ ){
      if( contours[i].size() > 5 ){
          minEllipse[i] = fitEllipse( Mat(contours[i]) );
      }
  }

  //Ellipse purge
  vector<Point2f> good_ellipses;
  ellipsePurge(intThres, minEllipse, good_ellipses, width, height);

  auto it = remove_if(good_ellipses.begin(),
                           good_ellipses.  end(),
                           [&minRect](Point2f &p){
                              return DoesntRectangleContainPoint(minRect,p);
                          });
  good_ellipses.erase(it,good_ellipses.end());

  // Update ROI
  if(good_ellipses.size() > 16){
      minRect = minAreaRect( Mat(good_ellipses) );
      minRect.size.width  = minRect.size.width *1.15f;
      minRect.size.height = minRect.size.height*1.15f;
  } else if(good_ellipses.size() > 10){
      minRect = minAreaRect( Mat(good_ellipses) );
      minRect.size.width  = minRect.size.width *1.5f;
      minRect.size.height = minRect.size.height*1.5f;
  } else {
      minRect = RotatedRect(Point(frame.cols,         0),
                            Point(         0,         0),
                            Point(         0,frame.rows));
  }

  vector<Point2f> sorted_ellipses = good_ellipses;

  if(good_ellipses.size() == numPatterns)
    if(numPatterns == 12)
      sorted_ellipses = ellipses_order12(good_ellipses);
    if(numPatterns == 20)
      sorted_ellipses = ellipses_order20(good_ellipses);

  if(sorted_ellipses.size() == 20 || sorted_ellipses.size() == 12)
      pointBuf = sorted_ellipses;

  // Save ellipse count
  ellipseCount = int(good_ellipses.size());

  for(int i=0; i<pointBuf.size(); i++){
    if(pointBuf[i].x == 0 && pointBuf[i].y == 0){
      pointBuf.clear();
      return false;
    }
  }

  //playVideo(gray, "Gray Scale", WIDTH, HEIGHT);
  //playVideo(gaussian, "Gaussian Filter", WIDTH, HEIGHT);
  //playVideo(thres, "Threshold", WIDTH, HEIGHT);
  //playVideo(adapThres, "Adaptive Threshold", WIDTH, HEIGHT);
  //playVideo(intThres, "Integral Threshold", WIDTH, HEIGHT);

  //rectangle(frame, Point2f(pointBuf[15].x-55, pointBuf[15].y-55), Point2f(pointBuf[4].x+55,pointBuf[4].y+55), PURPLE, 5, 8);
  for (size_t i = 0; i < sorted_ellipses.size(); ++i){
    circle(frame, sorted_ellipses[i], 2, SKYBLUE, -1, 8, 0);
    //putText(frame, to_string(i+1), sorted_ellipses[i], FONT_HERSHEY_DUPLEX, 0.65, SOFTGREEN,  4); // Line Thickness (Optional)
  }
    
  return (pointBuf.size() == numPatterns);
}
