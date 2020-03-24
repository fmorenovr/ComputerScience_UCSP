#include "pclgrabber.h"

#ifdef WITH_SERIALIZATION
#include "serialization.h"
#endif

int main(int argc, char * argv[]){
  cout << "Syntax is: " << argv[0] << " [-processor 0|1|2] -processor options 0,1,2,3 correspond to CPU, OPENCL, OPENGL, CUDA respectively\n";
  cout << "Press \'s\' to store a cloud" << endl;
  cout << "Press \'x\' to store the calibrations." << endl;
#ifdef WITH_SERIALIZATION
  cout << "Press \'z\' to start/stop serialization." << endl;
#endif
  Processor freenectprocessor = OPENGL;
  vector<int> ply_file_indices;

  if(argc > 1){
    freenectprocessor = static_cast<Processor>(atoi(argv[1]));
  }
    
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud;
  K2G k2g(freenectprocessor);
  cout << "getting cloud" << endl;
  cloud = k2g.getCloud();

  k2g.printParameters();

  cloud->sensor_orientation_.w() = 0.0;
  cloud->sensor_orientation_.x() = 1.0;
  cloud->sensor_orientation_.y() = 0.0;
  cloud->sensor_orientation_.z() = 0.0;

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

  PlySaver ps(cloud, false, false, k2g);
  viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&ps);

  cv::Mat color, depth, ir;
  bool showImages = false;

  while(!viewer->wasStopped()){

    viewer->spinOnce ();
    chrono::high_resolution_clock::time_point tnow = chrono::high_resolution_clock::now();

    k2g.get(color, depth, ir, cloud);
    // Showing only color since depth is float and needs conversion
    
    /* =================================
                 Resize
    ===================================*/
    
    cv::resize(color, color, newsz);
    cv::resize(depth, depth, newsz);
    cv::resize(ir, ir, newsz);
    
    /* =================================
               Showing Images
    ===================================*/
    
    if(showImages){
      cv::imshow("RGB", color);
      cv::imshow("Depth", depth/ 4096.0f);
      cv::imshow("Infra Rojo", ir/ 4096.0f);
    }
    
    /* =================================
         Waiting keys to do something
    ===================================*/
    
    int c = cv::waitKey(1);
    
    chrono::high_resolution_clock::time_point tpost = chrono::high_resolution_clock::now();
    cout << "delta " << chrono::duration_cast<chrono::duration<double>>(tpost-tnow).count() * 1000 << endl;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->updatePointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");    	
  }

  k2g.shutDown();
  return 0;
}

