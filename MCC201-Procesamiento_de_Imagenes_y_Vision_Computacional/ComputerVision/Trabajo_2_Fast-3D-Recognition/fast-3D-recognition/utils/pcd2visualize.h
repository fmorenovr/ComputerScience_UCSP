#include "norm2vfh.h"

void viewerOneOff (pcl::visualization::PCLVisualizer& viewer){
  /*viewer.setBackgroundColor (1.0, 0.5, 1.0);
  pcl::PointXYZ o;
  o.x = 1.0;
  o.y = 0;
  o.z = 0;
  viewer.addSphere (o, 0.25, "sphere", 0);*/
  cout << "i only run once" << endl;
}
    
void viewerPsycho (pcl::visualization::PCLVisualizer& viewer){
  static unsigned count = 0;
  stringstream ss;
  ss << "Once per viewer loop: " << count++;
  viewer.removeShape ("text", 0);
  viewer.addText (ss.str(), 200, 300, "text", 0);
  
  //FIXME: possible race condition here:
  //user_data++;
}

void visualizePCD(const string &filename, string name){
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::io::loadPCDFile (filename, *cloud);
  pcl::visualization::CloudViewer viewer(name);
  //blocks until the cloud is actually rendered
  viewer.showCloud(cloud);
  //This will only get called once
  viewer.runOnVisualizationThreadOnce (viewerOneOff);
  //This will get called once per visualization iteration
  //viewer.runOnVisualizationThread (viewerPsycho);
  //while (!viewer.wasStopped ()){
    //you can also do cool processing here
    //FIXME: Note that this is running in a separate thread from viewerPsycho
    //and you should guard against race conditions yourself...
    //user_data++;
  //}
}
