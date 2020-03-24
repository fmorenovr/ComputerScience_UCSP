#include "kinect2grabber.h"

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include <pcl/io/io.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

void printHelp (int, char **argv){
  print_error ("Syntax is: %s [-format 0|1] input.ply output.pcd\n", argv[0]);
}

bool loadCloudPLY (const string &filename, pcl::PCLPointCloud2 &cloud) {
  TicToc tt;
  tt.tic ();
  print_highlight ("Loading PLY file: "); print_value ("%s ", filename.c_str ());

  pcl::PLYReader reader;
  if (reader.read (filename, cloud) < 0)
    return (false);
    
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());
  return (true);
}

void saveCloudPLY (const string &filename, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, bool binary, bool use_camera){
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving PLY file: "); print_value ("%s ", filename.c_str ());
  
  pcl::PLYWriter writer;
  writer.write(filename, cloud, binary, use_camera);
  
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
}

bool loadCloudPCD (const string &filename, pcl::PCLPointCloud2 &cloud, Eigen::Vector4f &translation, Eigen::Quaternionf &orientation){
  TicToc tt;
  tt.tic ();
  print_highlight ("Loading PCD file: "); print_value ("%s ", filename.c_str ());

  if (loadPCDFile (filename, cloud, translation, orientation) < 0)
    return (false);
  
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());
  return (true);
}

void saveCloudPCD (const string &filename, const pcl::PCLPointCloud2 &cloud, const Eigen::Vector4f &translation, const Eigen::Quaternionf &orientation, bool format){
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving PCD file: "); print_value ("%s ", filename.c_str ());
  
  pcl::PCDWriter writer;
  writer.write (filename, cloud, translation, orientation, format);
  
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
}

bool PLY2PCD (const string &plyFilename, const string &pcdFilename, bool format) {
  print_info("Convert a PLY file to PCD format.\n");
  print_info("Syntax is: input.ply output.pcd 1\n");
  print_info ("PCD output format: "); print_value ("%s\n", (format ? "binary" : "ascii"));

  // Load the first file
  pcl::PCLPointCloud2 cloud;
  if (!loadCloudPLY(plyFilename, cloud)){
    cout << "Error saving pcd file." << endl;
    return false;
  }
  saveCloudPCD(pcdFilename, cloud, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), format);
  return true;
}
